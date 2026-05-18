# contrastive_learning.py - GraphContrastiveLearning (MoCo + Cohesive Subgraph Augmentation + BOTH option)
import random
from typing import List, Optional, Tuple, Dict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphContrastiveLearning(nn.Module):
    """
    Graph contrastive learning module:
    - MoCo-style cross-graph contrast that supports batch_size=1 using a history queue as negatives
    - Cohesive subgraph augmentation using k-core or k-truss
    - cohesion_property supports: 'kcore', 'ktruss', 'both' (dual-path: compute both losses and sum/avg)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        temperature: float = 0.2,
        use_ogsn: bool = False,
        use_distill: bool = False,
        device=None,
        memory_size: int = 180,
        cohesion_property: str = "kcore",
        augmentation_type: str = "probabilistic",
        drop_prob: float = 0.2,
        decay_factor: float = 0.2,
        both_reduce: str = "mean",  # 'mean' (recommended) or 'sum'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.use_ogsn = use_ogsn
        self.use_distill = use_distill
        self.memory_size = memory_size

        # cohesive augmentation params
        self.cohesion_property = cohesion_property  # kcore / ktruss / both
        self.augmentation_type = augmentation_type  # probabilistic / deterministic
        self.drop_prob = drop_prob
        self.decay_factor = decay_factor
        self.both_reduce = both_reduce

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # optional O-GSN head (kept for compatibility; only used if you pass graph_ids elsewhere)
        if self.use_ogsn:
            self.ogsn_projection = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.device = device

    # ------------------------- Graph utilities -------------------------

    @staticmethod
    def _build_nx_graph(adj: torch.Tensor) -> nx.Graph:
        """Build an undirected networkx graph from adjacency (>0 as edge)."""
        a = adj.detach().float()
        if a.dim() != 2:
            raise ValueError(f"adj must be 2D [N,N], got {a.shape}")
        a = (a > 0).cpu().numpy().astype(np.int32)
        g = nx.Graph()
        n = a.shape[0]
        g.add_nodes_from(range(n))
        rows, cols = np.where(a > 0)
        edges = [(int(r), int(c)) for r, c in zip(rows, cols) if r != c]
        g.add_edges_from(edges)
        return g

    @staticmethod
    def _max_k_truss_subgraph(g: nx.Graph) -> Optional[nx.Graph]:
        """
        Find the highest-order non-empty k-truss subgraph.
        For sparse graphs, can be empty -> return None.
        """
        if g.number_of_edges() == 0:
            return None
        best = None
        for k in range(2, 50):  # safe cap
            try:
                sg = nx.k_truss(g, k)
            except Exception:
                sg = None
            if sg is None or sg.number_of_edges() == 0 or sg.number_of_nodes() == 0:
                break
            best = sg
        return best

    def cohesion_nodes(self, adj: torch.Tensor, cohesion_property: Optional[str] = None) -> torch.Tensor:
        """
        Return a boolean mask [N] for cohesive nodes, using either 'kcore' or 'ktruss'.
        NOTE: when self.cohesion_property == 'both', you must pass cohesion_property explicitly.
        """
        prop = cohesion_property if cohesion_property is not None else self.cohesion_property
        if prop not in ("kcore", "ktruss"):
            raise ValueError(f"cohesion_nodes expects 'kcore' or 'ktruss', got: {prop}")

        g = self._build_nx_graph(adj)
        n = g.number_of_nodes()
        if n == 0:
            return torch.zeros(adj.size(0), device=adj.device, dtype=torch.bool)

        if prop == "kcore":
            core_num = nx.core_number(g) if n > 0 else {}
            if not core_num:
                return torch.zeros(n, device=adj.device, dtype=torch.bool)
            max_core = max(core_num.values())
            keep = [False] * n
            for node, cn in core_num.items():
                if cn == max_core:
                    keep[int(node)] = True
            return torch.tensor(keep, device=adj.device, dtype=torch.bool)

        # prop == 'ktruss'
        subg = self._max_k_truss_subgraph(g)
        keep = [False] * n
        if subg is not None and subg.number_of_nodes() > 0:
            for node in subg.nodes():
                keep[int(node)] = True
        return torch.tensor(keep, device=adj.device, dtype=torch.bool)

    # ------------------------- Cohesive augmentation -------------------------

    def augment_nodes_mask(self, cohesion_mask: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean keep-mask [N] for nodes.
        - deterministic: keep only cohesion nodes (fallback keep one node if all false)
        - probabilistic: drop with p=drop_prob for non-cohesive, p=drop_prob*decay_factor for cohesive
        """
        n = cohesion_mask.numel()
        device = cohesion_mask.device

        if self.augmentation_type == "deterministic":
            keep = cohesion_mask.clone()
            if keep.sum() == 0:
                keep[random.randrange(n)] = True
            return keep

        # probabilistic
        keep = torch.ones(n, device=device, dtype=torch.bool)
        p = torch.full((n,), float(self.drop_prob), device=device)
        p = torch.where(cohesion_mask, p * float(self.decay_factor), p)
        drop = torch.rand(n, device=device) < p
        keep = keep & (~drop)

        if keep.sum() == 0:
            idxs = torch.where(cohesion_mask)[0]
            if idxs.numel() > 0:
                keep[int(idxs[random.randrange(idxs.numel())])] = True
            else:
                keep[random.randrange(n)] = True
        return keep

    @staticmethod
    def pool_graph(node_embs: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked mean pooling.
        node_embs: [B,N,D] or [N,D] (we will treat as B=1)
        keep_mask: [N]
        return: [B,D]
        """
        if node_embs.dim() == 2:
            node_embs = node_embs.unsqueeze(0)
        b, n, d = node_embs.shape
        m = keep_mask.view(1, n, 1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (node_embs * m).sum(dim=1) / denom
        return pooled

    def graph_two_views_with_property(self, node_embs: torch.Tensor, adj: torch.Tensor, prop: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two graph-level views using a specific cohesion property."""
        cohesion_mask = self.cohesion_nodes(adj, cohesion_property=prop)
        keep1 = self.augment_nodes_mask(cohesion_mask)
        keep2 = self.augment_nodes_mask(cohesion_mask)
        v1 = self.pool_graph(node_embs, keep1)
        v2 = self.pool_graph(node_embs, keep2)
        return v1, v2

    def graph_two_views(self, node_embs: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-property two-view generator (uses self.cohesion_property; must not be 'both')."""
        if self.cohesion_property == "both":
            raise ValueError("graph_two_views cannot be used when cohesion_property=='both'. Use graph_two_views_with_property.")
        return self.graph_two_views_with_property(node_embs, adj, self.cohesion_property)

    # ------------------------- MoCo losses -------------------------

    def cross_graph_moco_loss(
        self,
        anchor_emb: torch.Tensor,    # dialogue
        positive_emb: torch.Tensor,  # value
        negatives: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        MoCo-style: anchor vs positive, with negatives from history queue.
        anchor_emb/positive_emb: [B,D] or [D]
        negatives: list of [D] or [1,D] or [*,D] or [K,1,D]
        """
        if anchor_emb.dim() == 1:
            anchor_emb = anchor_emb.unsqueeze(0)
        if positive_emb.dim() == 1:
            positive_emb = positive_emb.unsqueeze(0)

        q = F.normalize(self.projector(anchor_emb), dim=-1)   # [B,D]
        k = F.normalize(self.projector(positive_emb), dim=-1) # [B,D]

        pos_sim = (q * k).sum(dim=-1, keepdim=True)  # [B,1]

        neg_sim = None
        if negatives:
            neg_list = []
            for h in negatives:
                if h is None:
                    continue
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                elif h.dim() == 3:
                    h = h.reshape(-1, h.size(-1))
                neg_list.append(self.projector(h))
            if neg_list:
                negs = torch.cat(neg_list, dim=0)  # [K,D]
                negs = F.normalize(negs, dim=-1)
                neg_sim = torch.matmul(q, negs.t())  # [B,K]

        logits = pos_sim
        if neg_sim is not None:
            logits = torch.cat([pos_sim, neg_sim], dim=-1)
        logits = logits / self.temperature

        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    def single_graph_moco_loss(self, current: torch.Tensor, negatives: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Embedding-level MoCo loss with dropout as augmentation (fallback path)."""
        if current.dim() == 1:
            current = current.unsqueeze(0)

        v1 = F.dropout(current, p=0.1, training=True)
        v2 = F.dropout(current, p=0.1, training=True)

        z1 = F.normalize(self.projector(v1), dim=-1)
        z2 = F.normalize(self.projector(v2), dim=-1)

        pos_sim = (z1 * z2).sum(dim=-1, keepdim=True)  # [B,1]

        neg_sim = None
        if negatives:
            neg_list = []
            for h in negatives:
                if h is None:
                    continue
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                elif h.dim() == 3:
                    h = h.reshape(-1, h.size(-1))
                neg_list.append(self.projector(h))
            if neg_list:
                hist = torch.cat(neg_list, dim=0)  # [K,D]
                hist = F.normalize(hist, dim=-1)
                neg_sim = torch.matmul(z1, hist.t())  # [B,K]

        logits = pos_sim if neg_sim is None else torch.cat([pos_sim, neg_sim], dim=-1)
        logits = logits / self.temperature
        labels = torch.zeros(z1.size(0), dtype=torch.long, device=z1.device)
        return F.cross_entropy(logits, labels)

    # ------------------------- Cohesive-view within-graph loss -------------------------

    def single_graph_view_loss(
        self,
        node_embs: torch.Tensor,
        adj: torch.Tensor,
        history_embs: Optional[List[torch.Tensor]] = None,
        cohesion_property: Optional[str] = None,  # 'kcore' or 'ktruss'
    ) -> torch.Tensor:
        """
        Within-graph contrast using two cohesive augmented views; optional MoCo negatives.
        Robust to history shapes: [D], [1,D], [K,D], [K,1,D], etc.
        """
        prop = cohesion_property if cohesion_property is not None else self.cohesion_property
        if prop == "both":
            raise ValueError("single_graph_view_loss requires a concrete property ('kcore'/'ktruss') when using BOTH.")

        v1, v2 = self.graph_two_views_with_property(node_embs, adj, prop)  # [B,D]
        v1p = F.normalize(self.projector(v1), dim=-1)
        v2p = F.normalize(self.projector(v2), dim=-1)

        pos_sim = (v1p * v2p).sum(dim=-1, keepdim=True)  # [B,1]

        if history_embs:
            neg_list = []
            for h in history_embs:
                if h is None:
                    continue
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                elif h.dim() == 3:
                    h = h.reshape(-1, h.size(-1))
                else:
                    h = h.reshape(-1, h.size(-1))
                neg_list.append(self.projector(h))

            if neg_list:
                negs = torch.cat(neg_list, dim=0)  # [K,D]
                negs = F.normalize(negs, dim=-1)
                neg_sim = torch.matmul(v1p, negs.t())  # [B,K]
                logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature
                labels = torch.zeros(v1p.size(0), dtype=torch.long, device=v1p.device)
                return F.cross_entropy(logits, labels)

        return 1 - (v1p * v2p).sum(dim=-1).mean()

    # ------------------------- Main API -------------------------

    def multi_graph_contrast(
        self,
        value_embeddings: torch.Tensor,
        dialogue_embeddings: torch.Tensor,
        value_history: Optional[List[torch.Tensor]] = None,
        dialogue_history: Optional[List[torch.Tensor]] = None,
        value_node_embs: Optional[torch.Tensor] = None,
        value_adj: Optional[torch.Tensor] = None,
        dialogue_node_embs: Optional[torch.Tensor] = None,
        dialogue_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          contrast_loss (scalar tensor),
          details (dict of loss components)
        """
        device = value_embeddings.device
        zero = torch.tensor(0.0, device=device)

        value_history = [h for h in (value_history or []) if h is not None]
        dialogue_history = [h for h in (dialogue_history or []) if h is not None]

        def _compute_one_property(prop: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            details: Dict[str, torch.Tensor] = {}

            if (value_node_embs is not None and value_adj is not None and
                dialogue_node_embs is not None and dialogue_adj is not None):

                d1, _ = self.graph_two_views_with_property(dialogue_node_embs, dialogue_adj, prop)
                v1, _ = self.graph_two_views_with_property(value_node_embs, value_adj, prop)
                cross = self.cross_graph_moco_loss(d1, v1, value_history)
                details["cross_graph_loss"] = cross

                if len(value_history) > 0:
                    vh = self.single_graph_view_loss(value_node_embs, value_adj, value_history, cohesion_property=prop) * 0.3
                else:
                    vh = zero
                if len(dialogue_history) > 0:
                    dh = self.single_graph_view_loss(dialogue_node_embs, dialogue_adj, dialogue_history, cohesion_property=prop) * 0.3
                else:
                    dh = zero

                details["value_history_loss"] = vh
                details["dialogue_history_loss"] = dh

                total = cross + vh + dh
                details["total_contrast_loss"] = total
                return total, details

            # fallback embedding-level
            cross = self.cross_graph_moco_loss(dialogue_embeddings, value_embeddings, value_history)
            vh = self.single_graph_moco_loss(value_embeddings, value_history) * 0.3 if len(value_history) > 0 else zero
            dh = self.single_graph_moco_loss(dialogue_embeddings, dialogue_history) * 0.3 if len(dialogue_history) > 0 else zero

            total = cross + vh + dh
            details["cross_graph_loss"] = cross
            details["value_history_loss"] = vh
            details["dialogue_history_loss"] = dh
            details["total_contrast_loss"] = total
            return total, details

        if self.cohesion_property == "both":
            loss_kcore, det_kcore = _compute_one_property("kcore")
            loss_ktruss, det_ktruss = _compute_one_property("ktruss")

            total = (loss_kcore + loss_ktruss) if self.both_reduce == "sum" else 0.5 * (loss_kcore + loss_ktruss)

            details: Dict[str, torch.Tensor] = {
                "total_contrast_loss": total,
                "kcore_total": loss_kcore,
                "ktruss_total": loss_ktruss,
                "cross_graph_loss": det_kcore["cross_graph_loss"] + det_ktruss["cross_graph_loss"],
                "value_history_loss": det_kcore["value_history_loss"] + det_ktruss["value_history_loss"],
                "dialogue_history_loss": det_kcore["dialogue_history_loss"] + det_ktruss["dialogue_history_loss"],
                "kcore_cross_graph_loss": det_kcore["cross_graph_loss"],
                "kcore_value_history_loss": det_kcore["value_history_loss"],
                "kcore_dialogue_history_loss": det_kcore["dialogue_history_loss"],
                "ktruss_cross_graph_loss": det_ktruss["cross_graph_loss"],
                "ktruss_value_history_loss": det_ktruss["value_history_loss"],
                "ktruss_dialogue_history_loss": det_ktruss["dialogue_history_loss"],
            }

            if self.both_reduce != "sum":
                details["cross_graph_loss"] = 0.5 * details["cross_graph_loss"]
                details["value_history_loss"] = 0.5 * details["value_history_loss"]
                details["dialogue_history_loss"] = 0.5 * details["dialogue_history_loss"]

            return total, details

        return _compute_one_property(self.cohesion_property)

