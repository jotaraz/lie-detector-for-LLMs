"""
Novel probe architectures from Kramár et al., 2026 ("Building Production-Ready Probes For Gemini").

All probes here operate on a single layer (single int, not a list).
Specify via `probe_architecture: "<name>"` in the experiment YAML config.
"""

import logging
import pickle
from pathlib import Path
from typing import Literal, Self

import torch
import torch.nn as nn
from torch import Tensor

from deception_detection.activations import Activations
from deception_detection.detectors import Detector
from deception_detection.scores import Scores

logger = logging.getLogger(__name__)

ProbeArchitecture = Literal["ema", "mlp_paper", "attention", "multimax", "rolling_means_attention"]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _get_layer_idx(acts: Activations, layer: int) -> int:
    """Return the index into the layers dimension of acts.all_acts for the given layer number."""
    if acts.layers is None:
        return layer
    return acts.layers.index(layer)


def _split_by_sequence(acts: Activations, layer_idx: int, use_all: bool = False) -> list[Tensor]:
    """Return a list of [n_toks, emb] tensors, one per sequence.

    Uses detection_mask by default; attention_mask when use_all=True.
    """
    full_acts = acts.all_acts  # [batch, seqpos, n_layers, emb]
    mask = (
        acts.tokenized_dataset.attention_mask
        if use_all
        else acts.tokenized_dataset.detection_mask
    )
    return [full_acts[i, mask[i], layer_idx, :] for i in range(full_acts.shape[0])]


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    """Build an M-layer MLP as in Kramár et al.: A_1 · ReLU(A_2 · ReLU(... A_M · X ...)).

    depth=M is the total number of linear layers.
    """
    if depth == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# EMA Detector
# ---------------------------------------------------------------------------

class EMADetector(Detector):
    """EMA probe (Cunningham, Peng et al., 2025).

    Trains an MMS linear direction on detection tokens. At inference, computes
    an exponential moving average of per-token linear scores and returns the
    maximum as the dialogue score:

        EMA_0 = 0
        EMA_j = α * f_linear(x_{i,j}) + (1 - α) * EMA_{j-1}
        f_EMA(S_i) = max_j EMA_j
    """

    def __init__(self, layer: int, alpha: float = 0.5):
        self.layer = layer
        self.alpha = alpha
        self.direction: Tensor | None = None  # [emb]

    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        layer_idx = _get_layer_idx(positive_acts, self.layer)
        pos_flat = torch.cat(_split_by_sequence(positive_acts, layer_idx), dim=0).float()
        neg_flat = torch.cat(_split_by_sequence(negative_acts, layer_idx), dim=0).float()
        self.direction = (pos_flat.mean(0) - neg_flat.mean(0)).cpu()

    def _ema_trajectory(self, seq: Tensor) -> Tensor:
        """Compute EMA value at every token position. Returns [n_toks]."""
        assert self.direction is not None
        token_scores = seq.float().cpu() @ self.direction  # [n_toks]
        ema = torch.zeros(1)
        ema_vals: list[Tensor] = []
        for s in token_scores:
            ema = self.alpha * s + (1.0 - self.alpha) * ema
            ema_vals.append(ema.clone())
        return torch.stack(ema_vals).squeeze(-1)

    def score(self, acts: Activations, all_acts: bool = False) -> Scores:
        layer_idx = _get_layer_idx(acts, self.layer)
        seqs = _split_by_sequence(acts, layer_idx, use_all=all_acts)
        scores_by_dialogue: list[Tensor] = []
        for seq in seqs:
            if len(seq) == 0:
                scores_by_dialogue.append(torch.tensor([0.0]))
                continue
            ema_vals = self._ema_trajectory(seq)
            if all_acts:
                scores_by_dialogue.append(ema_vals)            # full trajectory
            else:
                scores_by_dialogue.append(ema_vals.max().unsqueeze(0))  # single scalar
        return Scores(scores_by_dialogue)

    def save(self, file_path: str | Path) -> None:
        with open(file_path, "wb") as f:
            pickle.dump({"layer": self.layer, "alpha": self.alpha, "direction": self.direction}, f)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        det = cls(data["layer"], data["alpha"])
        det.direction = data["direction"]
        return det


# ---------------------------------------------------------------------------
# MLP Paper Detector
# ---------------------------------------------------------------------------

class MLPPaperDetector(Detector):
    """Mean-pooled MLP probe (Kramár et al., 2026).

    Applies an M-layer MLP to each token independently, then mean-pools:

        f^M_MLP(S_i) = (1/n_i) Σ_j MLP_M(x_{i,j})

    Trained with sequence-level BCE loss.
    """

    def __init__(
        self,
        layer: int,
        hidden_dim: int = 100,
        mlp_depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 1000,
    ):
        self.layer = layer
        self.hidden_dim = hidden_dim
        self.mlp_depth = mlp_depth
        self.lr = lr
        self.epochs = epochs
        self.net: nn.Sequential | None = None
        self.input_dim: int | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        layer_idx = _get_layer_idx(positive_acts, self.layer)
        pos_seqs = _split_by_sequence(positive_acts, layer_idx)
        neg_seqs = _split_by_sequence(negative_acts, layer_idx)

        all_seqs = pos_seqs + neg_seqs
        all_labels = [1.0] * len(pos_seqs) + [0.0] * len(neg_seqs)

        self.input_dim = all_seqs[0].shape[-1]
        # MLP outputs a scalar logit per token; mean-pool gives the sequence logit
        self.net = _build_mlp(self.input_dim, self.hidden_dim, 1, self.mlp_depth).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        n = len(all_seqs)
        perm = torch.randperm(n).tolist()
        n_train = int(0.8 * n)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        def forward_batch(idx_list: list[int]) -> tuple[Tensor, Tensor]:
            seqs_dev = [all_seqs[i].to(self.device, torch.float32) for i in idx_list]
            lengths = [len(s) for s in seqs_dev]
            out = self.net(torch.cat(seqs_dev, dim=0)).squeeze(-1)  # [total_toks]
            logits = torch.stack([chunk.mean() for chunk in torch.split(out, lengths)])
            labels = torch.tensor([all_labels[i] for i in idx_list], device=self.device)
            return logits, labels

        best_val_loss = float("inf")
        check_interval = max(1, self.epochs // 20)
        for epoch in range(self.epochs):
            self.net.train()
            logits, labels = forward_batch(train_idx)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % check_interval == 0:
                self.net.eval()
                with torch.no_grad():
                    val_logits, val_labels = forward_batch(val_idx)
                    val_loss = criterion(val_logits, val_labels).item()
                logger.info(f"[MLPPaper] epoch {epoch}, val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                elif val_loss > best_val_loss:
                    logger.info(f"[MLPPaper] early stop at epoch {epoch}")
                    break

    def score(self, acts: Activations, all_acts: bool = False) -> Scores:
        assert self.net is not None
        layer_idx = _get_layer_idx(acts, self.layer)
        seqs = _split_by_sequence(acts, layer_idx, use_all=all_acts)
        self.net.eval()
        scores_by_dialogue: list[Tensor] = []
        with torch.no_grad():
            for seq in seqs:
                if len(seq) == 0:
                    scores_by_dialogue.append(torch.tensor([0.0]))
                    continue
                out = self.net(seq.to(self.device, torch.float32)).squeeze(-1)  # [n_toks]
                if all_acts:
                    scores_by_dialogue.append(out.cpu())
                else:
                    scores_by_dialogue.append(out.mean().unsqueeze(0).cpu())
        return Scores(scores_by_dialogue)

    def save(self, file_path: str | Path) -> None:
        torch.save(
            {
                "layer": self.layer,
                "hidden_dim": self.hidden_dim,
                "mlp_depth": self.mlp_depth,
                "lr": self.lr,
                "epochs": self.epochs,
                "input_dim": self.input_dim,
                "net_state_dict": self.net.state_dict() if self.net else None,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        data = torch.load(file_path)
        det = cls(data["layer"], data["hidden_dim"], data["mlp_depth"], data["lr"], data["epochs"])
        det.input_dim = data["input_dim"]
        det.net = _build_mlp(data["input_dim"], data["hidden_dim"], 1, data["mlp_depth"])
        det.net.load_state_dict(data["net_state_dict"])
        det.net.to(det.device)
        return det


# ---------------------------------------------------------------------------
# Shared network for attention-based probes
# ---------------------------------------------------------------------------

class _AttentionProbeNet(nn.Module):
    """MLP feature extractor + multi-head query/value projections.

    Used by AttentionProbeDetector, MultiMaxDetector, and RollingMeansAttentionDetector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, mlp_depth: int, n_heads: int):
        super().__init__()
        # MLP maps each token embedding to a hidden_dim feature vector
        self.mlp = _build_mlp(input_dim, hidden_dim, hidden_dim, mlp_depth)
        self.queries = nn.Parameter(torch.randn(n_heads, hidden_dim) * 0.01)
        self.values = nn.Parameter(torch.randn(n_heads, hidden_dim) * 0.01)
        self.n_heads = n_heads

    def get_features(self, x: Tensor) -> Tensor:
        """x: [n_toks, input_dim] -> [n_toks, hidden_dim]"""
        return self.mlp(x)

    def forward_softmax(self, features: Tensor) -> Tensor:
        """Softmax-aggregated score for a single sequence.

        f_Attn(S_i) = Σ_h [Σ_j softmax(q_h^T y_j) * (v_h^T y_j)]

        features: [n_toks, hidden_dim] -> scalar
        """
        q = self.queries @ features.T   # [n_heads, n_toks]
        v = self.values @ features.T    # [n_heads, n_toks]
        attn = torch.softmax(q, dim=-1) # [n_heads, n_toks]
        return (attn * v).sum()

    def forward_hardmax(self, features: Tensor) -> Tensor:
        """Hard-max-aggregated score for a single sequence.

        f_MultiMax(S_i) = Σ_h max_j (v_h^T y_j)

        features: [n_toks, hidden_dim] -> scalar
        """
        v = self.values @ features.T  # [n_heads, n_toks]
        return v.max(dim=-1).values.sum()

    def per_token_scores(self, features: Tensor) -> Tensor:
        """Attention-weighted per-token value contribution. For visualization.

        features: [n_toks, hidden_dim] -> [n_toks]
        """
        q = self.queries @ features.T   # [n_heads, n_toks]
        v = self.values @ features.T    # [n_heads, n_toks]
        attn = torch.softmax(q, dim=-1) # [n_heads, n_toks]
        return (attn * v).sum(dim=0)    # [n_toks]

    def rolling_means_score(self, features: Tensor, window_size: int) -> Tensor:
        """Sliding-window attention score, taking max over window positions.

        For each window ending at t:
            v̄_t = Σ_{j in window} softmax(q^T y_j) * v^T y_j

        f(S_i) = max_t v̄_t

        features: [n_toks, hidden_dim] -> scalar
        """
        n = features.shape[0]
        q = self.queries @ features.T  # [n_heads, n_toks]
        v = self.values @ features.T   # [n_heads, n_toks]

        window_scores: list[Tensor] = []
        for t in range(n):
            start = max(0, t - window_size + 1)
            attn = torch.softmax(q[:, start : t + 1], dim=-1)  # [n_heads, w]
            window_scores.append((attn * v[:, start : t + 1]).sum())

        return torch.stack(window_scores).max()


# ---------------------------------------------------------------------------
# Attention Probe Detector  (and base for MultiMax / RollingMeans)
# ---------------------------------------------------------------------------

class AttentionProbeDetector(Detector):
    """Multi-head attention probe (Kantamneni et al., 2025; Kramár et al., 2026).

    y_{i,j} = MLP_M(x_{i,j})
    f_Attn(S_i) = Σ_h [Σ_j softmax(q_h^T y_{i,j}) * (v_h^T y_{i,j})]

    Trained with sequence-level BCE loss. Subclasses override _inference_agg()
    to change the aggregation at inference time (e.g. hard max).
    """

    def __init__(
        self,
        layer: int,
        hidden_dim: int = 100,
        mlp_depth: int = 2,
        n_heads: int = 10,
        lr: float = 1e-3,
        epochs: int = 1000,
    ):
        self.layer = layer
        self.hidden_dim = hidden_dim
        self.mlp_depth = mlp_depth
        self.n_heads = n_heads
        self.lr = lr
        self.epochs = epochs
        self.net: _AttentionProbeNet | None = None
        self.input_dim: int | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _inference_agg(self, net: _AttentionProbeNet, features: Tensor) -> Tensor:
        """Aggregation function used at inference. Override in subclasses."""
        return net.forward_softmax(features)

    def fit(self, positive_acts: Activations, negative_acts: Activations) -> None:
        layer_idx = _get_layer_idx(positive_acts, self.layer)
        pos_seqs = _split_by_sequence(positive_acts, layer_idx)
        neg_seqs = _split_by_sequence(negative_acts, layer_idx)

        all_seqs = pos_seqs + neg_seqs
        all_labels = [1.0] * len(pos_seqs) + [0.0] * len(neg_seqs)

        self.input_dim = all_seqs[0].shape[-1]
        self.net = _AttentionProbeNet(
            self.input_dim, self.hidden_dim, self.mlp_depth, self.n_heads
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        n = len(all_seqs)
        perm = torch.randperm(n).tolist()
        n_train = int(0.8 * n)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        def forward_batch(idx_list: list[int]) -> tuple[Tensor, Tensor]:
            """Batch MLP features, then per-sequence softmax attention (always softmax in training)."""
            seqs_dev = [all_seqs[i].to(self.device, torch.float32) for i in idx_list]
            lengths = [len(s) for s in seqs_dev]
            all_feats = self.net.get_features(torch.cat(seqs_dev, dim=0))  # [total_toks, hidden]
            feat_split = torch.split(all_feats, lengths)
            # Training always uses softmax regardless of subclass inference aggregation
            logits = torch.stack([self.net.forward_softmax(f) for f in feat_split])
            labels = torch.tensor([all_labels[i] for i in idx_list], device=self.device)
            return logits, labels

        best_val_loss = float("inf")
        check_interval = max(1, self.epochs // 20)
        for epoch in range(self.epochs):
            self.net.train()
            logits, labels = forward_batch(train_idx)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % check_interval == 0:
                self.net.eval()
                with torch.no_grad():
                    val_logits, val_labels = forward_batch(val_idx)
                    val_loss = criterion(val_logits, val_labels).item()
                logger.info(f"[{type(self).__name__}] epoch {epoch}, val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                elif val_loss > best_val_loss:
                    logger.info(f"[{type(self).__name__}] early stop at epoch {epoch}")
                    break

    def score(self, acts: Activations, all_acts: bool = False) -> Scores:
        assert self.net is not None
        layer_idx = _get_layer_idx(acts, self.layer)
        seqs = _split_by_sequence(acts, layer_idx, use_all=all_acts)
        self.net.eval()
        scores_by_dialogue: list[Tensor] = []
        with torch.no_grad():
            for seq in seqs:
                if len(seq) == 0:
                    scores_by_dialogue.append(torch.tensor([0.0]))
                    continue
                feats = self.net.get_features(seq.to(self.device, torch.float32))
                if all_acts:
                    per_tok = self.net.per_token_scores(feats)
                    scores_by_dialogue.append(per_tok.cpu())
                else:
                    score = self._inference_agg(self.net, feats)
                    scores_by_dialogue.append(score.unsqueeze(0).cpu())
        return Scores(scores_by_dialogue)

    def _core_state(self) -> dict:
        return {
            "layer": self.layer,
            "hidden_dim": self.hidden_dim,
            "mlp_depth": self.mlp_depth,
            "n_heads": self.n_heads,
            "lr": self.lr,
            "epochs": self.epochs,
            "input_dim": self.input_dim,
            "net_state_dict": self.net.state_dict() if self.net else None,
        }

    def _restore_net(self, data: dict) -> None:
        self.input_dim = data["input_dim"]
        self.net = _AttentionProbeNet(
            data["input_dim"], data["hidden_dim"], data["mlp_depth"], data["n_heads"]
        )
        self.net.load_state_dict(data["net_state_dict"])
        self.net.to(self.device)

    def save(self, file_path: str | Path) -> None:
        torch.save(self._core_state(), file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        data = torch.load(file_path)
        det = cls(
            data["layer"], data["hidden_dim"], data["mlp_depth"],
            data["n_heads"], data["lr"], data["epochs"],
        )
        det._restore_net(data)
        return det


# ---------------------------------------------------------------------------
# MultiMax Detector
# ---------------------------------------------------------------------------

class MultiMaxDetector(AttentionProbeDetector):
    """MultiMax probe (Kramár et al., 2026).

    Identical training to AttentionProbeDetector (softmax). At inference,
    replaces softmax with a hard max per head:

        f_MultiMax(S_i) = Σ_h max_j (v_h^T y_{i,j})
    """

    def _inference_agg(self, net: _AttentionProbeNet, features: Tensor) -> Tensor:
        return net.forward_hardmax(features)


# ---------------------------------------------------------------------------
# Rolling Means Attention Detector
# ---------------------------------------------------------------------------

class RollingMeansAttentionDetector(AttentionProbeDetector):
    """Max of Rolling Means Attention probe (McKenzie et al., 2025 + Kramár et al., 2026).

    Identical training to AttentionProbeDetector (softmax over full sequence).
    At inference, uses attention-weighted sliding windows of width `window_size`
    and takes the maximum window score:

        v̄_t = Σ_{j=t-w+1}^{t} softmax(q^T y_j) * v^T y_j   (softmax within window)
        f(S_i) = max_t v̄_t
    """

    def __init__(
        self,
        layer: int,
        hidden_dim: int = 100,
        mlp_depth: int = 2,
        n_heads: int = 10,
        lr: float = 1e-3,
        epochs: int = 1000,
        window_size: int = 10,
    ):
        super().__init__(layer, hidden_dim, mlp_depth, n_heads, lr, epochs)
        self.window_size = window_size

    def _inference_agg(self, net: _AttentionProbeNet, features: Tensor) -> Tensor:
        return net.rolling_means_score(features, self.window_size)

    def _core_state(self) -> dict:
        return {**super()._core_state(), "window_size": self.window_size}

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        data = torch.load(file_path)
        det = cls(
            data["layer"], data["hidden_dim"], data["mlp_depth"],
            data["n_heads"], data["lr"], data["epochs"],
            window_size=data.get("window_size", 10),
        )
        det._restore_net(data)
        return det


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_probe_architecture_class(architecture: str) -> type[Detector]:
    classes: dict[str, type[Detector]] = {
        "ema": EMADetector,
        "mlp_paper": MLPPaperDetector,
        "attention": AttentionProbeDetector,
        "multimax": MultiMaxDetector,
        "rolling_means_attention": RollingMeansAttentionDetector,
    }
    if architecture not in classes:
        raise ValueError(
            f"Unknown probe_architecture: {architecture!r}. "
            f"Valid options: {list(classes)}"
        )
    return classes[architecture]
