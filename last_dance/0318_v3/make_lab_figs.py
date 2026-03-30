"""
Generate lab-set CDF (Figure 6) and t-SNE (Figure 8) for Proposed_V2_NoAttention (v3).
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "new_bench_03").exists():
            return candidate
    raise FileNotFoundError("new_bench_03 module root not found.")


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if SCRIPT_ROOT.name == "2026-03-30-v3":
    RESULT_ROOT = SCRIPT_ROOT
else:
    RESULT_ROOT = REPO_ROOT / "2026-03-30-v3"

from new_bench_03.data.provider import DataProvider
from new_bench_03.models.proposed.variants import ProposedV2NoAttentionModel


CHECKPOINT_PATH = REPO_ROOT / Path(
    "new_bench_03/2_models/proposed/checkpoints/ProposedV2NoAttentionModel/proposed_best_c2.pth"
)
PLOT_DIR = RESULT_ROOT / "last_dance" / "0318_v3" / "outputs" / "lab"
CLASS_NAMES: Dict[int, str] = {0: "LOS", 1: "Static NLOS"}
CLASS_COLORS: Dict[int, str] = {0: "#4472C4", 1: "#ED7D31"}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    provider = DataProvider(cache_name="dataset_cache_lab", dataset_mode="lab")
    bundle = provider.prepare_once(force=False)
    return bundle.test.X, bundle.test.y_pos, bundle.test.y_class


def load_model(input_dim: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = ProposedV2NoAttentionModel(input_dim=input_dim, num_classes=num_classes)
    state = model._init_state()  # type: ignore[attr-defined]
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state.model.load_state_dict(checkpoint)
    state.model.to(device)
    state.model.eval()
    model.artifacts = state  # type: ignore[attr-defined]
    return state.model


def run_inference(
    model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    shared_feats: list[torch.Tensor] = []

    def _hook(_module, _inp, out):
        shared_feats.append(out.detach().cpu())

    hook = model.shared.register_forward_hook(_hook)  # type: ignore[attr-defined]

    pos_list: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            pos, *_ = model(batch_X)
            pos_list.append(pos.detach().cpu())
    hook.remove()
    pred_pos = torch.cat(pos_list, dim=0).numpy()
    shared = torch.cat(shared_feats, dim=0).numpy()
    return pred_pos, shared


def plot_cdf(errors_cm: np.ndarray, save_path: Path) -> Tuple[float, float]:
    errors_sorted = np.sort(errors_cm)
    cdf = np.linspace(0, 1, len(errors_sorted), endpoint=True)
    cep50 = float(np.percentile(errors_cm, 50))
    cep95 = float(np.percentile(errors_cm, 95))

    plt.figure(figsize=(3.5, 2.8))
    plt.plot(errors_sorted, cdf, label="Proposed_V3 (NoAttention)", color="#E94F37", linewidth=1.2)
    plt.axvline(cep50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axvline(cep95, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axhline(0.95, color="#aaaaaa", linestyle=":", linewidth=0.8, alpha=0.7)
    plt.text(cep50 * 1.02, 0.52, "CEP50", fontsize=7, color="#555555")
    plt.text(cep95 * 1.02, 0.97, "CEP95", fontsize=7, color="#555555")
    plt.xlabel("Position Error (cm)")
    plt.ylabel("Cumulative Probability")
    plt.xlim(0, 50)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    return cep50, cep95


def _subsample_tsne(features: np.ndarray, labels: np.ndarray, max_points: int = 6000, per_class: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    idx: List[int] = []
    classes = np.unique(labels)
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        take = min(per_class, len(cls_idx))
        idx.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    idx = np.array(idx)
    if len(idx) > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    return features[idx], labels[idx]


def plot_tsne(shared_feats: np.ndarray, y_cls: np.ndarray, save_path: Path) -> None:
    feats, labels = _subsample_tsne(shared_feats, y_cls)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0, init="random")
    emb = tsne.fit_transform(feats)

    plt.figure(figsize=(3.5, 3.0))
    for cls, name in CLASS_NAMES.items():
        mask = labels == cls
        if not np.any(mask):
            continue
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=8,
            alpha=0.7,
            label=name,
            color=CLASS_COLORS.get(cls),
        )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.xticks([])
    plt.yticks([])
    plt.legend(markerscale=1.5, fontsize=7)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    device = _device()
    X, y_pos, y_cls = load_data()
    num_classes = int(np.max(y_cls) + 1)
    model = load_model(input_dim=X.shape[1], num_classes=num_classes, device=device)

    pred_pos, shared = run_inference(model, X, device)
    errors_cm = np.linalg.norm(pred_pos - y_pos, axis=1) * 100.0

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PLOT_DIR / "v3_lab_errors_cm.npy", errors_cm)
    cep50, cep95 = plot_cdf(errors_cm, PLOT_DIR / "v3_lab_cdf")
    plot_tsne(shared, y_cls, PLOT_DIR / "v3_lab_tsne")

    print(f"Lab CDF saved (CEP50={cep50:.2f} cm, CEP95={cep95:.2f} cm)")
    print("t-SNE saved.")


if __name__ == "__main__":
    main()
