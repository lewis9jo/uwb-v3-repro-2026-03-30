"""
Generate Fig6 (CDF) and Fig8 (t-SNE) for classroom:
- V3 (Proposed_V2_NoAttention, seed 44) vs LSTM / FC-SVM / EKF errors.
- Shared trunk 160D t-SNE for V3.
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


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]

from new_bench_03.data.provider import DataProvider
from new_bench_03.models.proposed.variants import ProposedV2NoAttentionModel


CHECKPOINT_PATH = REPO_ROOT / Path(
    "new_bench_03/2_models/proposed/checkpoints/ProposedV2NoAttentionModel/proposed_best_c3.pth"
)
PLOT_DIR = RESULT_ROOT / "last_dance" / "v3_clean_run" / "plots"
CLASS_NAMES: Dict[int, str] = {0: "LOS", 1: "Static NLOS", 2: "Dynamic NLOS"}
CLASS_COLORS: Dict[int, str] = {0: "#4472C4", 1: "#ED7D31", 2: "#70AD47"}
MODEL_COLORS: Dict[str, str] = {
    "Proposed_V3": "#E94F37",
    "LSTM": "#4472C4",
    "FC-SVM": "#70AD47",
    "EKF": "#7F7F7F",
}

# Baseline error files (cm) for seed 44 classroom
BASELINE_ERRORS = {
    "Proposed_V3": _first_existing(
        RESULT_ROOT / "last_dance" / "last_dance" / "v3_clean_run" / "errors" / "Proposed_V2_NoAttention_seed44_errors.npy",
        REPO_ROOT / "last_dance" / "last_dance" / "v3_clean_run" / "errors" / "Proposed_V2_NoAttention_seed44_errors.npy",
    ),
    "LSTM": _first_existing(
        RESULT_ROOT / "last_dance" / "last_dance" / "new_results_classroom_only" / "errors" / "LSTM_seed44_errors.npy",
        REPO_ROOT / "last_dance" / "last_dance" / "new_results_classroom_only" / "errors" / "LSTM_seed44_errors.npy",
    ),
    "FC-SVM": _first_existing(
        RESULT_ROOT / "last_dance" / "last_dance" / "new_results_classroom_only" / "errors" / "FC-SVM_seed44_errors.npy",
        REPO_ROOT / "last_dance" / "last_dance" / "new_results_classroom_only" / "errors" / "FC-SVM_seed44_errors.npy",
    ),
    "EKF": _first_existing(
        RESULT_ROOT / "last_dance" / "last_dance" / "new_results_classroom_only" / "errors" / "EKF_seed44_errors.npy",
        REPO_ROOT / "last_dance" / "last_dance" / "new_results_classroom_only" / "errors" / "EKF_seed44_errors.npy",
    ),
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    provider = DataProvider(cache_name="dataset_cache_classroom", dataset_mode="classroom")
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


def load_errors() -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, path in BASELINE_ERRORS.items():
        arr = np.load(path)
        out[name] = arr.astype(float)
    return out


def plot_cdf(errors_map: Dict[str, np.ndarray], save_path: Path) -> Dict[str, Tuple[float, float]]:
    plt.figure(figsize=(3.5, 2.8))
    stats: Dict[str, Tuple[float, float]] = {}

    def _plot_one(err: np.ndarray, label: str, color: str):
        sorted_e = np.sort(err)
        cdf = np.linspace(0, 1, len(sorted_e), endpoint=True)
        plt.plot(sorted_e, cdf, label=label, color=color, linewidth=1.2)
        stats[label] = (float(np.percentile(err, 50)), float(np.percentile(err, 95)))

    for name in ["Proposed_V3", "LSTM", "FC-SVM", "EKF"]:
        arr = errors_map[name]
        _plot_one(arr, name, MODEL_COLORS.get(name, None))

    cep50, cep95 = stats["Proposed_V3"]
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
    return stats


def _subsample_tsne(features: np.ndarray, labels: np.ndarray, max_points: int = 6000, per_class: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    idx: List[int] = []
    for cls in np.unique(labels):
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
    np.save(PLOT_DIR / "v3_classroom_errors_cm.npy", errors_cm)
    np.save(PLOT_DIR / "v3_classroom_features.npy", shared)
    np.save(PLOT_DIR / "v3_classroom_labels.npy", y_cls)

    errors_map = load_errors()
    errors_map["Proposed_V3"] = errors_cm  # override with freshly computed
    stats = plot_cdf(errors_map, PLOT_DIR / "v3_classroom_cdf")
    plot_tsne(shared, y_cls, PLOT_DIR / "v3_classroom_tsne")

    print("CDF saved:", PLOT_DIR / "v3_classroom_cdf.png")
    print("t-SNE saved:", PLOT_DIR / "v3_classroom_tsne.png")
    print("CEP50/CEP95 (Proposed_V3):", stats["Proposed_V3"])


if __name__ == "__main__":
    main()
