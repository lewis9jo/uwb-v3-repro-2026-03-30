"""Single-file v3 pipeline (No-Attention) with full flow.

This script is intended to be read as a "documented notebook-like" flow:
  - data load -> cleaning -> split
  - feature profile selection
  - model build -> training
  - inference
  - evaluation + cdf / tsne save
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import os
import random
import shutil
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_bench_03.data.provider import DataProvider, SplitData
from new_bench_03.models.proposed.variants import ProposedV2NoAttentionModel
from new_bench_03.models.base import ModelOutput
from new_bench_03.output.evaluator import Evaluator


MODEL_NAME = "Proposed_V2_NoAttention"
BASE_OUTPUT_DIR = PROJECT_ROOT / "last_dance" / "0318_v3" / "outputs"
PIPELINE_SUMMARY = BASE_OUTPUT_DIR / "pipeline_summary.csv"
PIPELINE_METRICS_FIELDS = [
    "timestamp",
    "task",
    "dataset_mode",
    "model",
    "seed",
    "profile",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "train_size",
    "val_size",
    "test_size",
    "num_features",
    "num_classes",
    "MAE",
    "RMSE",
    "CEP50",
    "CEP95",
    "Accuracy",
    "F1Macro",
    "PrecisionMacro",
    "RecallMacro",
    "TrainTimeSec",
    "InferTimeMS",
    "checkpoint",
    "status",
]

TASKS = {
    "lab": {
        "dataset_mode": "lab",
        "cache_name": "dataset_cache_lab_v3",
        "file_whitelist": ("cad_static_nlos.xlsx", "cad_dynamic_nlos.xlsx"),
        "labels": {0: "LOS", 1: "Static NLOS"},
        "colors": {0: "#4472C4", 1: "#ED7D31"},
        "tsne_per_class": 3000,
        "tsne_max": 6000,
        "features": "v3_lab",
    },
    "classroom": {
        "dataset_mode": "classroom",
        "cache_name": "dataset_cache_classroom_v3",
        "file_whitelist": ("los.xlsx", "nlos_static.xlsx", "nlos_dynamic.xlsx"),
        "labels": {0: "LOS", 1: "Static NLOS", 2: "Dynamic NLOS"},
        "colors": {0: "#4472C4", 1: "#ED7D31", 2: "#70AD47"},
        "tsne_per_class": 2000,
        "tsne_max": 6000,
        "features": "v3_classroom",
    },
}

CLASSROOM_BASELINES = {
    "LSTM": PROJECT_ROOT
    / "last_dance"
    / "last_dance"
    / "new_results_classroom_only"
    / "errors"
    / "LSTM_seed44_errors.npy",
    "FC-SVM": PROJECT_ROOT
    / "last_dance"
    / "last_dance"
    / "new_results_classroom_only"
    / "errors"
    / "FC-SVM_seed44_errors.npy",
    "EKF": PROJECT_ROOT
    / "last_dance"
    / "last_dance"
    / "new_results_classroom_only"
    / "errors"
    / "EKF_seed44_errors.npy",
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def format_count_map(df, label_col: str = "nlos_type") -> Dict[str, int]:
    if df is None or len(df) == 0 or label_col not in df.columns:
        return {}
    counts = df[label_col].value_counts().sort_index().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _load_pipeline(task_key: str, args: argparse.Namespace) -> Dict[str, object]:
    task = TASKS[task_key]
    provider = DataProvider(
        cache_name=task["cache_name"],
        dataset_mode=task["dataset_mode"],
        file_whitelist=task["file_whitelist"],
        train_ratio=0.6,
        val_ratio=0.2,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )

    if args.trace_data:
        raw = provider._load_raw()
        filtered = provider._apply_filter(raw)
        train_raw, val_raw, test_raw = provider._safe_point_wise_split(filtered)
        print("[trace] raw_rows =", len(raw))
        print("[trace] raw_count_by_class =", format_count_map(raw))
        print("[trace] filtered_rows =", len(filtered))
        print("[trace] filtered_count_by_class =", format_count_map(filtered))
        print(
            "[trace] split_rows =",
            {"train": len(train_raw), "val": len(val_raw), "test": len(test_raw)},
        )

    bundle = provider.prepare_once(force=args.rebuild_data)

    train_split = _select_profile(bundle.train, bundle, args.profile, split_name="train")
    val_split = (
        _select_profile(bundle.val, bundle, args.profile, split_name="val")
        if bundle.val is not None
        else None
    )
    test_split = _select_profile(bundle.test, bundle, args.profile, split_name="test")

    return {
        "provider": provider,
        "bundle": bundle,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "profile": args.profile,
    }


def _select_profile(
    split: Optional[SplitData],
    provider: DataProvider,
    profile: str,
    split_name: str,
) -> Optional[SplitData]:
    if split is None or split.X.size == 0:
        return SplitData(
            X=np.zeros((0, 0), dtype=np.float32),
            y_pos=np.zeros((0, 2), dtype=np.float32),
            y_class=np.zeros((0,), dtype=np.int64),
            meta=split.meta if split is not None else {},
        )

    feats = split.X
    if profile != "base":
        features_by_profile = provider.feature_store.get(split_name, {})
        feats = features_by_profile.get(profile, split.X)
        if feats.shape[0] != split.X.shape[0]:
            print(
                f"[warn] profile transform mismatch on {split_name}: expected {split.X.shape[0]} rows, got {feats.shape[0]}. "
                "fallback to base features."
            )
            feats = split.X

    return SplitData(
        X=feats.astype(np.float32),
        y_pos=split.y_pos.astype(np.float32),
        y_class=split.y_class.astype(np.int64),
        meta=split.meta,
    )


def print_bundle_summary(task_key: str, pipeline: Dict[str, object]) -> None:
    provider_bundle = pipeline["bundle"]
    train = pipeline["train_split"]
    val = pipeline["val_split"]
    test = pipeline["test_split"]
    train_split: SplitData = train  # type: ignore[assignment]
    val_split: Optional[SplitData] = val  # type: ignore[assignment]
    test_split: SplitData = test  # type: ignore[assignment]

    metadata = getattr(provider_bundle, "metadata", {})
    print(f"\n[{task_key}] metadata:", metadata)
    print(
        "split_sizes:",
        {
            "train": len(train_split.X),
            "val": 0 if val_split is None else len(val_split.X),
            "test": len(test_split.X),
        },
    )
    if train_split.y_class.size > 0:
        counts = np.bincount(train_split.y_class.astype(int))
        print("train_class_counts:", {int(i): int(v) for i, v in enumerate(counts)})


def build_model(input_dim: int, num_classes: int, args: argparse.Namespace) -> ProposedV2NoAttentionModel:
    return ProposedV2NoAttentionModel(
        input_dim=input_dim,
        num_classes=num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_epochs=args.min_epochs,
        cls_weight=args.cls_weight,
        conf_weight=args.conf_weight,
        pos_weight=args.pos_weight,
        nlos_boost=args.nlos_boost,
    )


def fit_model(
    model: ProposedV2NoAttentionModel,
    train_split: SplitData,
    val_split: Optional[SplitData],
) -> float:
    t0 = time.perf_counter()
    model.fit(train_split, val_split)
    return time.perf_counter() - t0


def infer_with_shared(
    model: ProposedV2NoAttentionModel,
    features: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model.artifacts is None:
        raise RuntimeError("model.fit has not been run, artifacts is None.")
    state = model.artifacts
    net = state.model
    net.eval()

    dataset = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    shared_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    cls_list: List[torch.Tensor] = []
    conf_list: List[torch.Tensor] = []

    def _hook(_module, _inp, out):
        shared_list.append(out.detach().cpu())

    hook = net.shared.register_forward_hook(_hook)
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(state.device)
            pos_pred, cls_logits, conf_pred, _ = net(x_batch)
            pos_list.append(pos_pred.detach().cpu())
            cls_list.append(torch.argmax(cls_logits, dim=1).detach().cpu())
            conf_list.append(conf_pred.squeeze(1).detach().cpu())
    hook.remove()

    if not pos_list:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 160), dtype=np.float32),
        )

    return (
        torch.cat(pos_list, dim=0).numpy(),
        torch.cat(cls_list, dim=0).numpy().astype(np.int64),
        torch.cat(conf_list, dim=0).numpy().astype(np.float32),
        torch.cat(shared_list, dim=0).numpy().astype(np.float32),
    )


def evaluate(
    y_pos: np.ndarray,
    y_cls: np.ndarray,
    pred_pos: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
) -> Dict[str, float]:
    evaluator = Evaluator()
    output = ModelOutput(position=pred_pos, classification=pred_cls, confidence=pred_conf)
    result = evaluator.evaluate(output, y_pos, y_cls)
    metrics = result.metrics
    errors = result.errors_cm
    metrics["error_mean"] = safe_div(float(np.sum(errors)), len(errors)) if len(errors) else float("nan")
    metrics["error_std"] = float(np.std(errors)) if errors.size else float("nan")
    return metrics


def draw_cdf(errors_map: Dict[str, np.ndarray], save_dir: Path, task_key: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.5, 2.8))

    color_map = {
        "Proposed_V3": "#E94F37",
        "LSTM": "#4472C4",
        "FC-SVM": "#70AD47",
        "EKF": "#7F7F7F",
    }

    for name, arr in errors_map.items():
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        x = np.sort(arr)
        y = np.linspace(0, 1, len(x), endpoint=True)
        plt.plot(x, y, label=name, color=color_map.get(name, "#333333"), linewidth=1.2)

    if "Proposed_V3" in errors_map:
        ce50 = safe_percentile(errors_map["Proposed_V3"], 50)
        ce95 = safe_percentile(errors_map["Proposed_V3"], 95)
        plt.axvline(ce50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.axvline(ce95, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.axhline(0.95, color="#aaaaaa", linestyle=":", linewidth=0.8, alpha=0.7)
        plt.text(ce50 * 1.02, 0.52, "CEP50", fontsize=7, color="#555555")
        plt.text(ce95 * 1.02, 0.97, "CEP95", fontsize=7, color="#555555")

    plt.xlabel("Position Error (cm)")
    plt.ylabel("Cumulative Probability")
    plt.xlim(0, 50)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / f"v3_{task_key}_cdf.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / f"v3_{task_key}_cdf.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def _subsample_for_tsne(
    feats: np.ndarray,
    labels: np.ndarray,
    per_class: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if feats.size == 0 or labels.size == 0:
        return np.zeros((0, feats.shape[1] if feats.ndim == 2 else 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(0)
    selected: List[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        k = min(per_class, len(idx))
        if k > 0:
            selected.extend(rng.choice(idx, size=k, replace=False).tolist())
    selected_arr = np.array(selected, dtype=int)
    if len(selected_arr) > max_points:
        selected_arr = rng.choice(selected_arr, size=max_points, replace=False)
    return feats[selected_arr], labels[selected_arr]


def draw_tsne(
    shared: np.ndarray,
    y_cls: np.ndarray,
    class_names: Dict[int, str],
    class_colors: Dict[int, str],
    save_dir: Path,
    task_key: str,
    per_class: int,
    max_points: int,
) -> None:
    if shared.size == 0:
        return
    feats, labels = _subsample_for_tsne(shared, y_cls, per_class=per_class, max_points=max_points)
    if feats.size == 0:
        return

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=0,
        init="random",
    )
    emb = tsne.fit_transform(feats)

    plt.figure(figsize=(3.5, 3.0))
    for cls, text in class_names.items():
        mask = labels == cls
        if not np.any(mask):
            continue
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=8,
            alpha=0.7,
            label=text,
            color=class_colors.get(cls, "#333333"),
        )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.xticks([])
    plt.yticks([])
    plt.legend(markerscale=1.5, fontsize=7)
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"v3_{task_key}_tsne.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / f"v3_{task_key}_tsne.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def write_csv_row(path: Path, row: Dict[str, float], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    append_header = (not path.exists()) or (path.stat().st_size == 0)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if append_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def run_task(task_key: str, args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    task = TASKS[task_key]
    out_dir = BASE_OUTPUT_DIR / task_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== TASK: {task_key} ===")
    pipeline = _load_pipeline(task_key, args)
    print_bundle_summary(task_key, pipeline)

    train = pipeline["train_split"]
    val = pipeline["val_split"]
    test = pipeline["test_split"]
    train_split: SplitData = train  # type: ignore[assignment]
    val_split: Optional[SplitData] = val  # type: ignore[assignment]
    test_split: SplitData = test  # type: ignore[assignment]

    if train_split.X.size == 0 or test_split.X.size == 0:
        raise RuntimeError(f"{task_key}: train or test split is empty. Check raw data and filter options.")

    input_dim = int(train_split.X.shape[1])
    num_classes = int(np.max(test_split.y_class) + 1) if test_split.y_class.size else 0

    model = build_model(input_dim, num_classes, args)
    train_time = fit_model(model, train_split, val_split)

    t0 = time.perf_counter()
    pred_pos, pred_cls, pred_conf, shared = infer_with_shared(model, test_split.X, args)
    infer_ms = safe_div((time.perf_counter() - t0) * 1000.0, len(test_split.X))

    metrics = evaluate(test_split.y_pos, test_split.y_class, pred_pos, pred_cls, pred_conf)
    errors_cm = np.linalg.norm(pred_pos - test_split.y_pos, axis=1) * 100.0
    print(
        f"result: MAE={metrics.get('MAE', safe_percentile(errors_cm, 50)):.3f} "
        f"CEP50={safe_percentile(errors_cm, 50):.3f} "
        f"CEP95={safe_percentile(errors_cm, 95):.3f} "
        f"ACC={metrics.get('Accuracy', float('nan')):.3f}"
    )

    # save raw artifacts
    err_dir = out_dir / "errors"
    err_dir.mkdir(parents=True, exist_ok=True)
    errors_path = err_dir / f"{task['features']}_seed{args.seed}_errors_cm.npy"
    feats_path = err_dir / f"{task['features']}_seed{args.seed}_features.npy"
    labels_path = err_dir / f"{task['features']}_seed{args.seed}_labels.npy"
    np.save(errors_path, errors_cm)
    np.save(feats_path, shared)
    np.save(labels_path, test_split.y_class)

    # copy best checkpoint
    source_ckpt = Path(model.cache_dir) / f"proposed_best_c{num_classes}.pth"
    target_ckpt = out_dir / f"{task['features']}_seed{args.seed}_best.pth"
    if source_ckpt.exists():
        shutil.copy2(source_ckpt, target_ckpt)
        checkpoint = str(target_ckpt)
    else:
        checkpoint = str(source_ckpt)

    # figure export
    cdf_map: Dict[str, np.ndarray] = {"Proposed_V3": errors_cm}
    if task_key == "classroom" and args.classroom_baseline:
        for name, path in CLASSROOM_BASELINES.items():
            if path.exists():
                cdf_map[name] = np.load(path).astype(float)
    draw_cdf(cdf_map, out_dir, task_key)
    draw_tsne(
        shared,
        test_split.y_class,
        task["labels"],
        task["colors"],
        out_dir,
        task_key,
        per_class=task["tsne_per_class"],
        max_points=task["tsne_max"],
    )

    row: Dict[str, float] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task_key,
        "dataset_mode": task["dataset_mode"],
        "model": MODEL_NAME,
        "seed": args.seed,
        "profile": args.profile,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_size": float(len(train_split.X)),
        "val_size": float(0 if val_split is None else len(val_split.X)),
        "test_size": float(len(test_split.X)),
        "num_features": float(input_dim),
        "num_classes": float(num_classes),
        "MAE": float(metrics.get("MAE", float("nan"))),
        "RMSE": float(metrics.get("RMSE", float("nan"))),
        "CEP50": float(metrics.get("CEP50", float("nan"))),
        "CEP95": float(metrics.get("CEP95", float("nan"))),
        "Accuracy": float(metrics.get("Accuracy", float("nan"))),
        "F1Macro": float(metrics.get("F1Macro", float("nan"))),
        "PrecisionMacro": float(metrics.get("PrecisionMacro", float("nan"))),
        "RecallMacro": float(metrics.get("RecallMacro", float("nan"))),
        "TrainTimeSec": float(train_time),
        "InferTimeMS": float(infer_ms),
        "checkpoint": checkpoint,
        "status": "ok",
    }

    metrics_csv = BASE_OUTPUT_DIR / "metrics" / f"{task['features']}_metrics.csv"
    write_csv_row(metrics_csv, row, PIPELINE_METRICS_FIELDS)
    write_csv_row(PIPELINE_SUMMARY, row, PIPELINE_METRICS_FIELDS)

    print("outputs:")
    print(f"  errors:   {errors_path}")
    print(f"  features: {feats_path}")
    print(f"  labels:   {labels_path}")
    print(f"  cdf:      {out_dir / f'v3_{task_key}_cdf.png'}")
    print(f"  tsne:     {out_dir / f'v3_{task_key}_tsne.png'}")
    print(f"  metrics:  {metrics_csv}")
    print(f"  ckpt:     {checkpoint}")

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3 full pipeline single file.")
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=["all", "lab", "classroom"])
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    parser.add_argument("--profile", default="base", help="Feature profile: base / base_no_snr / ...")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=45, help="Early stop patience")
    parser.add_argument("--min-epochs", type=int, default=60, help="Minimum epoch before early stop")
    parser.add_argument("--cls-weight", type=float, default=2.5, help="Classification loss weight")
    parser.add_argument("--conf-weight", type=float, default=0.7, help="Confidence loss weight")
    parser.add_argument("--pos-weight", type=float, default=3.0, help="Position loss weight")
    parser.add_argument("--nlos-boost", type=float, default=0.5, help="NLOS weight factor on position loss")
    parser.add_argument("--rebuild-data", action="store_true", help="Rebuild cache from raw data")
    parser.add_argument(
        "--trace-data",
        action="store_true",
        help="Print raw/filter/split row counts before training",
    )
    parser.add_argument(
        "--classroom-baseline",
        action="store_true",
        help="Add classroom baseline curves on CDF (LSTM/FC-SVM/EKF)",
    )
    parser.add_argument("--raw-dir", type=str, default=str(Path("new_bench_03/1_data/raw")))
    parser.add_argument("--processed-dir", type=str, default=str(Path("new_bench_03/1_data/processed")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = ["lab", "classroom"] if "all" in args.tasks else args.tasks
    rows: List[Dict[str, float]] = []
    for task_key in requested:
        rows.append(run_task(task_key, args))

    print("\nPipeline done.")
    print(f"script: {SCRIPT_PATH}")
    print(f"output_root: {BASE_OUTPUT_DIR}")
    if rows:
        print(f"last MAE: {rows[-1].get('MAE', float('nan')):.4f} cm")
    print(f"summary csv: {PIPELINE_SUMMARY}")


if __name__ == "__main__":
    main()
