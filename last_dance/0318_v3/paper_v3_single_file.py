"""Paper-oriented single-file v3 pipeline (No-Attention), readable style.

Flow:
1) Load / preprocess / split data
2) Choose feature profile
3) Build model
4) Train
5) Inference and shared-feature extraction
6) Evaluate
7) Save outputs / summary

No visualization code is included.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_bench_03.data.provider import DataProvider, SplitData
from new_bench_03.models.base import ModelOutput
from new_bench_03.models.proposed.variants import ProposedV2NoAttentionModel
from new_bench_03.output.evaluator import Evaluator


MODEL_NAME = "Proposed_V2_NoAttention"
OUTPUT_ROOT = PROJECT_ROOT / "last_dance" / "0318_v3" / "outputs"
PIPELINE_SUMMARY = OUTPUT_ROOT / "pipeline_summary.csv"
PIPELINE_FIELDS = [
    "timestamp",
    "task",
    "dataset_mode",
    "model",
    "profile",
    "seed",
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
    "error_mean",
    "error_std",
    "TrainTimeSec",
    "InferTimeMS",
    "checkpoint",
    "status",
]

TASKS: Dict[str, Dict[str, object]] = {
    "lab": {
        "dataset_mode": "lab",
        "cache_name": "dataset_cache_lab_v3",
        "file_whitelist": ("cad_static_nlos.xlsx", "cad_dynamic_nlos.xlsx"),
    },
    "classroom": {
        "dataset_mode": "classroom",
        "cache_name": "dataset_cache_classroom_v3",
        "file_whitelist": ("los.xlsx", "nlos_static.xlsx", "nlos_dynamic.xlsx"),
    },
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


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _count_by_label(df) -> Dict[str, int]:
    if df is None or len(df) == 0 or "nlos_type" not in df.columns:
        return {}
    cnt = df["nlos_type"].value_counts().sort_index().to_dict()
    return {str(k): int(v) for k, v in cnt.items()}


# ---------------------- 1) Data loading and feature profile ----------------------
def _select_feature_profile(
    split: Optional[SplitData],
    provider: DataProvider,
    profile: str,
    split_name: str,
) -> SplitData:
    if split is None or split.X.size == 0:
        return SplitData(
            X=np.zeros((0, 0), dtype=np.float32),
            y_pos=np.zeros((0, 2), dtype=np.float32),
            y_class=np.zeros((0,), dtype=np.int64),
            meta=split.meta if split is not None else {},
        )

    feats = split.X
    if profile != "base":
        store = provider.feature_store.get(split_name, {})
        feats = store.get(profile, split.X)
        if feats.shape[0] != split.X.shape[0]:
            print(
                f"[warn] profile '{profile}' mismatch on {split_name}: "
                f"expected {split.X.shape[0]}, got {feats.shape[0]}. use base."
            )
            feats = split.X
    return SplitData(
        X=feats.astype(np.float32),
        y_pos=split.y_pos.astype(np.float32),
        y_class=split.y_class.astype(np.int64),
        meta=split.meta,
    )


def load_task(task_key: str, args: argparse.Namespace) -> Tuple[SplitData, Optional[SplitData], SplitData]:
    cfg = TASKS[task_key]
    provider = DataProvider(
        cache_name=cfg["cache_name"],
        dataset_mode=cfg["dataset_mode"],
        file_whitelist=cfg["file_whitelist"],
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
        print("[trace] raw_count_by_class =", _count_by_label(raw))
        print("[trace] filtered_rows =", len(filtered))
        print("[trace] filtered_count_by_class =", _count_by_label(filtered))
        print("[trace] split_rows =", {"train": len(train_raw), "val": len(val_raw), "test": len(test_raw)})

    bundle = provider.prepare_once(force=args.rebuild_data)
    train = _select_feature_profile(bundle.train, provider, args.profile, "train")
    val = _select_feature_profile(bundle.val, provider, args.profile, "val") if bundle.val is not None else None
    test = _select_feature_profile(bundle.test, provider, args.profile, "test")
    return train, val, test


# ---------------------- 2) Model ----------------------
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


def fit_model(model: ProposedV2NoAttentionModel, train_split: SplitData, val_split: Optional[SplitData]) -> float:
    t0 = time.perf_counter()
    model.fit(train_split, val_split)
    return time.perf_counter() - t0


# ---------------------- 3) Inference ----------------------
def infer_with_shared(
    model: ProposedV2NoAttentionModel,
    split: SplitData,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if model.artifacts is None:
        raise RuntimeError("model.fit must run before inference.")

    net = model.artifacts.model
    device = model.artifacts.device
    net.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(split.X)), batch_size=batch_size, shuffle=False)

    shared_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    cls_list: List[torch.Tensor] = []
    conf_list: List[torch.Tensor] = []

    def _hook(_module, _inp, out):
        shared_list.append(out.detach().cpu())

    hook = net.shared.register_forward_hook(_hook)
    t0 = time.perf_counter()
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            pos_pred, cls_logit, conf_pred, _ = net(x_batch)
            pos_list.append(pos_pred.detach().cpu())
            cls_list.append(torch.argmax(cls_logit, dim=1).detach().cpu())
            conf_list.append(conf_pred.squeeze(1).detach().cpu())
    hook.remove()

    infer_ms = safe_div((time.perf_counter() - t0) * 1000.0, len(split.X))
    if not pos_list:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            float(infer_ms),
        )

    pred_pos = torch.cat(pos_list, dim=0).numpy()
    pred_cls = torch.cat(cls_list, dim=0).numpy().astype(np.int64)
    pred_conf = torch.cat(conf_list, dim=0).numpy().astype(np.float32)
    shared = torch.cat(shared_list, dim=0).numpy().astype(np.float32)
    return pred_pos, pred_cls, pred_conf, shared, float(infer_ms)


# ---------------------- 4) Evaluation ----------------------
def evaluate(
    y_pos: np.ndarray,
    y_cls: np.ndarray,
    pred_pos: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    out = ModelOutput(position=pred_pos, classification=pred_cls, confidence=pred_conf)
    ev = Evaluator()
    result = ev.evaluate(out, y_pos, y_cls)
    errors_cm = result.errors_cm.astype(np.float32)
    metrics = dict(result.metrics)
    metrics["error_mean"] = safe_div(float(np.sum(errors_cm)), len(errors_cm))
    metrics["error_std"] = float(np.std(errors_cm)) if errors_cm.size else float("nan")
    return metrics, errors_cm


def append_pipeline_row(row: Dict[str, float]) -> None:
    PIPELINE_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not PIPELINE_SUMMARY.exists()) or PIPELINE_SUMMARY.stat().st_size == 0
    with PIPELINE_SUMMARY.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PIPELINE_FIELDS})


def save_task_outputs(
    task_key: str,
    args: argparse.Namespace,
    split: SplitData,
    profile: str,
    seed: int,
    pred_pos: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    shared: np.ndarray,
    errors_cm: np.ndarray,
    metrics: Dict[str, float],
    train_sec: float,
    infer_ms: float,
    ckpt_path: str,
    input_dim: int,
    num_classes: int,
    train_size: int,
    val_size: int,
) -> Dict[str, float]:
    out = OUTPUT_ROOT / task_key
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / f"v3_{task_key}_seed{seed}_errors_cm.npy", errors_cm)
    np.save(out / f"v3_{task_key}_seed{seed}_y_true_pos.npy", split.y_pos)
    np.save(out / f"v3_{task_key}_seed{seed}_y_true_cls.npy", split.y_class)
    np.save(out / f"v3_{task_key}_seed{seed}_pred_pos.npy", pred_pos)
    np.save(out / f"v3_{task_key}_seed{seed}_pred_cls.npy", pred_cls)
    np.save(out / f"v3_{task_key}_seed{seed}_pred_conf.npy", pred_conf)
    np.save(out / f"v3_{task_key}_seed{seed}_shared.npy", shared)

    row: Dict[str, float] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task_key,
        "dataset_mode": TASKS[task_key]["dataset_mode"],
        "model": MODEL_NAME,
        "profile": profile,
        "seed": seed,
        "train_size": float(train_size),
        "val_size": float(val_size),
        "test_size": float(len(split.y_pos)),
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
        "error_mean": float(metrics.get("error_mean", float("nan"))),
        "error_std": float(metrics.get("error_std", float("nan"))),
        "TrainTimeSec": float(train_sec),
        "InferTimeMS": float(infer_ms),
        "checkpoint": ckpt_path,
        "status": "ok",
    }

    json_blob = dict(row)
    json_blob["metrics"] = {
        "MAE": float(metrics.get("MAE", float("nan"))),
        "RMSE": float(metrics.get("RMSE", float("nan"))),
        "CEP50": float(metrics.get("CEP50", float("nan"))),
        "CEP95": float(metrics.get("CEP95", float("nan"))),
        "Accuracy": float(metrics.get("Accuracy", float("nan"))),
        "F1Macro": float(metrics.get("F1Macro", float("nan"))),
        "PrecisionMacro": float(metrics.get("PrecisionMacro", float("nan"))),
        "RecallMacro": float(metrics.get("RecallMacro", float("nan"))),
        "error_mean": float(metrics.get("error_mean", float("nan"))),
        "error_std": float(metrics.get("error_std", float("nan"))),
    }
    with (out / f"run_seed{seed}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_blob, f, indent=2)

    append_pipeline_row(row)
    return row


# ---------------------- 5) Per-task run ----------------------
def run_task(task_key: str, args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    train_split, val_split, test_split = load_task(task_key, args)

    train_size = len(train_split.y_pos)
    val_size = 0 if val_split is None else len(val_split.y_pos)
    print(
        f"\n[{task_key}] split_sizes="
        f"{{'train': {train_size}, 'val': {val_size}, 'test': {len(test_split.y_pos)}}"
    )

    if train_split.X.size == 0 or test_split.X.size == 0:
        raise RuntimeError(f"{task_key}: empty train or test split.")

    num_classes = int(np.max(test_split.y_class) + 1) if test_split.y_class.size else 0
    input_dim = int(train_split.X.shape[1])

    model = build_model(input_dim=input_dim, num_classes=num_classes, args=args)
    train_sec = fit_model(model, train_split, val_split)
    pred_pos, pred_cls, pred_conf, shared, infer_ms = infer_with_shared(model, test_split, args.batch_size)
    metrics, errors_cm = evaluate(test_split.y_pos, test_split.y_class, pred_pos, pred_cls, pred_conf)

    src_ckpt = Path(model.cache_dir) / f"proposed_best_c{num_classes}.pth"
    task_out = OUTPUT_ROOT / task_key
    task_out.mkdir(parents=True, exist_ok=True)
    copied_ckpt = task_out / f"{task_key}_seed{args.seed}_best.pth"
    if src_ckpt.exists():
        shutil.copy2(src_ckpt, copied_ckpt)
        ckpt = str(copied_ckpt)
    else:
        ckpt = str(src_ckpt)

    row = save_task_outputs(
        task_key=task_key,
        args=args,
        split=test_split,
        profile=args.profile,
        seed=args.seed,
        pred_pos=pred_pos,
        pred_cls=pred_cls,
        pred_conf=pred_conf,
        shared=shared,
        errors_cm=errors_cm,
        metrics=metrics,
        train_sec=train_sec,
        infer_ms=infer_ms,
        ckpt_path=ckpt,
        input_dim=input_dim,
        num_classes=num_classes,
        train_size=train_size,
        val_size=val_size,
    )

    print(
        f"[done] {task_key}: "
        f"MAE={row['MAE']:.4f}, CEP50={row['CEP50']:.4f}, CEP95={row['CEP95']:.4f}, "
        f"ACC={row['Accuracy']:.4f}"
    )
    return row


# ---------------------- 6) CLI ----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proposed V2 NoAttention full one-file pipeline")
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=["all", "lab", "classroom"])
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--profile", type=str, default="base")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--cls-weight", type=float, default=2.5)
    parser.add_argument("--conf-weight", type=float, default=0.7)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--nlos-boost", type=float, default=0.5)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--trace-data", action="store_true")
    parser.add_argument("--raw-dir", type=str, default=str(Path("new_bench_03/1_data/raw")))
    parser.add_argument("--processed-dir", type=str, default=str(Path("new_bench_03/1_data/processed")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ["lab", "classroom"] if "all" in args.tasks else args.tasks
    for task_key in tasks:
        run_task(task_key, args)
    print("\nDone.")
    print(f"outputs: {OUTPUT_ROOT}")
    print(f"pipeline_summary: {PIPELINE_SUMMARY}")


if __name__ == "__main__":
    main()
