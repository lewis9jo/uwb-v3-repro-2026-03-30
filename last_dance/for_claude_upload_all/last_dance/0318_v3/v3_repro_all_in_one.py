"""Single-file reproducible v3 pipeline.

실행 모드
1) --mode train
   - lab / classroom 데이터셋에 대해 학습->평가->CDF/t-SNE 생성
   - 결과는 2026-03-30-v3 루트 하위로 저장

2) --mode figures
   - 이미 학습된 체크포인트가 있으면 figure/CDF/t-SNE만 재생성

기본 출력 경로:
  {RESULT_ROOT}/last_dance/0318_v3/outputs/{task}
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_BUNDLE_DIR = SCRIPT_DIR / "repro_bundle"
if (REPRO_BUNDLE_DIR / "new_bench_03").exists():
    if str(REPRO_BUNDLE_DIR) not in sys.path:
        sys.path.insert(0, str(REPRO_BUNDLE_DIR))


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "new_bench_03").exists():
            return candidate
    raise FileNotFoundError("new_bench_03 module root not found.")


try:
    REPO_ROOT = _find_repo_root(SCRIPT_DIR)
except FileNotFoundError:
    if (REPRO_BUNDLE_DIR / "new_bench_03").exists():
        REPO_ROOT = REPRO_BUNDLE_DIR
    else:
        raise
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_ROOT = Path(__file__).resolve()
for ancestor in (SCRIPT_ROOT, *SCRIPT_ROOT.parents):
    if ancestor.name == "2026-03-30-v3":
        RESULT_ROOT = ancestor
        break
else:
    RESULT_ROOT = REPO_ROOT / "2026-03-30-v3"

from new_bench_03.data.provider import DataProvider, SplitData
from new_bench_03.models.base import BaseModel, ModelOutput
from new_bench_03.models.proposed.variants import ProposedV2NoAttentionModel
from new_bench_03.models.registry import create_model
from new_bench_03.output.evaluator import Evaluator


OUTPUT_ROOT = RESULT_ROOT / "last_dance" / "0318_v3" / "outputs"
PIPELINE_SUMMARY = OUTPUT_ROOT / "pipeline_summary.csv"
REPORT_FIELDS = [
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
    "TrainTimeSec",
    "InferTimeMS",
    "checkpoint",
    "status",
]


TASKS = {
    "lab": {
        "name": "lab",
        "dataset_mode": "lab",
        "cache_name": "dataset_cache_lab_v3",
        "file_whitelist": ("cad_static_nlos.xlsx", "cad_dynamic_nlos.xlsx"),
        "out_dir": OUTPUT_ROOT / "lab",
        "class_labels": {0: "LOS", 1: "Static NLOS"},
        "class_colors": {0: "#4472C4", 1: "#ED7D31"},
        "tsne_per_class": 3000,
        "tsne_max_points": 6000,
        "baseline_ckpt": REPO_ROOT
        / "new_bench_03/2_models/proposed/checkpoints/ProposedV2NoAttentionModel/proposed_best_c2.pth",
        "num_classes": 2,
    },
    "classroom": {
        "name": "classroom",
        "dataset_mode": "classroom",
        "cache_name": "dataset_cache_classroom_v3",
        "file_whitelist": ("los.xlsx", "nlos_static.xlsx", "nlos_dynamic.xlsx"),
        "out_dir": OUTPUT_ROOT / "classroom",
        "class_labels": {0: "LOS", 1: "Static NLOS", 2: "Dynamic NLOS"},
        "class_colors": {0: "#4472C4", 1: "#ED7D31", 2: "#70AD47"},
        "tsne_per_class": 2000,
        "tsne_max_points": 6000,
        "baseline_ckpt": REPO_ROOT
        / "new_bench_03/2_models/proposed/checkpoints/ProposedV2NoAttentionModel/proposed_best_c3.pth",
        "num_classes": 3,
    },
}

MODEL_NAME = "Proposed_V2_NoAttention"
MODEL_CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

BASELINE_CLASSROOM = {
    "Proposed_V3": RESULT_ROOT
    / "last_dance"
    / "last_dance"
    / "v3_clean_run"
    / "errors"
    / "Proposed_V2_NoAttention_seed44_errors.npy",
    "LSTM": RESULT_ROOT
    / "last_dance"
    / "last_dance"
    / "new_results_classroom_only"
    / "errors"
    / "LSTM_seed44_errors.npy",
    "FC-SVM": RESULT_ROOT
    / "last_dance"
    / "last_dance"
    / "new_results_classroom_only"
    / "errors"
    / "FC-SVM_seed44_errors.npy",
    "EKF": RESULT_ROOT
    / "last_dance"
    / "last_dance"
    / "new_results_classroom_only"
    / "errors"
    / "EKF_seed44_errors.npy",
}


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


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


def ensure_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _trace_preprocessing(provider: DataProvider) -> None:
    raw = provider._load_raw()
    print(f"[trace] raw_rows={len(raw)}")
    if raw.empty:
        return
    filtered = provider._apply_filter(raw)
    print(
        f"[trace] after_filter_rows={len(filtered)} (filter={provider.filter_type})"
    )
    train_raw, val_raw, test_raw = provider._safe_point_wise_split(filtered)
    print(
        "[trace] split_rows=",
        {
            "train": len(train_raw),
            "val": len(val_raw),
            "test": len(test_raw),
        },
    )


def _load_bundle(
    cache_name: str,
    dataset_mode: str,
    file_whitelist: Optional[Tuple[str, ...]],
    force_data: bool,
    trace: bool,
):
    provider = DataProvider(
        cache_name=cache_name,
        dataset_mode=dataset_mode,
        file_whitelist=file_whitelist,
        train_ratio=0.6,
        val_ratio=0.2,
    )
    if trace:
        _trace_preprocessing(provider)
    bundle = provider.prepare_once(force=force_data)
    return provider, bundle


def _extract_features(bundle: SplitData, split_name: str, profile: str) -> np.ndarray:
    split = getattr(bundle, split_name)
    if split is None:
        return np.zeros((0, 0), dtype=np.float32)
    if profile == "base":
        return split.X
    store = bundle.feature_store.get(split_name, {})
    return store.get(profile, split.X)


def _repack_with_features(original: SplitData, features: np.ndarray) -> SplitData:
    meta: Dict[str, np.ndarray] = {}
    if getattr(original, "meta", None):
        for key, value in original.meta.items():
            meta[key] = np.array(value, copy=True)
    return SplitData(features.astype(np.float32), original.y_pos, original.y_class, meta)


def _build_model(
    task_name: str,
    seed: int,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    cls_weight: float,
    conf_weight: float,
    pos_weight: float,
    nlos_boost: float,
) -> BaseModel:
    cache_dir = MODEL_CHECKPOINT_DIR / task_name / f"seed_{seed}"
    return create_model(
        MODEL_NAME,
        input_dim=input_dim,
        num_classes=num_classes,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_epochs=min_epochs,
        cls_weight=cls_weight,
        conf_weight=conf_weight,
        pos_weight=pos_weight,
        nlos_boost=nlos_boost,
        cache_dir=str(cache_dir),
    )


def _fit_and_time(model: BaseModel, train_split: SplitData, val_split: Optional[SplitData]) -> float:
    t0 = time.perf_counter()
    if model.requires_fit:
        model.fit(train_split, val_split)
    return time.perf_counter() - t0


def _predict_with_shared(
    model: BaseModel,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if getattr(model, "artifacts", None) is None:
        raise RuntimeError("Model not initialized (call fit first).")
    state = model.artifacts
    net = state.model
    ds = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    shared_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    cls_list: List[torch.Tensor] = []
    conf_list: List[torch.Tensor] = []

    def _hook(_module, _inp, out):
        shared_list.append(out.detach().cpu())

    hook = net.shared.register_forward_hook(_hook)
    net.eval()

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            pos, cls_logits, conf, _ = net(x)
            pos_list.append(pos.detach().cpu())
            cls_list.append(torch.argmax(cls_logits, dim=1).detach().cpu())
            conf_list.append(conf.squeeze(1).detach().cpu())

    hook.remove()

    pred_pos = torch.cat(pos_list, dim=0).numpy()
    pred_cls = torch.cat(cls_list, dim=0).numpy()
    pred_conf = torch.cat(conf_list, dim=0).numpy()
    shared = torch.cat(shared_list, dim=0).numpy()
    return pred_pos, pred_cls, pred_conf, shared


def _predict_with_pretrained(
    input_dim: int,
    num_classes: int,
    features: np.ndarray,
    checkpoint_path: Path,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model = ProposedV2NoAttentionModel(input_dim=input_dim, num_classes=num_classes)
    state = model._init_state()  # type: ignore[attr-defined]
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            weights = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            weights = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            weights = ckpt["model"]
        else:
            weights = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    state.model.load_state_dict(weights)
    state.model.to(device)
    state.model.eval()
    model.artifacts = state  # type: ignore[attr-defined]
    ds = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    shared_feats: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []

    def _hook(_module, _inp, out):
        shared_feats.append(out.detach().cpu())

    hook = state.model.shared.register_forward_hook(_hook)  # type: ignore[attr-defined]
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            pos, *_ = state.model(batch_X)
            pos_list.append(pos.detach().cpu())
    hook.remove()

    pred_pos = torch.cat(pos_list, dim=0).numpy()
    shared = torch.cat(shared_feats, dim=0).numpy()
    return pred_pos, shared


def evaluate_and_pack(
    y_pos: np.ndarray,
    y_cls: np.ndarray,
    pred_pos: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
) -> Dict[str, float]:
    evaluator = Evaluator()
    output = ModelOutput(position=pred_pos, classification=pred_cls, confidence=pred_conf)
    result = evaluator.evaluate(output, y_pos, y_cls)
    stats = dict(result.metrics)
    errors = result.errors_cm
    stats["error_mean"] = float(np.mean(errors)) if errors.size else float("nan")
    stats["error_std"] = float(np.std(errors)) if errors.size else float("nan")
    return stats


def _plot_cdf(
    errors: np.ndarray,
    extra_errors: Dict[str, np.ndarray],
    save_dir: Path,
    task_name: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.5, 2.8))

    def _plot(name: str, arr: np.ndarray, color: str) -> None:
        if arr.size == 0:
            return
        sorted_errors = np.sort(arr)
        cdf = np.linspace(0, 1, len(sorted_errors), endpoint=True)
        plt.plot(sorted_errors, cdf, label=name, color=color, linewidth=1.2)

    color_map = {
        "Proposed_V3": "#E94F37",
        "LSTM": "#4472C4",
        "FC-SVM": "#70AD47",
        "EKF": "#7F7F7F",
    }
    for name, arr in extra_errors.items():
        _plot(name, arr.astype(float), color_map.get(name, "#333333"))

    if "Proposed_V3" in extra_errors:
        cep50 = _safe_percentile(extra_errors["Proposed_V3"], 50)
        cep95 = _safe_percentile(extra_errors["Proposed_V3"], 95)
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

    prefix = save_dir / f"v3_{task_name}_cdf"
    plt.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def _subsample_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    per_class: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    selected: List[int] = []
    for cls in np.unique(labels):
        candidates = np.where(labels == cls)[0]
        if len(candidates) == 0:
            continue
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    idx = np.array(selected, dtype=int)
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    return features[idx], labels[idx]


def _plot_tsne(
    shared: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    class_colors: Dict[int, str],
    save_dir: Path,
    task_name: str,
    per_class: int,
    max_points: int,
) -> None:
    if shared.size == 0:
        return
    feats, y = _subsample_tsne(shared, labels, per_class=per_class, max_points=max_points)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=0,
        init="random",
    )
    emb = tsne.fit_transform(feats)

    plt.figure(figsize=(3.5, 3.0))
    for cls, name in class_names.items():
        mask = y == cls
        if not np.any(mask):
            continue
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=8,
            alpha=0.7,
            label=name,
            color=class_colors.get(cls),
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.xticks([])
    plt.yticks([])
    plt.legend(markerscale=1.5, fontsize=7)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = save_dir / f"v3_{task_name}_tsne"
    plt.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def _append_rows(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or (path.stat().st_size == 0)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REPORT_FIELDS})


def _baseline_map(task_key: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if task_key != "classroom":
        return out
    for name, path in BASELINE_CLASSROOM.items():
        if path.exists():
            out[name] = np.load(path).astype(float)
        else:
            print(f"[warn] missing baseline error file: {path}")
    return out


def run_training_task(task_key: str, task: dict, args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    task_dir: Path = task["out_dir"]
    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Task: {task_key} ({'train+figure' if not args.skip_figure else 'train only'}) ===")

    provider, bundle = _load_bundle(
        cache_name=task["cache_name"],
        dataset_mode=task["dataset_mode"],
        file_whitelist=task["file_whitelist"],
        force_data=args.rebuild_data,
        trace=args.trace_data,
    )

    print("bundle:", bundle.metadata)
    print(
        "split_size:",
        {
            "train": len(bundle.train.X),
            "val": 0 if bundle.val is None else len(bundle.val.X),
            "test": len(bundle.test.X),
        },
    )

    profile = args.profile
    train_X = _extract_features(bundle, "train", profile)
    val_X = _extract_features(bundle, "val", profile)
    test_X = _extract_features(bundle, "test", profile)

    train_split = _repack_with_features(bundle.train, train_X)
    val_split = _repack_with_features(bundle.val, val_X) if bundle.val is not None else None
    test_split = _repack_with_features(bundle.test, test_X)

    num_classes = int(np.max(test_split.y_class) + 1) if test_split.y_class.size else task.get("num_classes", 0)
    input_dim = int(train_split.X.shape[1])
    print("input_dim:", input_dim, "num_classes:", num_classes)

    model = _build_model(
        task_key,
        args.seed,
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

    train_sec = _fit_and_time(model, train_split, val_split)

    t0 = time.perf_counter()
    pred_pos, pred_cls, pred_conf, shared = _predict_with_shared(
        model, test_split.X, args.batch_size, ensure_device()
    )
    infer_ms = (time.perf_counter() - t0) / max(1, len(test_split.X)) * 1000.0

    metrics = evaluate_and_pack(
        test_split.y_pos, test_split.y_class, pred_pos, pred_cls, pred_conf
    )
    errors_cm = np.linalg.norm(pred_pos - test_split.y_pos, axis=1) * 100.0

    np.save(task_dir / f"v3_{task_key}_seed{args.seed}_errors_cm.npy", errors_cm)
    np.save(task_dir / f"v3_{task_key}_seed{args.seed}_features.npy", shared)
    np.save(task_dir / f"v3_{task_key}_seed{args.seed}_labels.npy", test_split.y_class)

    ckpt_src = (
        Path(model.cache_dir)
        / model.__class__.__name__
        / f"proposed_best_c{num_classes}.pth"
    )
    ckpt_dst = task_dir / f"v3_{task_key}_seed{args.seed}_best.pth"
    if ckpt_src.exists():
        shutil.copy2(ckpt_src, ckpt_dst)

    if not args.skip_figure:
        cdf_errors: Dict[str, np.ndarray] = {"Proposed_V3": errors_cm}
        if task_key == "classroom" and args.classroom_baseline:
            cdf_errors.update(_baseline_map(task_key))

        _plot_cdf(errors_cm, cdf_errors, task_dir, task_key)
        _plot_tsne(
            shared,
            test_split.y_class,
            task["class_labels"],
            task["class_colors"],
            task_dir,
            task_key,
            per_class=task["tsne_per_class"],
            max_points=task["tsne_max_points"],
        )

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task_key,
        "dataset_mode": task["dataset_mode"],
        "model": MODEL_NAME,
        "profile": profile,
        "seed": args.seed,
        "train_size": float(len(train_split.X)),
        "val_size": float(0 if val_split is None else len(val_split.X)),
        "test_size": float(len(test_split.X)),
        "num_features": float(input_dim),
        "num_classes": float(num_classes),
        "MAE": metrics.get("MAE", float("nan")),
        "RMSE": metrics.get("RMSE", float("nan")),
        "CEP50": metrics.get("CEP50", float("nan")),
        "CEP95": metrics.get("CEP95", float("nan")),
        "Accuracy": metrics.get("Accuracy", float("nan")),
        "F1Macro": metrics.get("F1Macro", float("nan")),
        "PrecisionMacro": metrics.get("PrecisionMacro", float("nan")),
        "RecallMacro": metrics.get("RecallMacro", float("nan")),
        "TrainTimeSec": float(train_sec),
        "InferTimeMS": float(infer_ms),
        "checkpoint": str(ckpt_dst) if ckpt_dst.exists() else "",
        "status": "ok",
    }
    return row


def _resolve_checkpoint_for_figures(task_key: str, task: dict, args: argparse.Namespace) -> Path:
    preferred = (
        task["out_dir"] / f"v3_{task_key}_seed{args.seed}_best.pth"
    )
    if preferred.exists():
        return preferred
    fallback = task["baseline_ckpt"]
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No checkpoint found for {task_key} in {preferred} or {fallback}")


def run_figure_task(task_key: str, task: dict, args: argparse.Namespace) -> None:
    set_seed(args.seed)
    print(f"\n=== Figure task only: {task_key} ===")
    task_dir: Path = task["out_dir"]
    task_dir.mkdir(parents=True, exist_ok=True)
    provider, bundle = _load_bundle(
        cache_name=task["cache_name"],
        dataset_mode=task["dataset_mode"],
        file_whitelist=task["file_whitelist"],
        force_data=args.rebuild_data,
        trace=args.trace_data,
    )

    profile = args.profile
    test_X = _extract_features(bundle, "test", profile)
    test_split = _repack_with_features(bundle.test, test_X)
    input_dim = int(test_split.X.shape[1])
    num_classes = int(np.max(test_split.y_class) + 1) if test_split.y_class.size else task["num_classes"]

    ckpt = _resolve_checkpoint_for_figures(task_key, task, args)
    pred_pos, shared = _predict_with_pretrained(
        input_dim=input_dim,
        num_classes=num_classes,
        features=test_split.X,
        checkpoint_path=ckpt,
        batch_size=args.batch_size,
        device=ensure_device(),
    )
    errors_cm = np.linalg.norm(pred_pos - test_split.y_pos, axis=1) * 100.0

    np.save(task_dir / f"v3_{task_key}_seed{args.seed}_errors_cm.npy", errors_cm)
    np.save(task_dir / f"v3_{task_key}_seed{args.seed}_features.npy", shared)
    np.save(task_dir / f"v3_{task_key}_seed{args.seed}_labels.npy", test_split.y_class)

    cdf_errors: Dict[str, np.ndarray] = {"Proposed_V3": errors_cm}
    if task_key == "classroom" and args.classroom_baseline:
        cdf_errors.update(_baseline_map(task_key))

    _plot_cdf(errors_cm, cdf_errors, task_dir, task_key)
    _plot_tsne(
        shared,
        test_split.y_class,
        task["class_labels"],
        task["class_colors"],
        task_dir,
        task_key,
        per_class=task["tsne_per_class"],
        max_points=task["tsne_max_points"],
    )

    cep50 = _safe_percentile(errors_cm, 50)
    cep95 = _safe_percentile(errors_cm, 95)
    print(f"[{task_key}] saved figures. CEP50={cep50:.2f} cm, CEP95={cep95:.2f} cm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3 full pipeline + figure bundle")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "figures"],
        help="train: run training + eval + figure output. figures: regenerate figures from existing checkpoints",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "lab", "classroom"],
        help="Run lab / classroom / all",
    )
    parser.add_argument("--seed", type=int, default=44, help="Random seed / run id")
    parser.add_argument("--profile", default="base", help="Input feature profile")
    parser.add_argument("--batch-size", type=int, default=128, help="Training/inference batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=45, help="Early-stop patience")
    parser.add_argument("--min-epochs", type=int, default=60, help="Minimum epochs before early stop")
    parser.add_argument("--cls-weight", type=float, default=2.5, help="Classification loss scale")
    parser.add_argument("--conf-weight", type=float, default=0.7, help="Confidence loss scale")
    parser.add_argument("--pos-weight", type=float, default=3.0, help="Position loss scale")
    parser.add_argument("--nlos-boost", type=float, default=0.5, help="NLOS position-upweight")
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="In train mode, skip CDF/t-SNE generation.",
    )
    parser.add_argument(
        "--classroom-baseline",
        action="store_true",
        help="For classroom, add LSTM/FC-SVM/EKF baseline overlays in CDF.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild dataset cache from raw files.",
    )
    parser.add_argument(
        "--trace-data",
        action="store_true",
        help="Print raw/filter/point-wise split trace before preparing cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = ["lab", "classroom"] if "all" in args.tasks else args.tasks

    if args.mode == "figures":
        for task_key in requested:
            run_figure_task(task_key, TASKS[task_key], args)
        print(f"\nDone. figures only -> {OUTPUT_ROOT}")
        return

    rows: List[Dict[str, float]] = []
    for task_key in requested:
        rows.append(run_training_task(task_key, TASKS[task_key], args))

    _append_rows(PIPELINE_SUMMARY, rows)
    print(f"\nDone. outputs -> {OUTPUT_ROOT}")
    print(f"summary -> {PIPELINE_SUMMARY}")


if __name__ == "__main__":
    main()
