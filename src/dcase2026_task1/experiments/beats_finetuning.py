from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from tqdm import tqdm

from dcase2026_task1.data.splits import (
    DEFAULT_BSD_SPLIT_SEED,
    get_experiment_records,
)
from dcase2026_task1.models.beats import BEATs, BEATsConfig

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*LeafSpec.*deprecated.*"
)

DEFAULT_WANDB_PROJECT = "dcase2026-task1"
DEFAULT_BSD10K_ROOT = Path.home() / "data" / "BSD10k"
DEFAULT_BSD35K_ROOT = Path.home() / "data" / "BSD35k-CS"
DEFAULT_CHECKPOINT_DIR = (
    Path("/opt/scratch/dcase2026_task1/checkpoints")
    if Path("/opt/scratch").exists()
    else Path.home() / "checkpoints"
)

DEFAULT_OUTPUT_ROOT = (
    Path("/opt/scratch/dcase2026_task1/beats_finetuning")
    if Path("/opt/scratch").exists()
    else Path("outputs/beats_finetuning")
)

DEFAULT_CHECKPOINT_ALIAS = "beats_iter3plus_as2m"


@dataclass(frozen=True)
class LabelSpec:
    label_id: int
    dataset_class_idx: int
    class_name: str


class WaveformClassificationDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        indices: list[int],
        label_map: dict[int, int],
        target_sample_rate: int,
    ) -> None:
        self.records = records
        self.indices = indices
        self.label_map = label_map
        self.target_sample_rate = target_sample_rate

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from dcase2026_task1.data.datasets import load_audio_waveform

        record = self.records[self.indices[index]]
        waveform, sample_rate = load_audio_waveform(record["audio_path"])
        waveform = waveform.mean(axis=0)

        if sample_rate != self.target_sample_rate:
            waveform = _resample_audio(waveform, sample_rate, self.target_sample_rate)

        return {
            "waveform": waveform.astype(np.float32, copy=False),
            "label": self.label_map[int(record["class_idx"])],
            "sound_id": int(record["sound_id"]),
            "source_dataset": str(record["source_dataset"]),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the original Microsoft BEATs model on BSD datasets with PyTorch Lightning."
    )
    parser.add_argument("--bsd10k-root", default=None)
    parser.add_argument("--bsd35k-root", default=None)
    parser.add_argument(
        "--include-bsd35k-cs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add BSD35k-CS to the training split only.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory used for auto-downloaded BEATs checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-alias",
        choices=[DEFAULT_CHECKPOINT_ALIAS],
        default=DEFAULT_CHECKPOINT_ALIAS,
        help="Official BEATs checkpoint alias to download when --checkpoint-path is not provided.",
    )

    parser.add_argument(
        "--trust-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow torch.load(..., weights_only=False) for original BEATs checkpoints. "
            "Disable with --no-trust-checkpoint for untrusted files."
        ),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    return parser


def resolve_dataset_roots(
    bsd10k_root: str | None,
    bsd35k_root: str | None,
) -> dict[str, Path]:
    return {
        "BSD10k": Path(bsd10k_root) if bsd10k_root is not None else DEFAULT_BSD10K_ROOT,
        "BSD35k-CS": Path(bsd35k_root) if bsd35k_root is not None else DEFAULT_BSD35K_ROOT,
    }


def build_label_specs(records: list[dict[str, Any]]) -> list[LabelSpec]:
    unique: dict[int, str] = {}
    for record in records:
        class_idx = int(record["class_idx"])
        class_name = str(record["class"])
        existing = unique.get(class_idx)
        if existing is not None and existing != class_name:
            raise ValueError(
                f"Conflicting class names for class_idx={class_idx}: {existing!r} vs {class_name!r}."
            )
        unique[class_idx] = class_name

    return [
        LabelSpec(label_id=label_id, dataset_class_idx=class_idx, class_name=unique[class_idx])
        for label_id, class_idx in enumerate(sorted(unique))
    ]


def build_label_map(label_specs: list[LabelSpec]) -> dict[int, int]:
    return {spec.dataset_class_idx: spec.label_id for spec in label_specs}


def build_id2label(label_specs: list[LabelSpec]) -> dict[int, str]:
    return {spec.label_id: spec.class_name for spec in label_specs}


def create_experiment_dir(output_root: Path, include_bsd35k_cs: bool) -> Path:
    dataset_name = "BSD10k_plus_BSD35k-CS" if include_bsd35k_cs else "BSD10k"
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_beats_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def maybe_limit(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return indices
    return indices[:limit]


def resolve_checkpoint_path(
    checkpoint_dir: str | Path,
    checkpoint_alias: str
) -> Path:

    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    destination = resolved_checkpoint_dir / f"{checkpoint_alias}.pt"
    if not destination.exists():
        raise FileNotFoundError(f"Checkpoint not found at {destination}.")

    return destination


def validate_checkpoint_file(path: Path) -> None:
    header = path.read_bytes()[:512]
    lowered = header.lower().lstrip()
    if (
        lowered.startswith(b"<!doctype html")
        or lowered.startswith(b"<html")
        or lowered.startswith(b"<?xml")
    ):
        raise ValueError(
            f"{path} does not look like a PyTorch checkpoint. It looks like an HTML/XML response instead, "
            "which usually means the OneDrive share page was downloaded instead of the checkpoint file. "
            f"Delete {path} and re-run with a real checkpoint file via --checkpoint-path, or provide a direct "
            "download URL via --checkpoint-url."
        )
    if b"<title>" in lowered[:256] or b"onedrive" in lowered[:256]:
        raise ValueError(
            f"{path} appears to contain a web page instead of checkpoint bytes. "
            f"Delete {path} and re-run with --checkpoint-path pointing at a valid .pt file."
        )


def _resample_audio(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=False)

    import torch
    import torchaudio.functional as F

    audio = torch.as_tensor(waveform, dtype=torch.float32).unsqueeze(0)
    resampled = F.resample(audio, source_sample_rate, target_sample_rate).squeeze(0)
    return resampled.numpy().astype(np.float32, copy=False)


def collate_waveforms(features: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    lengths = [len(feature["waveform"]) for feature in features]
    max_length = max(lengths)
    batch_size = len(features)

    waveforms = torch.zeros((batch_size, max_length), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
    labels = torch.tensor([feature["label"] for feature in features], dtype=torch.long)

    for index, feature in enumerate(features):
        waveform = torch.from_numpy(feature["waveform"])
        length = waveform.shape[0]
        waveforms[index, :length] = waveform
        padding_mask[index, :length] = False

    return {
        "waveforms": waveforms,
        "padding_mask": padding_mask,
        "labels": labels,
    }


def split_waveforms_into_segments(
    waveforms: Any,
    padding_mask: Any,
    max_segment_samples: int,
) -> tuple[Any, Any, Any]:
    import torch

    batch_size, waveform_length = waveforms.shape
    segment_waveforms: list[Any] = []
    segment_padding_masks: list[Any] = []
    segment_batch_indices: list[int] = []

    for batch_index in range(batch_size):
        if padding_mask is None:
            valid_length = waveform_length
        else:
            valid_length = int((~padding_mask[batch_index]).sum().item())
        valid_length = max(valid_length, 1)

        sample_waveform = waveforms[batch_index]
        for start in range(0, valid_length, max_segment_samples):
            stop = min(start + max_segment_samples, valid_length)
            segment_length = stop - start

            segment = torch.zeros(
                max_segment_samples,
                dtype=waveforms.dtype,
                device=waveforms.device,
            )
            segment[:segment_length] = sample_waveform[start:stop]
            segment_waveforms.append(segment)

            segment_mask = torch.ones(
                max_segment_samples,
                dtype=torch.bool,
                device=waveforms.device,
            )
            segment_mask[:segment_length] = False
            segment_padding_masks.append(segment_mask)
            segment_batch_indices.append(batch_index)

    return (
        torch.stack(segment_waveforms, dim=0),
        torch.stack(segment_padding_masks, dim=0),
        torch.tensor(segment_batch_indices, dtype=torch.long, device=waveforms.device),
    )


def mean_segment_logits(
    segment_logits: Any,
    segment_batch_indices: Any,
    batch_size: int,
) -> Any:
    import torch

    logits = torch.zeros(
        (batch_size, segment_logits.shape[-1]),
        dtype=segment_logits.dtype,
        device=segment_logits.device,
    )
    counts = torch.zeros(batch_size, dtype=segment_logits.dtype, device=segment_logits.device)

    logits.index_add_(0, segment_batch_indices, segment_logits)
    counts.index_add_(
        0,
        segment_batch_indices,
        torch.ones(segment_batch_indices.shape[0], dtype=segment_logits.dtype, device=segment_logits.device),
    )
    return logits / counts.unsqueeze(-1).clamp_min(1.0)


def compute_classification_metrics(
    logits: Any,
    labels: Any,
    num_labels: int,
    id2label: dict[int, str] | None = None,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    predictions = np.asarray(logits).argmax(axis=-1)
    labels_np = np.asarray(labels)
    all_labels = list(range(num_labels))
    metrics = {
        "accuracy": float(accuracy_score(labels_np, predictions)),
        "balanced_accuracy": float(
            recall_score(
                labels_np,
                predictions,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels_np,
                predictions,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }
    if id2label is not None:
        y_true = [id2label[int(label)] for label in labels_np.tolist()]
        y_pred = [id2label[int(prediction)] for prediction in predictions.tolist()]
        hierarchical_metrics = compute_hierarchical_metrics(y_true, y_pred)
        metrics.update(hierarchical_metrics)
    return metrics


def compute_hierarchical_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, float]:
    def partial_match(y_t: str, y_p: str, d: float = 0.75) -> float:
        if y_t == y_p:
            return 1.0
        if y_t.split("-")[0] == y_t.split("-")[0]:
            return d / 2
        return 0.0

    class_hierarchical_precision = {
        class_name: np.mean(
            [
                partial_match(target_class, predicted_class)
                for target_class, predicted_class in zip(y_true, y_pred, strict=True)
                if predicted_class == class_name
            ]
        ).item() if class_name in y_pred else 0.0
        for class_name in set(y_true)
    }
    class_hierarchical_recall = {
        class_name: np.mean(
            [
                partial_match(target_class, predicted_class)
                for target_class, predicted_class in zip(y_true, y_pred, strict=True)
                if target_class == class_name
            ]
        ).item()
        for class_name in set(y_true)
    }
    class_hierarchical_f1 = {
        class_name: (
            2 * class_hierarchical_precision[class_name] * class_hierarchical_recall[class_name]
        ) / (class_hierarchical_precision[class_name] + class_hierarchical_recall[class_name])
        for class_name in set(y_true)
    }
    return {
        "hierarchical_precision": np.mean(list(class_hierarchical_precision.values())).item(),
        "hierarchical_recall": np.mean(list(class_hierarchical_recall.values())).item(),
        "hierarchical_f1": np.mean(list(class_hierarchical_f1.values())).item(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_devices_argument(devices: str) -> Any:
    normalized = devices.strip().lower()
    if normalized == "auto":
        return "auto"
    if "," in normalized:
        return [int(part.strip()) for part in normalized.split(",") if part.strip()]
    return int(normalized)


def _get_lightning_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import WandbLogger
        return pl, ModelCheckpoint, LearningRateMonitor, WandbLogger
    except ImportError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger
        return pl, ModelCheckpoint, LearningRateMonitor, WandbLogger


def _get_progress_bar_callback(pl: Any) -> Any:
    try:
        from lightning.pytorch.callbacks import TQDMProgressBar
        return TQDMProgressBar(refresh_rate=1)
    except ImportError:
        from pytorch_lightning.callbacks import TQDMProgressBar
        return TQDMProgressBar(refresh_rate=1)


def run_experiment(args: argparse.Namespace) -> Path:
    import torch

    pl, ModelCheckpoint, LearningRateMonitor, WandbLogger = _get_lightning_runtime()
    progress_bar = _get_progress_bar_callback(pl)

    dataset_roots = resolve_dataset_roots(args.bsd10k_root, args.bsd35k_root)
    train_records, val_records, test_records = get_experiment_records(
        bsd10k_root=dataset_roots["BSD10k"],
        bsd35k_root=dataset_roots["BSD35k-CS"],
        include_bsd35k_cs=args.include_bsd35k_cs,
        fold=args.fold,
        n_splits=args.n_splits,
        validation_size=args.validation_size,
    )
    label_specs = build_label_specs(train_records + val_records + test_records)
    label_map = build_label_map(label_specs)
    id2label = build_id2label(label_specs)
    clean_train_size = sum(record["source_dataset"] == "BSD10k" for record in train_records)
    noisy_train_size = sum(record["source_dataset"] == "BSD35k-CS" for record in train_records)
    experiment_dir = create_experiment_dir(Path(args.output_root), args.include_bsd35k_cs)

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_alias=args.checkpoint_alias
    )
    validate_checkpoint_file(checkpoint_path)
    if not args.trust_checkpoint:
        raise ValueError(
            "Original BEATs checkpoints require torch.load(..., weights_only=False). "
            "Re-run with --trust-checkpoint only if the checkpoint source is trusted."
        )
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    config = BEATsConfig(checkpoint["cfg"])
    config.finetuned_model = False
    sample_rate = 16000

    train_indices = maybe_limit(list(range(len(train_records))), args.max_train_items)
    val_indices = maybe_limit(list(range(len(val_records))), args.max_val_items)
    test_indices = maybe_limit(list(range(len(test_records))), args.max_test_items)

    train_dataset = WaveformClassificationDataset(train_records, train_indices, label_map, sample_rate)
    val_dataset = WaveformClassificationDataset(val_records, val_indices, label_map, sample_rate)
    test_dataset = WaveformClassificationDataset(test_records, test_indices, label_map, sample_rate)

    class BSDDataModule(pl.LightningDataModule):
        def train_dataloader(self) -> Any:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_waveforms,
                pin_memory=True,
            )

        def val_dataloader(self) -> Any:
            return torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_waveforms,
                pin_memory=True,
            )

        def test_dataloader(self) -> Any:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_waveforms,
                pin_memory=True,
            )

    class BEATsLightningModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.save_hyperparameters(
                {
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "warmup_steps": args.warmup_steps,
                    "num_labels": len(label_specs),
                    "freeze_encoder": args.freeze_encoder,
                    "checkpoint_path": str(checkpoint_path),
                }
            )

            self.beats = BEATs(config)
            state_dict = {
                key: value
                for key, value in checkpoint["model"].items()
                if not key.startswith("predictor.")
            }
            missing_keys, unexpected_keys = self.beats.load_state_dict(state_dict, strict=False)
            if unexpected_keys:
                raise RuntimeError(f"Unexpected BEATs checkpoint keys: {unexpected_keys}")
            non_predictor_missing = [
                key for key in missing_keys if not key.startswith("predictor")
            ]
            if non_predictor_missing:
                raise RuntimeError(f"Missing BEATs checkpoint keys: {non_predictor_missing}")

            if args.freeze_encoder:
                for parameter in self.beats.parameters():
                    parameter.requires_grad = False

            self.dropout = torch.nn.Dropout(args.head_dropout)
            self.classifier = torch.nn.Linear(config.encoder_embed_dim, len(label_specs))
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.validation_outputs: list[dict[str, Any]] = []
            self.test_outputs: list[dict[str, Any]] = []
            self.max_audio_seconds = 10
            self.sample_rate = sample_rate

        def forward(self, waveforms: Any, padding_mask: Any) -> Any:
            max_segment_samples = self.max_audio_seconds * self.sample_rate
            segmented_waveforms, segmented_padding_mask, segment_batch_indices = split_waveforms_into_segments(
                waveforms,
                padding_mask,
                max_segment_samples=max_segment_samples,
            )
            features, feature_padding_mask = self.beats.extract_features(
                segmented_waveforms,
                padding_mask=segmented_padding_mask,
            )
            if feature_padding_mask is not None:
                valid = (~feature_padding_mask).unsqueeze(-1)
                pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            else:
                pooled = features.mean(dim=1)
            segment_logits = self.classifier(self.dropout(pooled))
            return mean_segment_logits(
                segment_logits,
                segment_batch_indices,
                batch_size=waveforms.shape[0],
            )

        def _log_wandb_confusion_matrix(
            self,
            split: str,
            labels: np.ndarray,
            predictions: np.ndarray,
        ) -> None:
            if args.wandb_mode == "disabled":
                return
            logger = getattr(self, "logger", None)
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                return
            try:
                import wandb
            except ImportError:
                return
            class_names = [spec.class_name for spec in label_specs]
            experiment.log(
                {
                    f"{split}/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=labels.tolist(),
                        preds=predictions.tolist(),
                        class_names=class_names,
                    ),
                },
                step=self.global_step,
            )

        def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
            logits = self(batch["waveforms"], batch["padding_mask"])
            loss = self.loss_fn(logits, batch["labels"])
            accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["labels"].size(0))
            self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["labels"].size(0))
            return loss

        def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
            logits = self(batch["waveforms"], batch["padding_mask"])
            loss = self.loss_fn(logits, batch["labels"])
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["labels"].size(0))
            self.validation_outputs.append(
                {
                    "logits": logits.detach().cpu().numpy(),
                    "labels": batch["labels"].detach().cpu().numpy(),
                }
            )
            return loss

        def on_validation_epoch_end(self) -> None:
            if not self.validation_outputs:
                return
            logits = np.concatenate([item["logits"] for item in self.validation_outputs], axis=0)
            labels = np.concatenate([item["labels"] for item in self.validation_outputs], axis=0)
            predictions = logits.argmax(axis=-1)
            metrics = compute_classification_metrics(logits, labels, len(label_specs), id2label=id2label)
            self.log_dict({f"val/{key}": value for key, value in metrics.items()}, prog_bar=True)
            self._log_wandb_confusion_matrix("val", labels, predictions)
            self.validation_outputs.clear()

        def test_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
            logits = self(batch["waveforms"], batch["padding_mask"])
            loss = self.loss_fn(logits, batch["labels"])
            self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
            self.test_outputs.append(
                {
                    "logits": logits.detach().cpu().numpy(),
                    "labels": batch["labels"].detach().cpu().numpy(),
                }
            )
            return loss

        def on_test_epoch_end(self) -> None:
            if not self.test_outputs:
                return
            logits = np.concatenate([item["logits"] for item in self.test_outputs], axis=0)
            labels = np.concatenate([item["labels"] for item in self.test_outputs], axis=0)
            predictions = logits.argmax(axis=-1)
            metrics = compute_classification_metrics(logits, labels, len(label_specs), id2label=id2label)
            self.log_dict({f"test/{key}": value for key, value in metrics.items()})
            self._log_wandb_confusion_matrix("test", labels, predictions)
            self.test_outputs.clear()

        def configure_optimizers(self) -> Any:
            optimizer = torch.optim.AdamW(
                (parameter for parameter in self.parameters() if parameter.requires_grad),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            if args.warmup_steps <= 0:
                return optimizer

            def lr_lambda(current_step: int) -> float:
                if current_step < args.warmup_steps:
                    return float(current_step + 1) / float(max(1, args.warmup_steps))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

    experiment_config = {
        "dataset": "BSD10k",
        "include_bsd35k_cs": args.include_bsd35k_cs,
        "dataset_roots": {name: str(path) for name, path in dataset_roots.items()},
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "checkpoint_alias": args.checkpoint_alias,
        "trust_checkpoint": args.trust_checkpoint,
        "num_labels": len(label_specs),
        "labels": [
            {
                "label_id": spec.label_id,
                "dataset_class_idx": spec.dataset_class_idx,
                "class_name": spec.class_name,
            }
            for spec in label_specs
        ],
        "split": {
            "fold": args.fold,
            "n_splits": args.n_splits,
            "validation_size": args.validation_size,
            "split_seed": DEFAULT_BSD_SPLIT_SEED,
            "test_dataset": "BSD10k",
            "validation_dataset": "BSD10k",
            "clean_train_size": clean_train_size,
            "noisy_train_size": noisy_train_size,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "test_size": len(test_indices),
        },
        "training": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "head_dropout": args.head_dropout,
            "max_epochs": args.max_epochs,
            "warmup_steps": args.warmup_steps,
            "gradient_clip_val": args.gradient_clip_val,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "freeze_encoder": args.freeze_encoder,
            "precision": args.precision,
            "devices": args.devices,
            "accelerator": args.accelerator,
            "sample_rate": sample_rate,
            "seed": args.seed,
        },
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
            "run_name": experiment_dir.name,
        },
        "beats_checkpoint_cfg": checkpoint["cfg"],
    }
    write_json(experiment_dir / "config.json", experiment_config)

    logger = None
    if args.wandb_mode != "disabled":
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_dir=str(experiment_dir),
            name=experiment_dir.name,
            mode=args.wandb_mode,
        )
        logger.experiment.config.update(experiment_config, allow_val_change=True)

    model_checkpoint = ModelCheckpoint(
        dirpath=str(experiment_dir / "checkpoints"),
        filename="epoch{epoch:02d}-step{step:06d}",
        monitor="val/hierarchical_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        default_root_dir=str(experiment_dir),
        accelerator=args.accelerator,
        devices=parse_devices_argument(args.devices),
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[model_checkpoint, LearningRateMonitor(logging_interval="step"), progress_bar],
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datamodule = BSDDataModule()
    lightning_module = BEATsLightningModule()
    trainer.fit(lightning_module, datamodule=datamodule)
    test_results = trainer.test(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path="best",
    )

    summary = {
        "best_model_path": model_checkpoint.best_model_path,
        "best_model_score": (
            float(model_checkpoint.best_model_score.item())
            if model_checkpoint.best_model_score is not None
            else None
        ),
        "test_results": test_results,
    }
    write_json(experiment_dir / "summary.json", summary)

    if logger is not None:
        logger.experiment.finish()

    return experiment_dir


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = run_experiment(args)
    print(f"Wrote BEATs fine-tuning outputs to {experiment_dir}")


if __name__ == "__main__":
    main()
