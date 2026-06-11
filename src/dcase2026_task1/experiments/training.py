from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import torch
from tqdm import tqdm

from dcase2026_task1.data.splits import (
    DEFAULT_BSD_SPLIT_SEED,
    get_experiment_records,
)
from dcase2026_task1.models.bart_decoder import BartMetadataDecoder
from dcase2026_task1.models.beats import BEATs, BEATsConfig, ChunkedBEATs

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*LeafSpec.*deprecated.*"
)

DEFAULT_WANDB_PROJECT = "dcase2026-task1"
DEFAULT_BSD10K_ROOT = (
    Path("/opt/scratch/paul/data/BSD10k")
    if Path("/opt/scratch").exists()
    else Path.home() / "data" / "BSD10k"
)
DEFAULT_BSD35K_ROOT = (
    Path("/opt/scratch/paul/data/BSD35k-CS")
    if Path("/opt/scratch").exists()
    else Path.home() / "data" / "BSD35k-CS"
)
DEFAULT_CHECKPOINT_DIR = (
    Path("/opt/scratch/paul/dcase2026_task1/checkpoints")
    if Path("/opt/scratch").exists()
    else Path.home() / "checkpoints"
)

DEFAULT_OUTPUT_ROOT = (
    Path("/opt/scratch/paul/dcase2026_task1/training")
    if Path("/opt/scratch").exists()
    else Path("outputs/training")
)

DEFAULT_CHECKPOINT_ALIAS = "beats_iter3plus_as2m"
DEFAULT_EMBEDDING_MODEL = "beats"
DEFAULT_DECODER_MODEL = "none"
DEFAULT_BART_MODEL_ID = "facebook/bart-base"
MAX_RANDOM_SEED = (2**32) - 1


@dataclass(frozen=True)
class LabelSpec:
    label_id: int
    dataset_class_idx: int
    class_name: str


class AudioEmbeddingModel(Protocol):
    output_dim: int

    def __call__(
        self,
        waveforms: Any,
        padding_mask: Any,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, Any]:
        ...


class MetadataDecoderModel(Protocol):
    output_dim: int

    def __call__(
        self,
        audio_embeddings: Any,
        audio_embedding_padding_mask: Any = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> Any:
        ...


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
            "metadata": dict(record["metadata"]),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an audio classifier on BSD datasets with PyTorch Lightning."
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
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Audio embedding backbone used by the training script.",
    )
    parser.add_argument(
        "--decoder-model",
        default=DEFAULT_DECODER_MODEL,
        help="Optional metadata decoder applied after the audio encoder.",
    )
    parser.add_argument(
        "--decoder-pretrained-model-name",
        default=DEFAULT_BART_MODEL_ID,
        help="Hugging Face model id used when --decoder-model=bart.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory used for embedding-model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-alias",
        default=DEFAULT_CHECKPOINT_ALIAS,
        help="Checkpoint alias to load for the selected embedding model.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, a random 32-bit seed is chosen for the run.",
    )
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=6)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.01)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-llm-prior-embedding-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse pooled audio embeddings with a learnable class-embedding mixture built "
            "from metadata_class_probabilities."
        ),
    )
    parser.add_argument("--max-epochs", "--max_epochs", type=int, default=10)
    parser.add_argument(
        "--warmup-epochs",
        "--warmup_epochs",
        "--warmup-steps",
        dest="warmup_epochs",
        type=float,
        default=0.0,
        help="Number of epochs used for linear warmup from 0 to the base learning rate.",
    )
    parser.add_argument(
        "--lr-decay-start-epoch",
        "--lr_decay_start_epoch",
        "--lr-decay-start-step",
        dest="lr_decay_start_epoch",
        type=float,
        default=None,
        help="Epoch at which linear learning-rate decay begins after the constant phase.",
    )
    parser.add_argument("--min-learning-rate", "--min_learning_rate", type=float, default=0.0)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--precision", default="16-mixed")
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


def epochs_to_update_steps(epochs: float | None, update_steps_per_epoch: int) -> int | None:
    if epochs is None:
        return None

    return max(0, int(epochs * max(1, update_steps_per_epoch)))


def build_lr_lambda(
    warmup_steps: int,
    decay_start_step: int | None,
    total_steps: int,
    min_lr_scale: float = 0.0,
):
    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)
    min_lr_scale = min(max(0.0, min_lr_scale), 1.0)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            if warmup_steps == 1:
                return 0.0
            return float(current_step) / float(warmup_steps - 1)

        if decay_start_step is None or decay_start_step >= total_steps:
            return 1.0

        if current_step < decay_start_step:
            return 1.0

        decay_span_steps = total_steps - decay_start_step
        if decay_span_steps <= 1:
            return min_lr_scale

        decay_progress = min(current_step - decay_start_step, decay_span_steps - 1)
        decay_fraction = float(decay_progress) / float(decay_span_steps - 1)
        return 1.0 - ((1.0 - min_lr_scale) * decay_fraction)

    return lr_lambda


def build_id2label(label_specs: list[LabelSpec]) -> dict[int, str]:
    return {spec.label_id: spec.class_name for spec in label_specs}


def create_experiment_dir(
    output_root: Path,
    include_bsd35k_cs: bool,
    embedding_model: str,
) -> Path:
    dataset_name = "BSD10k_plus_BSD35k-CS" if include_bsd35k_cs else "BSD10k"
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{embedding_model}_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def maybe_limit(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return indices
    return indices[:limit]


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return random.SystemRandom().randint(0, MAX_RANDOM_SEED)


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


def load_embedding_checkpoint(args: argparse.Namespace, torch_module: Any) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_alias=args.checkpoint_alias,
    )
    validate_checkpoint_file(checkpoint_path)
    if not args.trust_checkpoint:
        raise ValueError(
            "Original BEATs checkpoints require torch.load(..., weights_only=False). "
            "Re-run with --trust-checkpoint only if the checkpoint source is trusted."
        )
    checkpoint = torch_module.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    return {
        "path": checkpoint_path,
        "checkpoint": checkpoint,
    }


def build_beats_embedding_model(
    checkpoint: dict[str, Any],
    sample_rate: int,
) -> AudioEmbeddingModel:
    config = BEATsConfig(checkpoint["cfg"])
    config.finetuned_model = False
    beats_model = BEATs(config)
    state_dict = {
        key: value
        for key, value in checkpoint["model"].items()
        if not key.startswith("predictor.")
    }
    missing_keys, unexpected_keys = beats_model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected BEATs checkpoint keys: {unexpected_keys}")
    non_predictor_missing = [
        key for key in missing_keys if not key.startswith("predictor")
    ]
    if non_predictor_missing:
        raise RuntimeError(f"Missing BEATs checkpoint keys: {non_predictor_missing}")

    model = ChunkedBEATs(
        beats_model,
        sample_rate=sample_rate,
        max_audio_seconds=10,
    )
    model.output_dim = int(config.encoder_embed_dim)
    return model


def build_embedding_model(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    sample_rate: int,
) -> AudioEmbeddingModel:
    if args.embedding_model == "beats":
        return build_beats_embedding_model(checkpoint, sample_rate=sample_rate)
    raise ValueError(f"Unsupported embedding model: {args.embedding_model!r}")


def build_metadata_decoder(
    args: argparse.Namespace,
    audio_embedding_dim: int,
) -> MetadataDecoderModel | None:
    if args.decoder_model == "none":
        return None
    if args.decoder_model == "bart":
        return BartMetadataDecoder(
            audio_embedding_dim=audio_embedding_dim,
            model_id=args.decoder_pretrained_model_name,
        )
    raise ValueError(f"Unsupported decoder model: {args.decoder_model!r}")


def pool_embedding_sequence(embedding_sequence: Any) -> Any:
    return embedding_sequence.mean(dim=1)


def masked_mean_embedding_sequence(
    embedding_sequence: Any,
    embedding_padding_mask: Any | None = None,
) -> Any:
    if embedding_padding_mask is None:
        return pool_embedding_sequence(embedding_sequence)
    valid = (~embedding_padding_mask).unsqueeze(-1)
    return (embedding_sequence * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


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
        "metadata": [dict(feature["metadata"]) for feature in features],
    }


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


def build_label2id(id2label: dict[int, str]) -> dict[str, int]:
    return {label: label_id for label_id, label in id2label.items()}


def extract_llm_label_prior(
    metadata_item: dict[str, Any] | None,
    label2id: dict[str, int],
) -> tuple[set[int], np.ndarray | None]:
    allowed_label_ids: set[int] = set()
    prior = np.zeros(len(label2id), dtype=np.float64)
    if metadata_item is None:
        return allowed_label_ids, None

    raw_predictions = metadata_item.get("metadata_class_probabilities")
    if not isinstance(raw_predictions, list):
        return allowed_label_ids, None

    found_any = False
    for item in raw_predictions:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        probability = item.get("probability")
        if label == "other" or not isinstance(label, str):
            continue
        if label not in label2id:
            continue
        if not isinstance(probability, (int, float)):
            continue
        probability_value = float(probability)
        if probability_value <= 0.0:
            continue
        label_id = label2id[label]
        allowed_label_ids.add(label_id)
        prior[label_id] = probability_value
        found_any = True

    if not found_any:
        return set(), None

    prior_sum = prior.sum()
    if prior_sum <= 0.0:
        return set(), None
    return allowed_label_ids, prior / prior_sum


def apply_hard_llm_constraints(
    logits: Any,
    metadata: list[dict[str, Any]] | None,
    id2label: dict[int, str],
) -> np.ndarray:
    logits_np = np.asarray(logits, dtype=np.float64).copy()
    if metadata is None:
        return logits_np

    label2id = build_label2id(id2label)
    if len(metadata) != logits_np.shape[0]:
        raise ValueError(
            f"Expected metadata for {logits_np.shape[0]} samples, got {len(metadata)}."
        )

    floor_value = np.finfo(logits_np.dtype).min
    for row_index, metadata_item in enumerate(metadata):
        allowed_label_ids, _ = extract_llm_label_prior(metadata_item, label2id)
        if not allowed_label_ids:
            continue
        constrained_row = np.full(logits_np.shape[1], floor_value, dtype=logits_np.dtype)
        for label_id in allowed_label_ids:
            constrained_row[label_id] = logits_np[row_index, label_id]
        logits_np[row_index] = constrained_row
    return logits_np


def apply_soft_llm_constraints(
    logits: Any,
    metadata: list[dict[str, Any]] | None,
    id2label: dict[int, str],
    epsilon: float = 1e-8,
) -> np.ndarray:
    logits_np = np.asarray(logits, dtype=np.float64).copy()
    if metadata is None:
        return logits_np

    label2id = build_label2id(id2label)
    if len(metadata) != logits_np.shape[0]:
        raise ValueError(
            f"Expected metadata for {logits_np.shape[0]} samples, got {len(metadata)}."
        )

    for row_index, metadata_item in enumerate(metadata):
        _, prior = extract_llm_label_prior(metadata_item, label2id)
        if prior is None:
            continue
        safe_prior = np.clip(prior, epsilon, None)
        logits_np[row_index] = logits_np[row_index] + np.log(safe_prior)
    return logits_np


def build_llm_prior_weights(
    metadata: list[dict[str, Any]] | None,
    id2label: dict[int, str],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    label2id = build_label2id(id2label)
    num_labels = len(id2label)
    if metadata is None:
        return torch.zeros((0, num_labels), device=device, dtype=dtype or torch.float32)

    weights = torch.zeros((len(metadata), num_labels), device=device, dtype=dtype or torch.float32)
    for row_index, metadata_item in enumerate(metadata):
        _, prior = extract_llm_label_prior(metadata_item, label2id)
        if prior is None:
            continue
        weights[row_index] = torch.as_tensor(prior, device=device, dtype=weights.dtype)
    return weights


def build_prediction_head(
    input_dim: int,
    output_dim: int,
    dropout: float,
) -> torch.nn.Sequential:
    hidden_dim = max(input_dim, output_dim)
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, output_dim),
    )


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
    seed = resolve_seed(args.seed)
    args.seed = seed

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
    experiment_dir = create_experiment_dir(
        Path(args.output_root),
        args.include_bsd35k_cs,
        args.embedding_model,
    )

    checkpoint_info = load_embedding_checkpoint(args, torch)
    checkpoint_path = checkpoint_info["path"]
    checkpoint = checkpoint_info["checkpoint"]
    sample_rate = 16000

    train_indices = maybe_limit(list(range(len(train_records))), args.max_train_items)
    val_indices = maybe_limit(list(range(len(val_records))), args.max_val_items)
    test_indices = maybe_limit(list(range(len(test_records))), args.max_test_items)

    train_dataset = WaveformClassificationDataset(train_records, train_indices, label_map, sample_rate)
    val_dataset = WaveformClassificationDataset(val_records, val_indices, label_map, sample_rate)
    test_dataset = WaveformClassificationDataset(test_records, test_indices, label_map, sample_rate)
    train_batches_per_epoch = max(1, math.ceil(len(train_dataset) / args.batch_size))
    update_steps_per_epoch = max(1, math.ceil(train_batches_per_epoch / args.accumulate_grad_batches))
    total_update_steps = max(1, update_steps_per_epoch * args.max_epochs)
    warmup_steps = epochs_to_update_steps(args.warmup_epochs, update_steps_per_epoch) or 0
    decay_start_step = epochs_to_update_steps(args.lr_decay_start_epoch, update_steps_per_epoch)

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

    class ClassificationLightningModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.save_hyperparameters(
                {
                    "embedding_model": args.embedding_model,
                    "decoder_model": args.decoder_model,
                    "decoder_pretrained_model_name": args.decoder_pretrained_model_name,
                    "learning_rate": args.learning_rate,
                    "min_learning_rate": args.min_learning_rate,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "warmup_steps": warmup_steps,
                    "lr_decay_start_epoch": args.lr_decay_start_epoch,
                    "lr_decay_start_step": decay_start_step,
                    "update_steps_per_epoch": update_steps_per_epoch,
                    "total_update_steps": total_update_steps,
                    "num_labels": len(label_specs),
                    "freeze_encoder": args.freeze_encoder,
                    "checkpoint_path": str(checkpoint_path),
                    "use_llm_prior_embedding_fusion": args.use_llm_prior_embedding_fusion,
                }
            )

            self.embedding_model = build_embedding_model(
                args,
                checkpoint=checkpoint,
                sample_rate=sample_rate,
            )
            self.metadata_decoder = build_metadata_decoder(
                args,
                audio_embedding_dim=self.embedding_model.output_dim,
            )

            if args.freeze_encoder:
                for parameter in self.embedding_model.parameters():
                    parameter.requires_grad = False

            self.dropout = torch.nn.Dropout(args.head_dropout)
            self.use_llm_prior_embedding_fusion = args.use_llm_prior_embedding_fusion
            if self.use_llm_prior_embedding_fusion:
                fusion_input_dim = self.embedding_model.output_dim * 2
                classifier_input_dim = self.embedding_model.output_dim
                self.llm_class_embedding_bank = torch.nn.Embedding(
                    len(label_specs),
                    self.embedding_model.output_dim,
                )
                self.fusion_head = build_prediction_head(
                    fusion_input_dim,
                    classifier_input_dim,
                    args.head_dropout,
                )
            else:
                classifier_input_dim = (
                    self.metadata_decoder.output_dim
                    if self.metadata_decoder is not None
                    else self.embedding_model.output_dim
                )
                self.llm_class_embedding_bank = None
                self.fusion_head = None
            self.classifier = torch.nn.Linear(classifier_input_dim, len(label_specs))
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.validation_outputs: list[dict[str, Any]] = []
            self.test_outputs: list[dict[str, Any]] = []

        def forward(
            self,
            waveforms: Any,
            padding_mask: Any,
            metadata: list[dict[str, Any]] | None = None,
        ) -> Any:
            embedding_sequence, embedding_padding_mask = self.embedding_model(
                waveforms,
                padding_mask,
                metadata=metadata,
            )
            audio_features = masked_mean_embedding_sequence(
                embedding_sequence,
                embedding_padding_mask=embedding_padding_mask,
            )
            if self.use_llm_prior_embedding_fusion:
                llm_prior_weights = build_llm_prior_weights(
                    metadata,
                    id2label=id2label,
                    device=audio_features.device,
                    dtype=audio_features.dtype,
                )
                llm_features = llm_prior_weights @ self.llm_class_embedding_bank.weight
                fused_features = torch.cat([audio_features, llm_features], dim=-1)
                features = self.fusion_head(self.dropout(fused_features))
            elif self.metadata_decoder is not None:
                features = self.metadata_decoder(
                    embedding_sequence,
                    audio_embedding_padding_mask=embedding_padding_mask,
                    metadata=metadata,
                )
            else:
                features = audio_features
            return self.classifier(self.dropout(features))

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
            logits = self(batch["waveforms"], batch["padding_mask"], metadata=batch["metadata"])
            loss = self.loss_fn(logits, batch["labels"])
            accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["labels"].size(0))
            self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["labels"].size(0))
            return loss

        def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
            logits = self(batch["waveforms"], batch["padding_mask"], metadata=batch["metadata"])
            loss = self.loss_fn(logits, batch["labels"])
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["labels"].size(0))
            self.validation_outputs.append(
                {
                    "logits": logits.detach().cpu().numpy(),
                    "labels": batch["labels"].detach().cpu().numpy(),
                    "metadata": [dict(item) for item in batch["metadata"]],
                }
            )
            return loss

        def on_validation_epoch_end(self) -> None:
            if not self.validation_outputs:
                return
            logits = np.concatenate([item["logits"] for item in self.validation_outputs], axis=0)
            labels = np.concatenate([item["labels"] for item in self.validation_outputs], axis=0)
            metadata = [
                metadata_item
                for item in self.validation_outputs
                for metadata_item in item["metadata"]
            ]
            predictions = logits.argmax(axis=-1)
            metrics = compute_classification_metrics(logits, labels, len(label_specs), id2label=id2label)
            hard_constrained_logits = apply_hard_llm_constraints(logits, metadata, id2label=id2label)
            hard_constrained_metrics = compute_classification_metrics(
                hard_constrained_logits,
                labels,
                len(label_specs),
                id2label=id2label,
            )
            soft_constrained_logits = apply_soft_llm_constraints(logits, metadata, id2label=id2label)
            soft_constrained_metrics = compute_classification_metrics(
                soft_constrained_logits,
                labels,
                len(label_specs),
                id2label=id2label,
            )
            self.log_dict({f"val/{key}": value for key, value in metrics.items()}, prog_bar=True)
            self.log_dict(
                {f"val_hard_constrained/{key}": value for key, value in hard_constrained_metrics.items()}
            )
            self.log_dict(
                {f"val_soft_constrained/{key}": value for key, value in soft_constrained_metrics.items()}
            )
            self._log_wandb_confusion_matrix("val", labels, predictions)
            self.validation_outputs.clear()

        def test_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
            logits = self(batch["waveforms"], batch["padding_mask"], metadata=batch["metadata"])
            loss = self.loss_fn(logits, batch["labels"])
            self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
            self.test_outputs.append(
                {
                    "logits": logits.detach().cpu().numpy(),
                    "labels": batch["labels"].detach().cpu().numpy(),
                    "metadata": [dict(item) for item in batch["metadata"]],
                }
            )
            return loss

        def on_test_epoch_end(self) -> None:
            if not self.test_outputs:
                return
            logits = np.concatenate([item["logits"] for item in self.test_outputs], axis=0)
            labels = np.concatenate([item["labels"] for item in self.test_outputs], axis=0)
            metadata = [
                metadata_item
                for item in self.test_outputs
                for metadata_item in item["metadata"]
            ]
            predictions = logits.argmax(axis=-1)
            metrics = compute_classification_metrics(logits, labels, len(label_specs), id2label=id2label)
            hard_constrained_logits = apply_hard_llm_constraints(logits, metadata, id2label=id2label)
            hard_constrained_metrics = compute_classification_metrics(
                hard_constrained_logits,
                labels,
                len(label_specs),
                id2label=id2label,
            )
            soft_constrained_logits = apply_soft_llm_constraints(logits, metadata, id2label=id2label)
            soft_constrained_metrics = compute_classification_metrics(
                soft_constrained_logits,
                labels,
                len(label_specs),
                id2label=id2label,
            )
            self.log_dict({f"test/{key}": value for key, value in metrics.items()})
            self.log_dict(
                {f"test_hard_constrained/{key}": value for key, value in hard_constrained_metrics.items()}
            )
            self.log_dict(
                {f"test_soft_constrained/{key}": value for key, value in soft_constrained_metrics.items()}
            )
            self._log_wandb_confusion_matrix("test", labels, predictions)
            self.test_outputs.clear()

        def configure_optimizers(self) -> Any:
            optimizer = torch.optim.AdamW(
                (parameter for parameter in self.parameters() if parameter.requires_grad),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            if warmup_steps <= 0 and decay_start_step is None:
                return optimizer

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=build_lr_lambda(
                    warmup_steps=warmup_steps,
                    decay_start_step=decay_start_step,
                    total_steps=total_update_steps,
                    min_lr_scale=(
                        args.min_learning_rate / args.learning_rate
                        if args.learning_rate > 0
                        else 0.0
                    ),
                ),
            )
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
        "embedding_model": args.embedding_model,
        "decoder_model": args.decoder_model,
        "decoder_pretrained_model_name": args.decoder_pretrained_model_name,
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
            "warmup_epochs": args.warmup_epochs,
            "warmup_steps": warmup_steps,
            "lr_decay_start_epoch": args.lr_decay_start_epoch,
            "lr_decay_start_step": decay_start_step,
            "min_learning_rate": args.min_learning_rate,
            "train_batches_per_epoch": train_batches_per_epoch,
            "update_steps_per_epoch": update_steps_per_epoch,
            "total_update_steps": total_update_steps,
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
        "embedding_checkpoint_cfg": checkpoint["cfg"],
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
        logger.experiment.config.update({"seed": seed}, allow_val_change=True)

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

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    datamodule = BSDDataModule()
    lightning_module = ClassificationLightningModule()
    trainer.fit(lightning_module, datamodule=datamodule)
    test_results = trainer.test(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=None,
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
    print(f"Wrote training outputs to {experiment_dir}")


if __name__ == "__main__":
    main()
