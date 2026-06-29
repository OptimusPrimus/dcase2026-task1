from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import torch

from dcase2026_task1.data.splits import (
    DEFAULT_BSD_SPLIT_SEED,
    get_experiment_records,
)
from dcase2026_task1.models.M2D import (
    build_m2d_embedding_model,
    build_m2d_text_encoder_embedding_model,
)
from dcase2026_task1.models.beats import build_beats_embedding_model
from dcase2026_task1.models.clap import (
    KEYWORD_METADATA_KEY,
    build_clap_embedding_model,
)
from dcase2026_task1.models.lclap import (
    build_lclap_audio_encoder,
    build_lclap_embedding_model,
    build_lclap_text_encoder,
)
from dcase2026_task1.models.passt import build_passt_embedding_model

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
DEFAULT_BSD2K_ROOT = (
    Path("/opt/scratch/paul/data/BSD2k")
    if Path("/opt/scratch").exists()
    else Path.home() / "data" / "BSD2k"
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

DEFAULT_EMBEDDING_MODEL = "beats"
DEFAULT_LLM_EMBEDDING_DIM = 512
PSEUDO_LABEL_FILENAMES = {
    "BSD10k": "bsd10k_logits.npz",
    "BSD35k-CS": "bsd35k_cs_logits.npz",
}
EMBEDDING_SAMPLE_RATES = {
    "beats": 16000,
    "clap": 32000,
    "clap_kw": 32000,
    "lclap": 48000,
    "lclap_audio": 48000,
    "lclap_text": 48000,
    "lclap_kw": 48000,
    "llm": 16000,
    "m2d": 16000,
    "m2d_te": 16000,
    "passt": 32000,
}
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


class LLMPriorEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        id2label: dict[int, str],
        *,
        output_dim: int = DEFAULT_LLM_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.id2label = dict(id2label)
        self.output_dim = output_dim
        self.class_embedding_bank = torch.nn.Embedding(len(self.id2label), output_dim)
        self.checkpoint_cfg = {
            "arch": "llm",
            "output_dim": output_dim,
        }

    def forward(
        self,
        waveforms: Any,
        padding_mask: Any | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = waveforms.shape[0]
        weights = build_llm_prior_weights(
            metadata,
            id2label=self.id2label,
            device=self.class_embedding_bank.weight.device,
            dtype=self.class_embedding_bank.weight.dtype,
        )
        if weights.shape[0] == 0:
            weights = torch.zeros(
                (batch_size, len(self.id2label)),
                device=self.class_embedding_bank.weight.device,
                dtype=self.class_embedding_bank.weight.dtype,
            )
        features = weights @ self.class_embedding_bank.weight
        padding = torch.zeros((features.shape[0], 1), device=features.device, dtype=torch.bool)
        return features.unsqueeze(1), padding


class WaveformClassificationDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        indices: list[int],
        label_map: dict[int, int],
        target_sample_rate: int,
        pseudo_labels: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.records = records
        self.indices = indices
        self.label_map = label_map
        self.target_sample_rate = target_sample_rate
        self.pseudo_labels = pseudo_labels or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from dcase2026_task1.data.datasets import load_audio_waveform

        record = self.records[self.indices[index]]
        waveform, sample_rate = load_audio_waveform(record["audio_path"])
        waveform = waveform.mean(axis=0)

        if sample_rate != self.target_sample_rate:
            waveform = _resample_audio(waveform, sample_rate, self.target_sample_rate)

        pseudo_label = None
        file_id = resolve_record_file_id(record)
        source_dataset = str(record["source_dataset"])
        for key in (f"{source_dataset}:{file_id}", file_id):
            if key in self.pseudo_labels:
                pseudo_label = self.pseudo_labels[key]
                break

        return {
            "waveform": waveform.astype(np.float32, copy=False),
            "label": self.label_map[int(record["class_idx"])],
            "pseudo_label": pseudo_label,
            "sound_id": int(record["sound_id"]),
            "source_dataset": str(record["source_dataset"]),
            "metadata": dict(record["metadata"]),
        }


class WaveformInferenceDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        target_sample_rate: int,
    ) -> None:
        self.records = records
        self.target_sample_rate = target_sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from dcase2026_task1.data.datasets import load_audio_waveform

        record = self.records[index]
        waveform, sample_rate = load_audio_waveform(record["audio_path"])
        waveform = waveform.mean(axis=0)

        if sample_rate != self.target_sample_rate:
            waveform = _resample_audio(waveform, sample_rate, self.target_sample_rate)

        return {
            "waveform": waveform.astype(np.float32, copy=False),
            "file_id": resolve_record_file_id(record),
            "metadata": dict(record["metadata"]),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an audio classifier on BSD datasets with PyTorch Lightning."
    )
    parser.add_argument("--bsd10k-root", default=None)
    parser.add_argument("--bsd35k-root", default=None)
    parser.add_argument("--bsd2k-root", default=None)
    parser.add_argument(
        "--include-bsd35k-cs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add BSD35k-CS to the training split only.",
    )
    parser.add_argument(
        "--only-bsd35k-cs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use BSD35k-CS as the training split only.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Audio embedding backbone used by the training script.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory used for embedding-model checkpoints.",
    )
    parser.add_argument(
        "--init-checkpoint-path",
        default=None,
        help=(
            "Optional training run name under --output-root, run directory, or checkpoint "
            "file used to initialize model weights before starting a new training run."
        ),
    )
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Lightning checkpoint saving during training.",
    )

    parser.add_argument(
        "--trust-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow torch.load(..., weights_only=False) for original embedding checkpoints. "
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
        "--label-smoothing", "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for cross-entropy loss.",
    )
    parser.add_argument(
        "--use-class-frequency-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use inverse-frequency class weights from the effective training split in "
            "cross-entropy loss."
        ),
    )
    parser.add_argument(
        "--pseudo-label-dir", "--pseudo_label_dir",
        default=None,
        help="Training run or ensemble directory containing pseudo labels.",
    )
    parser.add_argument(
        "--pseudo-label-weight", "--pseudo_label_weight",
        type=float,
        default=0.0,
        help="Weight for the pseudo-label loss term.",
    )
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
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience in validation epochs for val/hierarchical_f1.",
    )
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
    bsd2k_root: str | None,
) -> dict[str, Path]:
    return {
        "BSD10k": Path(bsd10k_root) if bsd10k_root is not None else DEFAULT_BSD10K_ROOT,
        "BSD35k-CS": Path(bsd35k_root) if bsd35k_root is not None else DEFAULT_BSD35K_ROOT,
        "BSD2k": Path(bsd2k_root) if bsd2k_root is not None else DEFAULT_BSD2K_ROOT,
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


def build_class_frequency_loss_weights(
    records: list[dict[str, Any]],
    indices: list[int],
    label_map: dict[int, int],
) -> torch.Tensor:
    num_labels = len(label_map)
    if num_labels == 0:
        return torch.zeros(0, dtype=torch.float32)

    counts = torch.zeros(num_labels, dtype=torch.float32)
    for index in indices:
        record = records[index]
        label_id = label_map[int(record["class_idx"])]
        counts[label_id] += 1.0

    if torch.any(counts <= 0):
        missing = torch.nonzero(counts <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            "Cannot build class-frequency loss weights with missing labels in the training set: "
            f"{missing}."
        )

    total_count = float(counts.sum().item())
    return total_count / (counts.numel() * counts)


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
    only_bsd35k_cs: bool,
    embedding_model: str,
) -> Path:
    if only_bsd35k_cs:
        dataset_name = "BSD35k-CS_train_only"
    elif include_bsd35k_cs:
        dataset_name = "BSD10k_plus_BSD35k-CS"
    else:
        dataset_name = "BSD10k"
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{embedding_model}_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def resolve_pseudo_label_dir(pseudo_label_dir: str | None, output_root: str | Path) -> Path | None:
    if pseudo_label_dir is None:
        return None
    path = Path(pseudo_label_dir).expanduser()
    if path.exists():
        return path
    return Path(output_root).expanduser() / path


def resolve_checkpoint_path(checkpoint_dir: str | Path, checkpoint_alias: str) -> Path:
    from dcase2026_task1.models.beats import resolve_checkpoint_path as resolve_beats_checkpoint_path

    return resolve_beats_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        checkpoint_alias=checkpoint_alias,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def resolve_initial_checkpoint_path(
    init_checkpoint_path: str | Path,
    output_root: str | Path,
) -> Path:
    path = Path(init_checkpoint_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(output_root).expanduser() / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.is_dir():
            return resolve_training_run_checkpoint_path(candidate)

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Initial checkpoint {str(init_checkpoint_path)!r} was not found as a checkpoint file "
        f"or training run directory. Searched: {searched}."
    )


def resolve_training_run_checkpoint_path(run_dir: str | Path) -> Path:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    summary_path = resolved_run_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        best_model_path = summary.get("best_model_path")
        if isinstance(best_model_path, str) and best_model_path:
            checkpoint_path = Path(best_model_path).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = resolved_run_dir / checkpoint_path
            if checkpoint_path.is_file():
                return checkpoint_path.resolve()
            raise FileNotFoundError(
                f"Best checkpoint from {summary_path} does not exist: {checkpoint_path}."
            )

    checkpoint_paths = sorted((resolved_run_dir / "checkpoints").glob("*.ckpt"))
    if len(checkpoint_paths) == 1:
        return checkpoint_paths[0].resolve()
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No initial checkpoint found for training run {resolved_run_dir}. "
            "Expected summary.json with best_model_path or one .ckpt file under checkpoints/."
        )
    raise ValueError(
        f"Training run {resolved_run_dir} contains multiple checkpoint files and no "
        "summary.json best_model_path to choose from."
    )


def maybe_limit(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return indices
    return indices[:limit]


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return random.SystemRandom().randint(0, MAX_RANDOM_SEED)


def build_embedding_model(
    args: argparse.Namespace,
    sample_rate: int,
    id2label: dict[int, str] | None = None,
) -> AudioEmbeddingModel:
    if args.embedding_model == "llm":
        if id2label is None:
            raise ValueError("id2label is required for the llm embedding model.")
        return LLMPriorEmbeddingModel(id2label=id2label)
    if args.embedding_model == "beats":
        return build_beats_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "passt":
        return build_passt_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "m2d":
        return build_m2d_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "m2d_te":
        return build_m2d_text_encoder_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "clap":
        return build_clap_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "clap_kw":
        return build_clap_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
            metadata_text_key=KEYWORD_METADATA_KEY,
            arch="clap_kw",
        )
    if args.embedding_model == "lclap":
        return build_lclap_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "lclap_audio":
        return build_lclap_audio_encoder(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "lclap_text":
        return build_lclap_text_encoder(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
        )
    if args.embedding_model == "lclap_kw":
        return build_lclap_embedding_model(
            checkpoint_dir=args.checkpoint_dir,
            trust_checkpoint=args.trust_checkpoint,
            sample_rate=sample_rate,
            metadata_text_key=KEYWORD_METADATA_KEY,
            arch="lclap_kw",
        )
    raise ValueError(f"Unsupported embedding model: {args.embedding_model!r}")


def resolve_embedding_sample_rate(embedding_model: str) -> int:
    try:
        return EMBEDDING_SAMPLE_RATES[embedding_model]
    except KeyError as exc:
        raise ValueError(f"Unsupported embedding model: {embedding_model!r}") from exc


def load_initial_training_state_dict(checkpoint_path: str | Path) -> dict[str, Any]:
    resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(str(resolved_checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Training checkpoint at {resolved_checkpoint_path} must contain a dict payload."
        )
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Training checkpoint at {resolved_checkpoint_path} does not contain a valid state_dict."
        )
    return state_dict


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
    pseudo_label_dim = next(
        (
            len(feature["pseudo_label"])
            for feature in features
            if feature.get("pseudo_label") is not None
        ),
        0,
    )
    pseudo_labels = torch.zeros((batch_size, pseudo_label_dim), dtype=torch.float32)
    pseudo_label_mask = torch.zeros(batch_size, dtype=torch.bool)

    for index, feature in enumerate(features):
        waveform = torch.from_numpy(feature["waveform"])
        length = waveform.shape[0]
        waveforms[index, :length] = waveform
        padding_mask[index, :length] = False
        if feature.get("pseudo_label") is not None:
            pseudo_labels[index] = torch.as_tensor(feature["pseudo_label"], dtype=torch.float32)
            pseudo_label_mask[index] = True

    return {
        "waveforms": waveforms,
        "padding_mask": padding_mask,
        "labels": labels,
        "pseudo_labels": pseudo_labels,
        "pseudo_label_mask": pseudo_label_mask,
        "metadata": [dict(feature["metadata"]) for feature in features],
    }


def collate_inference_waveforms(features: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(feature["waveform"]) for feature in features]
    max_length = max(lengths)
    batch_size = len(features)

    waveforms = torch.zeros((batch_size, max_length), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)

    for index, feature in enumerate(features):
        waveform = torch.from_numpy(feature["waveform"])
        length = waveform.shape[0]
        waveforms[index, :length] = waveform
        padding_mask[index, :length] = False

    return {
        "waveforms": waveforms,
        "padding_mask": padding_mask,
        "file_ids": [str(feature["file_id"]) for feature in features],
        "metadata": [dict(feature["metadata"]) for feature in features],
    }


def resolve_record_file_id(record: dict[str, Any]) -> str:
    anonymous_id = record.get("anonymous_id")
    if anonymous_id not in (None, ""):
        return str(anonymous_id)
    sound_id = record.get("sound_id")
    if sound_id not in (None, ""):
        return str(sound_id)
    audio_path = record.get("audio_path")
    if audio_path not in (None, ""):
        return Path(str(audio_path)).stem
    raise KeyError(f"Could not resolve file_id for record: {record!r}")


def _label_names(label_specs: list[LabelSpec]) -> list[str]:
    return [spec.class_name for spec in sorted(label_specs, key=lambda spec: spec.label_id)]


def _reorder_vector(
    values: Any,
    source_labels: list[str],
    target_labels: list[str],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if source_labels == target_labels:
        return array
    by_label = {label: array[index] for index, label in enumerate(source_labels)}
    return np.asarray([by_label[label] for label in target_labels], dtype=np.float32)


def _load_pseudo_labels_json(path: Path, label_specs: list[LabelSpec]) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_labels = _label_names(label_specs)
    source_labels = [str(label) for label in payload["label_names"]]
    datasets = payload.get("datasets", {})
    pseudo_labels: dict[str, np.ndarray] = {}
    for dataset_name, json_key in (
        ("BSD10k", "BSD10k"),
        ("BSD35k-CS", "BSD35k-CS"),
        ("BSD35k-CS", "BSD35k"),
    ):
        for row in datasets.get(json_key, []):
            file_id = str(row["file_id"])
            probabilities = _reorder_vector(row["probabilities"], source_labels, target_labels)
            probabilities = probabilities / probabilities.sum()
            pseudo_labels[file_id] = probabilities
            pseudo_labels[f"{dataset_name}:{file_id}"] = probabilities
    return pseudo_labels


def _load_pseudo_labels_npz(path: Path, dataset_name: str, label_specs: list[LabelSpec]) -> dict[str, np.ndarray]:
    target_labels = _label_names(label_specs)
    pseudo_labels: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        source_labels = [str(label) for label in data["label_names"].tolist()]
        for file_id in data.files:
            if file_id == "label_names":
                continue
            logits = _reorder_vector(data[file_id], source_labels, target_labels)
            probabilities = torch.softmax(torch.as_tensor(logits), dim=0).numpy()
            pseudo_labels[file_id] = probabilities
            pseudo_labels[f"{dataset_name}:{file_id}"] = probabilities
    return pseudo_labels


def load_pseudo_labels(path: str | Path, label_specs: list[LabelSpec]) -> dict[str, np.ndarray]:
    pseudo_label_dir = Path(path).expanduser()
    predictions_json = pseudo_label_dir / "predictions.json"
    if predictions_json.exists():
        return _load_pseudo_labels_json(predictions_json, label_specs)

    pseudo_labels: dict[str, np.ndarray] = {}
    for dataset_name, filename in PSEUDO_LABEL_FILENAMES.items():
        logits_path = pseudo_label_dir / filename
        if logits_path.exists():
            pseudo_labels.update(_load_pseudo_labels_npz(logits_path, dataset_name, label_specs))
    if not pseudo_labels:
        raise FileNotFoundError(f"No pseudo labels found in {pseudo_label_dir}.")
    return pseudo_labels


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def load_full_dataset_records(dataset_name: str, root: Path) -> list[dict[str, Any]]:
    from dcase2026_task1.data.datasets import BSDDataset

    dataset = BSDDataset(root=root, dataset_name=dataset_name, load_audio=False)
    return list(dataset.records)


def predict_logits_for_records(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    sample_rate: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    dataset = WaveformInferenceDataset(records=records, target_sample_rate=sample_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_inference_waveforms,
        pin_memory=(device.type == "cuda"),
    )

    all_file_ids: list[str] = []
    all_logits: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            waveforms = batch["waveforms"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            logits = model(waveforms, padding_mask, metadata=batch["metadata"])
            all_file_ids.extend(batch["file_ids"])
            all_logits.append(logits.detach().cpu().numpy())

    if not all_logits:
        return all_file_ids, np.zeros((0, 0), dtype=np.float32)
    return all_file_ids, np.concatenate(all_logits, axis=0)


def write_logits_npz(
    path: Path,
    file_ids: list[str],
    logits: np.ndarray,
    label_specs: list[LabelSpec],
) -> None:
    payload: dict[str, np.ndarray] = {
        file_id: np.asarray(row, dtype=np.float32)
        for file_id, row in zip(file_ids, logits, strict=True)
    }
    payload["label_names"] = np.asarray(
        [spec.class_name for spec in label_specs],
        dtype=np.str_,
    )
    np.savez(path, **payload)


def load_model_for_prediction_exports(
    *,
    lightning_module: torch.nn.Module,
    build_model: Any,
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> torch.nn.Module:
    if checkpoint_path is None:
        prediction_model = lightning_module
    else:
        resolved_checkpoint_path = Path(checkpoint_path)
        if not resolved_checkpoint_path.exists():
            prediction_model = lightning_module
        else:
            checkpoint = torch.load(
                str(resolved_checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise RuntimeError(
                    f"Training checkpoint at {resolved_checkpoint_path} does not contain a state_dict."
                )
            prediction_model = build_model()
            prediction_model.load_state_dict(checkpoint["state_dict"])

    prediction_model.to(device)
    prediction_model.eval()
    return prediction_model


def compute_classification_metrics(
    logits: Any,
    labels: Any,
    num_labels: int,
    id2label: dict[int, str] | None = None,
    include_class_wise_hierarchical: bool = False,
) -> dict[str, Any]:
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
        hierarchical_metrics = compute_hierarchical_metrics(
            y_true,
            y_pred,
            class_names=(
                [id2label[index] for index in all_labels]
                if include_class_wise_hierarchical
                else None
            ),
            include_class_wise=include_class_wise_hierarchical,
        )
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
    class_names: list[str] | None = None,
    include_class_wise: bool = False,
) -> dict[str, Any]:
    def safe_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(np.mean(values).item())

    def safe_f1(precision: float, recall: float) -> float:
        denominator = precision + recall
        if denominator == 0.0:
            return 0.0
        return (2 * precision * recall) / denominator

    def partial_match(y_t: str, y_p: str, d: float = 0.75) -> float:
        if y_t == y_p:
            return 1.0
        if y_t.split("-")[0] == y_p.split("-")[0]:
            return d / 2
        return 0.0

    metric_class_names = class_names if class_names is not None else sorted(set(y_true))
    class_hierarchical_precision = {
        class_name: safe_mean(
            [
                partial_match(target_class, predicted_class)
                for target_class, predicted_class in zip(y_true, y_pred, strict=True)
                if predicted_class == class_name
            ]
        )
        for class_name in metric_class_names
    }
    class_hierarchical_recall = {
        class_name: safe_mean(
            [
                partial_match(target_class, predicted_class)
                for target_class, predicted_class in zip(y_true, y_pred, strict=True)
                if target_class == class_name
            ]
        )
        for class_name in metric_class_names
    }
    class_hierarchical_f1 = {
        class_name: safe_f1(
            class_hierarchical_precision[class_name],
            class_hierarchical_recall[class_name],
        )
        for class_name in metric_class_names
    }
    metrics: dict[str, Any] = {
        "hierarchical_precision": safe_mean(list(class_hierarchical_precision.values())),
        "hierarchical_recall": safe_mean(list(class_hierarchical_recall.values())),
        "hierarchical_f1": safe_mean(list(class_hierarchical_f1.values())),
    }
    if include_class_wise:
        metrics["class_wise_hierarchical"] = {
            class_name: {
                "hierarchical_precision": class_hierarchical_precision[class_name],
                "hierarchical_recall": class_hierarchical_recall[class_name],
                "hierarchical_f1": class_hierarchical_f1[class_name],
            }
            for class_name in metric_class_names
        }
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_devices_argument(devices: str) -> Any:
    normalized = devices.strip().lower()
    if normalized == "auto":
        return "auto"
    if "," in normalized:
        return [int(part.strip()) for part in normalized.split(",") if part.strip()]
    return int(normalized)


def _get_lightning_runtime() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import WandbLogger
        return pl, ModelCheckpoint, EarlyStopping, LearningRateMonitor, WandbLogger
    except ImportError:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger
        return pl, ModelCheckpoint, EarlyStopping, LearningRateMonitor, WandbLogger


def _get_progress_bar_callback(pl: Any) -> Any:
    try:
        from lightning.pytorch.callbacks import TQDMProgressBar
        return TQDMProgressBar(refresh_rate=1)
    except ImportError:
        from pytorch_lightning.callbacks import TQDMProgressBar
        return TQDMProgressBar(refresh_rate=1)


def run_experiment(args: argparse.Namespace) -> Path:
    if args.include_bsd35k_cs and args.only_bsd35k_cs:
        raise ValueError(
            "--include-bsd35k-cs and --only-bsd35k-cs cannot both be enabled."
        )
    if args.pseudo_label_weight < 0:
        raise ValueError("--pseudo-label-weight must be non-negative.")

    seed = resolve_seed(args.seed)
    args.seed = seed

    pl, ModelCheckpoint, EarlyStopping, LearningRateMonitor, WandbLogger = _get_lightning_runtime()
    progress_bar = _get_progress_bar_callback(pl)

    dataset_roots = resolve_dataset_roots(args.bsd10k_root, args.bsd35k_root, args.bsd2k_root)
    train_records, val_records, test_records = get_experiment_records(
        bsd10k_root=dataset_roots["BSD10k"],
        bsd35k_root=dataset_roots["BSD35k-CS"],
        include_bsd35k_cs=args.include_bsd35k_cs,
        only_bsd35k_cs=args.only_bsd35k_cs,
        fold=args.fold,
        n_splits=args.n_splits,
        validation_size=args.validation_size,
    )
    label_specs = build_label_specs(train_records + val_records + test_records)
    label_map = build_label_map(label_specs)
    id2label = build_id2label(label_specs)
    pseudo_label_dir = resolve_pseudo_label_dir(args.pseudo_label_dir, args.output_root)
    pseudo_labels = (
        load_pseudo_labels(pseudo_label_dir, label_specs)
        if pseudo_label_dir is not None
        else {}
    )
    clean_train_size = sum(record["source_dataset"] == "BSD10k" for record in train_records)
    noisy_train_size = sum(record["source_dataset"] == "BSD35k-CS" for record in train_records)
    experiment_dir = create_experiment_dir(
        Path(args.output_root),
        args.include_bsd35k_cs,
        args.only_bsd35k_cs,
        args.embedding_model,
    )

    sample_rate = resolve_embedding_sample_rate(args.embedding_model)

    train_indices = maybe_limit(list(range(len(train_records))), args.max_train_items)
    val_indices = maybe_limit(list(range(len(val_records))), args.max_val_items)
    test_indices = maybe_limit(list(range(len(test_records))), args.max_test_items)

    train_dataset = WaveformClassificationDataset(
        train_records,
        train_indices,
        label_map,
        sample_rate,
        pseudo_labels=pseudo_labels,
    )
    val_dataset = WaveformClassificationDataset(val_records, val_indices, label_map, sample_rate)
    test_dataset = WaveformClassificationDataset(test_records, test_indices, label_map, sample_rate)
    class_frequency_loss_weights = None
    if args.use_class_frequency_loss:
        class_frequency_loss_weights = build_class_frequency_loss_weights(
            train_records,
            train_indices,
            label_map,
        )
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
                    "learning_rate": args.learning_rate,
                    "min_learning_rate": args.min_learning_rate,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "use_class_frequency_loss": args.use_class_frequency_loss,
                    "pseudo_label_dir": str(pseudo_label_dir) if pseudo_label_dir is not None else None,
                    "pseudo_label_weight": args.pseudo_label_weight,
                    "warmup_epochs": args.warmup_epochs,
                    "warmup_steps": warmup_steps,
                    "lr_decay_start_epoch": args.lr_decay_start_epoch,
                    "lr_decay_start_step": decay_start_step,
                    "update_steps_per_epoch": update_steps_per_epoch,
                    "total_update_steps": total_update_steps,
                    "num_labels": len(label_specs),
                    "freeze_encoder": args.freeze_encoder,
                    "use_llm_prior_embedding_fusion": args.use_llm_prior_embedding_fusion,
                }
            )

            self.embedding_model = build_embedding_model(
                args,
                sample_rate=sample_rate,
                id2label=id2label,
            )
            self.embedding_checkpoint_cfg = getattr(self.embedding_model, "checkpoint_cfg", None)

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
                classifier_input_dim = self.embedding_model.output_dim
                self.llm_class_embedding_bank = None
                self.fusion_head = None
            self.classifier = torch.nn.Linear(classifier_input_dim, len(label_specs))
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_frequency_loss_weights,
                label_smoothing=args.label_smoothing,
            )
            self.pseudo_label_weight = args.pseudo_label_weight
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
            logits = self(
                batch["waveforms"],
                batch["padding_mask"],
                metadata=batch["metadata"],
            )
            loss = self.loss_fn(logits, batch["labels"])
            pseudo_mask = batch["pseudo_label_mask"].to(logits.device)
            if self.pseudo_label_weight > 0 and pseudo_mask.any():
                pseudo_loss = soft_cross_entropy(
                    logits[pseudo_mask],
                    batch["pseudo_labels"].to(logits.device)[pseudo_mask],
                )
                loss = (1 - self.pseudo_label_weight) * loss + self.pseudo_label_weight * pseudo_loss
                self.log(
                    "train/pseudo_label_loss",
                    pseudo_loss,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch["labels"].size(0),
                )
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
                    "labels": batch["labels"].detach().cpu().numpy()
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
            logits = self(batch["waveforms"], batch["padding_mask"], metadata=batch["metadata"])
            loss = self.loss_fn(logits, batch["labels"])
            self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
            self.test_outputs.append(
                {
                    "logits": logits.detach().cpu().numpy(),
                    "labels": batch["labels"].detach().cpu().numpy()
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    initial_checkpoint_path = (
        resolve_initial_checkpoint_path(args.init_checkpoint_path, args.output_root)
        if args.init_checkpoint_path is not None
        else None
    )

    lightning_module = ClassificationLightningModule()
    if initial_checkpoint_path is not None:
        initial_state_dict = load_initial_training_state_dict(initial_checkpoint_path)
        lightning_module.load_state_dict(initial_state_dict)

    experiment_config = {
        "dataset": "BSD35k-CS" if args.only_bsd35k_cs else "BSD10k",
        "include_bsd35k_cs": args.include_bsd35k_cs,
        "only_bsd35k_cs": args.only_bsd35k_cs,
        "dataset_roots": {name: str(path) for name, path in dataset_roots.items()},
        "embedding_model": args.embedding_model,
        "init_checkpoint_path": (
            str(initial_checkpoint_path) if initial_checkpoint_path is not None else None
        ),
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
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
        "prediction_datasets": {
            "hidden_test_dataset": "BSD2k",
            "hidden_test_root": str(dataset_roots["BSD2k"]),
            "bsd10k_root": str(dataset_roots["BSD10k"]),
            "bsd35k_root": str(dataset_roots["BSD35k-CS"]),
        },
        "training": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "head_dropout": args.head_dropout,
            "label_smoothing": args.label_smoothing,
            "pseudo_label_dir": str(pseudo_label_dir) if pseudo_label_dir is not None else None,
            "pseudo_label_weight": args.pseudo_label_weight,
            "num_pseudo_labels": len(pseudo_labels),
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.early_stopping_patience,
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
            "save_checkpoints": args.save_checkpoints,
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
        "embedding_checkpoint_cfg": lightning_module.embedding_checkpoint_cfg,
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

    model_checkpoint = None
    if args.save_checkpoints:
        model_checkpoint = ModelCheckpoint(
            dirpath=str(experiment_dir / "checkpoints"),
            filename="epoch{epoch:02d}-step{step:06d}",
            monitor="val/hierarchical_f1",
            mode="max",
            save_top_k=1,
            save_last=False
        )
    early_stopping = EarlyStopping(
        monitor="val/hierarchical_f1",
        mode="max",
        patience=args.early_stopping_patience,
    )
    callbacks = [
        early_stopping,
        LearningRateMonitor(logging_interval="step"),
        progress_bar,
    ]
    if model_checkpoint is not None:
        callbacks.insert(0, model_checkpoint)
    trainer = pl.Trainer(
        default_root_dir=str(experiment_dir),
        accelerator=args.accelerator,
        devices=parse_devices_argument(args.devices),
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_checkpointing=args.save_checkpoints
    )

    datamodule = BSDDataModule()
    trainer.fit(lightning_module, datamodule=datamodule)
    best_model_path = model_checkpoint.best_model_path if model_checkpoint is not None else None
    if not best_model_path:
        best_model_path = None
    test_results = trainer.test(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=best_model_path,
    )

    best_checkpoint_path = Path(best_model_path) if best_model_path else None
    export_paths: dict[str, str] = {}
    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        best_checkpoint = torch.load(str(best_checkpoint_path), map_location="cpu", weights_only=False)
        best_model = ClassificationLightningModule()
        best_model.load_state_dict(best_checkpoint["state_dict"])
        inference_device = getattr(getattr(trainer, "strategy", None), "root_device", torch.device("cpu"))
        if not isinstance(inference_device, torch.device):
            inference_device = torch.device(str(inference_device))
        best_model.to(inference_device)

        prediction_specs = [
            ("bsd2k_hidden_test_logits.npz", load_full_dataset_records("BSD2k", dataset_roots["BSD2k"])),
            ("bsd10k_logits.npz", load_full_dataset_records("BSD10k", dataset_roots["BSD10k"])),
            ("bsd35k_cs_logits.npz", load_full_dataset_records("BSD35k-CS", dataset_roots["BSD35k-CS"])),
        ]
        for filename, records in prediction_specs:
            file_ids, logits = predict_logits_for_records(
                best_model,
                records,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sample_rate=sample_rate,
                device=inference_device,
            )
            npz_path = experiment_dir / filename
            write_logits_npz(npz_path, file_ids, logits, label_specs)
            export_paths[filename] = str(npz_path)

    summary = {
        "best_model_path": best_model_path,
        "best_model_score": (
            float(model_checkpoint.best_model_score.item())
            if model_checkpoint is not None and model_checkpoint.best_model_score is not None
            else None
        ),
        "test_results": test_results,
        "prediction_exports": export_paths,
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
