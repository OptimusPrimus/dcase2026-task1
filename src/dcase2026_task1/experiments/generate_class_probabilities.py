from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import soundfile as sf
from tqdm import tqdm

from dcase2026_task1.data.datasets import (
    DEFAULT_BSD10K_ROOT,
    DEFAULT_BSD35K_ROOT,
    DEFAULT_BSD2K_ROOT,
    BSDDataset,
)

DEFAULT_MODEL_ID = "gpt-5.4-mini"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_OUTPUT_ROOT = "outputs/experiments"
BATCH_DIR_PREFIX = "batch_"
REQUESTS_FILENAME = "batch_requests.jsonl"
INPUT_ROWS_FILENAME = "input_rows.jsonl"
BATCH_STATE_FILENAME = "batch_state.json"
RAW_OUTPUT_FILENAME = "batch_output.jsonl"
RAW_ERROR_FILENAME = "batch_errors.jsonl"
PREDICTIONS_FILENAME = "predictions.jsonl"
BASE_PREDICTIONS_FILENAME = "base_predictions.jsonl"
CONFIG_FILENAME = "config.json"


def audio_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    # Reads only header metadata, not full audio data.
    info = sf.info(path)

    return {
        "file": str(path),
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration_sec": round(info.duration, 3),
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
    }


@dataclass(frozen=True)
class BatchState:
    batch_id: str
    input_file_id: str
    created_at: str
    endpoint: str = "/v1/responses"
    output_file_id: str | None = None
    error_file_id: str | None = None
    status: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate metadata summaries with OpenAI batch jobs and resumable outputs."
    )
    parser.add_argument(
        "action",
        choices=["prepare", "submit", "submit-batch", "status", "download", "complete"],
        nargs="?",
        default="submit",
    )
    parser.add_argument("--dataset", choices=["BSD10k", "BSD35k-CS", "BSD2k"], default="BSD10k")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--enable-reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_dataset_root(dataset_name: str, explicit_root: str | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    if dataset_name == "BSD10k":
        return DEFAULT_BSD10K_ROOT
    elif dataset_name == "BSD2k":
        return DEFAULT_BSD2K_ROOT
    return DEFAULT_BSD35K_ROOT


def build_prompt(item: dict[str, Any]) -> str:
    prompt = """Task: Classify the audio clip into a high-recall list of possible labels from the allowed classes below using metadata only. Assign each possible label a probability score.

Rules:

* Return ONLY a valid JSON array.
* Each array item must contain exactly two fields: `label` and `probability`.
* `label` must be one of the allowed class labels below or `other`.
* `probability` must be a number between 0 and 1.
* Return all reasonably plausible classes to maximize recall.
* Sort classes from highest to lowest probability.
* The probability scores must sum to 1.
* Do not return explanations.
* Choose probabilities based on the dominant/intended content, not incidental or background sounds.
* Always include `other` with a probability representing the remaining probability of all other classes.
* If metadata is missing or ambiguous, assign an appropriate probability to `other`.
* If metadata provides no meaningful evidence, return `[{"label":"other","probability":1.0}]`.
* Prefer the most specific valid classes.

CLASSES

# -------------------------------------------------------------------
# MUSIC (`m-*`)
# Intentional musical content including melodies, harmonies, rhythms,
# vocals, instrumental performances, loops, beats, musical phrases,
# musical textures, compositions, or musical productions.
# -------------------------------------------------------------------

1. `m-sp` — Music / Solo percussion
   Musical content containing only percussion or rhythmic percussion performance.
   Includes acoustic or electronic drums and unpitched percussion.
   Examples:
   drum solo, tabla rhythm, conga groove, cymbal performance, percussion loop

2. `m-si` — Music / Solo instrument
   Musical performance with exactly one non-percussive instrument OR solo singing.
   No accompaniment or layered instrumentation.
   Examples:
   solo piano melody, solo violin phrase, flute passage, solo vocal singing, guitar riff

3. `m-m` — Music / Multiple instruments
   Musical recordings containing more than one instrument or layered musical parts.
   Includes ensembles, bands, orchestras, accompaniment, backing tracks, and produced songs.
   Examples:
   orchestra, band performance, duet, cinematic score, EDM track, song with vocals and instruments

# -------------------------------------------------------------------
# INSTRUMENT SAMPLES (`is-*`)
# Isolated recordings intended as instrument samples, note references,
# articulations, scales, chromatic runs, or sound library material.
# Usually dry, short, and focused on demonstrating a single sound.
# -------------------------------------------------------------------

4. `is-p` — Instrument sample / Percussion
   Isolated percussion instrument samples or hits.
   Examples:
   kick sample, snare hit, cymbal strike, bell sample, xylophone note

5. `is-s` — Instrument sample / String
   Isolated samples from string instruments.
   Examples:
   violin sustain note, guitar pluck, harp glissando sample, cello articulation

6. `is-w` — Instrument sample / Wind
   Isolated samples from wind instruments.
   Examples:
   flute note, saxophone articulation, trumpet stab, clarinet sustain

7. `is-k` — Instrument sample / Keyboard instruments
   Isolated recordings of piano or acoustic/electromechanical keyboard instruments.
   Excludes synthesized sounds.
   Examples:
   piano note, organ chord, harpsichord sample, Rhodes key sample

8. `is-e` — Instrument sample / Synths and electronic
   Synthesized or electronically generated instrument samples.
   Includes analog or digital synthesizers and electronic tonal patches.
   Examples:
   synth stab, analog bass note, electronic lead sample, FM synth tone

# -------------------------------------------------------------------
# SPEECH (`sp-*`)
# Speech is dominant.
# Includes spoken communication, narration, dialogue, announcements,
# broadcast speech, and synthetic speech systems.
# -------------------------------------------------------------------

9. `sp-s` — Speech / Solo speech
   One person speaking clearly.
   Excludes singing and non-speech vocalizations.
   Examples:
   narration, monologue, podcast host, audiobook reading, lecture

10. `sp-c` — Speech / Conversation or crowd
    Multiple people speaking or conversational crowd speech.
    Includes overlapping dialogue and public conversational environments. Excludes non-speech sounds like applause, etc.
    Examples:
    interview, discussion, crowd chatter, meeting room, people talking in public

11. `sp-p` — Speech / Processed or synthetic
    Speech transmitted through devices or heavily processed/generated speech.
    Includes robotic, vocoded, radio, phone, AI-generated, or TTS voices.
    Examples:
    radio announcer, walkie-talkie speech, robotic assistant voice, synthetic narration

# -------------------------------------------------------------------
# SOUND EFFECTS (`fx-*`)
# Isolated discrete sound events or actions.
# Usually foreground sounds occurring one at a time rather than
# continuous environmental ambience.
# -------------------------------------------------------------------

12. `fx-o` — Sound effects / Objects and household appliances
    Sounds from small objects, tools, domestic items, or household appliances.
    Examples:
    door close, cup drop, zipper, keys jingling, microwave beep, scissors, typing

13. `fx-v` — Sound effects / Vehicles
    Sounds produced by transportation vehicles.
    Examples:
    car pass-by, motorcycle rev, airplane flyover, train brake, boat engine

14. `fx-m` — Sound effects / Machines and engines
    Mechanical or industrial machine sounds excluding vehicles and small home appliances.
    Examples:
    factory machine, drill, chainsaw, engine idle, hydraulic press, generator

15. `fx-h` — Sound effects / Human sounds and actions
    Human body sounds excluding speech and singing.
    Examples:
    footsteps, breathing, coughing, laughing, clapping, heartbeat, chewing, sneezing

16. `fx-a` — Sound effects / Animals
    Animal vocalizations or animal-generated sounds.
    Examples:
    dog bark, bird chirp, cat meow, insect buzzing, horse gallop, growling

17. `fx-n` — Sound effects / Natural elements and explosions
    Isolated natural events or elemental sounds.
    Examples:
    thunder clap, water splash, fire crackle, rock fall, explosion, gust of wind

18. `fx-ex` — Sound effects / Experimental
    Heavily processed, manipulated, abstract, or unconventional sound effects.
    Often artistic, distorted, reversed, granular, or noisy.
    Examples:
    reversed audio, glitch textures, spectral processing, extreme distortion effects

19. `fx-el` — Sound effects / Electronic or designed
    Artificially designed or synthesized non-musical sound effects.
    Includes UI sounds, sci-fi effects, cartoon sounds, and interface notifications.
    Examples:
    notification ping, laser blast, whoosh, arcade sound, UI click, futuristic effect

# -------------------------------------------------------------------
# SOUNDSCAPES (`ss-*`)
# Continuous ambient environments with multiple overlapping sound
# sources and environmental context. Might contain speech, music, or
# sound effects but the overall environment or atmosphere is the focus.
# Focus is on the environment as a whole rather than isolated events.
# -------------------------------------------------------------------

20. `ss-n` — Soundscape / Nature
    Ambient recordings from natural outdoor environments.
    Examples:
    forest ambience, jungle atmosphere, seaside waves, river ambience, rain in nature

21. `ss-i` — Soundscape / Indoors
    Ambient recordings from enclosed or indoor spaces.
    Examples:
    office room tone, restaurant ambience, shopping mall atmosphere, factory interior

22. `ss-u` — Soundscape / Urban
    Outdoor human-made environments and city ambiences.
    Examples:
    city street ambience, airport terminal, traffic ambience, subway station, marketplace

23. `ss-s` — Soundscape / Synthetic or artificial
    Artificially generated, designed, fictional, or synthesized environments.
    Examples:
    sci-fi ambience, fantasy environment, drone atmosphere, synthetic environmental beds

Output format:
Return a valid JSON array containing one or more predictions, ordered from highest to lowest probability. Always include `other` as one of the predictions.

Example:
[{"label":"fx-o","probability":0.65},{"label":"fx-m","probability":0.25},{"label":"other","probability":0.10}]"""

    return (
        prompt + "\n"
        "Clip metadata:\n"
        f'- title="{item.get("title", "")}"\n'
        f'- tags="{item.get("tags", "")}"\n'
        f'- description="{item.get("description", "")}"\n'
        f'- duration={audio_metadata(item["audio_path"])["duration_sec"]} sec\n'
    )


def create_experiment_dir(output_root: Path, dataset_name: str, model_id: str) -> Path:
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{model_id.replace('/', '_')}_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def resolve_experiment_dir(args: argparse.Namespace) -> Path:
    if args.experiment_dir is not None:
        return Path(args.experiment_dir)
    if args.action not in {"prepare", "submit"}:
        raise ValueError("--experiment-dir is required for submit-batch, status, download, and complete actions.")
    return create_experiment_dir(Path(args.output_root), args.dataset, args.model_id)


def ensure_openai_client(api_key: str | None, api_base: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("generate_metadata_summaries.py requires openai>=1.0.0.") from exc

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=api_base or os.environ.get("OPENAI_BASE_URL"),
    )
    return client


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def write_batch_state(path: Path, state: BatchState) -> None:
    write_json(path, state.__dict__)


def load_batch_state(path: Path) -> BatchState:
    return BatchState(**load_json(path))


def actual_num_batches(num_batches: int, num_rows: int) -> int:
    if num_batches < 1:
        raise ValueError("--num-batches must be at least 1.")
    if num_rows <= 0:
        return 1
    return min(num_batches, num_rows)


def batch_dir_name(index: int) -> str:
    return f"{BATCH_DIR_PREFIX}{index:04d}"


def resolve_batch_dirs(experiment_dir: Path) -> list[Path]:
    batch_dirs = sorted(
        path for path in experiment_dir.iterdir() if path.is_dir() and path.name.startswith(BATCH_DIR_PREFIX)
    )
    if batch_dirs:
        return batch_dirs
    return [experiment_dir]


def resolve_target_batch_dir(experiment_dir: Path, batch_index: int | None) -> Path:
    batch_dirs = resolve_batch_dirs(experiment_dir)
    if len(batch_dirs) == 1:
        if batch_index not in {None, 0}:
            raise ValueError(f"--batch-index {batch_index} is invalid: {experiment_dir} contains a single batch only.")
        return batch_dirs[0]

    if batch_index is None:
        raise ValueError("--batch-index is required when the experiment contains multiple batches.")

    target_name = batch_dir_name(batch_index)
    for batch_dir in batch_dirs:
        if batch_dir.name == target_name:
            return batch_dir
    raise FileNotFoundError(f"No batch directory found for --batch-index {batch_index} ({target_name}).")


def build_request_body(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model_id,
        "input": prompt,
        "max_output_tokens": args.max_new_tokens
    }
    if args.enable_reasoning:
        body["reasoning"] = {
            "effort": args.reasoning_effort,
            "summary": "auto",
        }
    return body


def build_request_record(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def split_rows_into_batches(rows: list[dict[str, Any]], num_batches: int) -> list[list[dict[str, Any]]]:
    if not rows:
        return [[]]

    actual_batches = actual_num_batches(num_batches, len(rows))
    base_size, remainder = divmod(len(rows), actual_batches)
    batches: list[list[dict[str, Any]]] = []
    start = 0
    for batch_index in range(actual_batches):
        batch_size = base_size + (1 if batch_index < remainder else 0)
        end = start + batch_size
        batches.append(rows[start:end])
        start = end
    return batches


def write_request_files(args: argparse.Namespace, target_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (target_dir / INPUT_ROWS_FILENAME).open("w", encoding="utf-8") as input_handle, (
        target_dir / REQUESTS_FILENAME
    ).open("w", encoding="utf-8") as request_handle:
        for row in rows:
            input_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            request_handle.write(
                json.dumps(
                    build_request_record(row["custom_id"], build_request_body(args, row["prompt"])),
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare_batch_directories(
    args: argparse.Namespace,
    experiment_dir: Path,
    rows: list[dict[str, Any]],
) -> list[Path]:
    row_batches = split_rows_into_batches(rows, args.num_batches)
    if len(row_batches) == 1:
        write_request_files(args, experiment_dir, row_batches[0])
        return [experiment_dir]

    batch_dirs: list[Path] = []
    for batch_index, batch_rows in enumerate(row_batches):
        batch_dir = experiment_dir / batch_dir_name(batch_index)
        batch_dir.mkdir(parents=True, exist_ok=False)
        write_request_files(args, batch_dir, batch_rows)
        batch_dirs.append(batch_dir)
    return batch_dirs


def prepare_input_rows(args: argparse.Namespace, experiment_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    dataset = BSDDataset(root=dataset_root, dataset_name=args.dataset, load_audio=False)
    limit = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)
    if args.dry_run and args.max_items is None:
        limit = min(limit, 5)

    rows: list[dict[str, Any]] = []
    for index in tqdm(range(limit), desc="Preparing requests", unit="item"):
        item = dataset[index]
        prompt = build_prompt(item)
        custom_id = f"dataset-index-{index}"
        row = {
            "custom_id": custom_id,
            "dataset_index": index,
            "sound_id": item["sound_id"],
            "source_dataset": item["source_dataset"],
            "audio_path": item["audio_path"],
            "title": item.get("title", ""),
            "tags": item.get("tags", ""),
            "description": item.get("description", ""),
            "target_class_idx": int(item.get("class_idx") or -1),
            "target_class": item.get("class"),
            "prompt": prompt,
        }
        rows.append(row)

    config = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "model": "openai",
        "model_id": args.model_id,
        "api_base": args.api_base,
        "max_new_tokens": args.max_new_tokens,
        "enable_reasoning": args.enable_reasoning,
        "reasoning_effort": args.reasoning_effort,
        "completion_window": args.completion_window,
        "dry_run": args.dry_run,
        "num_items": limit,
        "num_batches_requested": args.num_batches,
        "num_batches_actual": actual_num_batches(args.num_batches, len(rows)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(experiment_dir / CONFIG_FILENAME, config)
    write_jsonl_rows(experiment_dir / INPUT_ROWS_FILENAME, rows)

    return config, rows


def upload_batch_file(client: Any, request_path: Path) -> str:
    with request_path.open("rb") as handle:
        uploaded_file = client.files.create(file=handle, purpose="batch")
    return uploaded_file.id


def submit_batch_job(client: Any, args: argparse.Namespace, input_file_id: str, experiment_dir: Path) -> BatchState:
    metadata = {
        "experiment_dir": str(experiment_dir),
        "dataset": args.dataset,
        "model_id": args.model_id,
    }
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
        metadata=metadata,
    )
    return BatchState(
        batch_id=batch.id,
        input_file_id=input_file_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        output_file_id=getattr(batch, "output_file_id", None),
        error_file_id=getattr(batch, "error_file_id", None),
        status=getattr(batch, "status", None),
    )


def refresh_batch_state(client: Any, state: BatchState) -> tuple[BatchState, Any]:
    batch = client.batches.retrieve(state.batch_id)
    refreshed = BatchState(
        batch_id=state.batch_id,
        input_file_id=state.input_file_id,
        created_at=state.created_at,
        output_file_id=getattr(batch, "output_file_id", None),
        error_file_id=getattr(batch, "error_file_id", None),
        status=getattr(batch, "status", None),
    )
    return refreshed, batch


def read_text_response(file_response: Any) -> str:
    text = getattr(file_response, "text", None)
    if isinstance(text, str):
        return text

    content = getattr(file_response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content

    read = getattr(file_response, "read", None)
    if callable(read):
        payload = read()
        if isinstance(payload, bytes):
            return payload.decode("utf-8")
        if isinstance(payload, str):
            return payload

    raise TypeError("Unable to read file content from OpenAI SDK response.")



def download_file_text(client: Any, file_id: str) -> str:
    return read_text_response(client.files.content(file_id))


def load_input_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_custom_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_rows(path):
        rows_by_custom_id[row["custom_id"]] = row
    return rows_by_custom_id


def extract_output_text(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    texts: list[str] = []
    for item in response_body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []) or []:
            if content_item.get("type") not in {"output_text", "text"}:
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n\n".join(texts).strip()


def extract_reasoning_summary(response_body: dict[str, Any]) -> str | None:
    summaries: list[str] = []
    for item in response_body.get("output", []) or []:
        if item.get("type") != "reasoning":
            continue
        for part in item.get("summary", []) or []:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                summaries.append(text.strip())
    if not summaries:
        return None
    return "\n\n".join(summaries)


def normalize_raw_response(raw_response: Any) -> str | None:
    if not isinstance(raw_response, str):
        return None

    raw_response = raw_response.strip()
    if not raw_response:
        return None

    try:
        json.loads(raw_response)
        return raw_response
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    try:
        parsed, end_index = decoder.raw_decode(raw_response)
    except json.JSONDecodeError:
        return None

    if raw_response[end_index:].strip():
        return json.dumps(parsed, ensure_ascii=False)
    return raw_response


def build_prediction_rows(
    input_rows_by_custom_id: dict[str, dict[str, Any]],
    raw_output_text: str,
) -> list[dict[str, Any]]:
    output_rows_by_custom_id: dict[str, dict[str, Any]] = {}
    for line in raw_output_text.splitlines():
        if not line.strip():
            continue
        batch_row = json.loads(line)
        custom_id = batch_row.get("custom_id")
        if not isinstance(custom_id, str):
            continue
        output_rows_by_custom_id[custom_id] = batch_row

    rows: list[dict[str, Any]] = []
    ordered_input_rows = sorted(
        input_rows_by_custom_id.values(),
        key=lambda row: int(row["dataset_index"]),
    )
    for input_row in ordered_input_rows:
        batch_row = output_rows_by_custom_id.get(input_row["custom_id"], {})
        response = batch_row.get("response") or {}
        response_body = response.get("body") or {}
        normalized_raw_response = normalize_raw_response(extract_output_text(response_body))
        rows.append(
            {
                **input_row,
                "batch_request_id": batch_row.get("id"),
                "status_code": response.get("status_code"),
                "raw_response": normalized_raw_response,
                "reasoning": extract_reasoning_summary(response_body),
                "error": batch_row.get("error"),
            }
        )
    return rows


def prediction_failed(row: dict[str, Any]) -> bool:
    raw_response = normalize_raw_response(row.get("raw_response"))
    if row.get("error") is not None:
        return True
    if row.get("status_code") != 200:
        return True
    return raw_response is None


def merge_prediction_rows(
    base_rows: list[dict[str, Any]],
    retry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_custom_id: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        rows_by_custom_id[row["custom_id"]] = row
    for row in retry_rows:
        rows_by_custom_id[row["custom_id"]] = row
    return sorted(rows_by_custom_id.values(), key=lambda row: int(row["dataset_index"]))


def clone_args_for_completion(args: argparse.Namespace, source_config: dict[str, Any]) -> argparse.Namespace:
    cloned = vars(args).copy()
    cloned["dataset"] = source_config["dataset"]
    cloned["dataset_root"] = source_config["dataset_root"]
    cloned["model_id"] = source_config["model_id"]
    cloned["api_base"] = source_config.get("api_base")
    cloned["enable_reasoning"] = bool(source_config["enable_reasoning"])
    cloned["reasoning_effort"] = source_config["reasoning_effort"]
    cloned["completion_window"] = source_config["completion_window"]
    return argparse.Namespace(**cloned)


def run_submit(args: argparse.Namespace, experiment_dir: Path) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = experiment_dir / PREDICTIONS_FILENAME

    if (experiment_dir / BATCH_STATE_FILENAME).exists():
        raise FileExistsError(
            f"Batch state already exists at {experiment_dir / BATCH_STATE_FILENAME}. Use 'status' or 'download' with --experiment-dir instead of resubmitting."
        )
    if predictions_path.exists():
        raise FileExistsError(
            f"Predictions already exist at {predictions_path}. Refusing to overwrite an existing experiment directory."
        )

    _config, rows = prepare_input_rows(args, experiment_dir)
    batch_dirs = prepare_batch_directories(args, experiment_dir, rows)
    if args.dry_run:
        return experiment_dir

    client = ensure_openai_client(args.api_key, args.api_base)
    for batch_dir in batch_dirs:
        input_file_id = upload_batch_file(client, batch_dir / REQUESTS_FILENAME)
        batch_state = submit_batch_job(client, args, input_file_id, batch_dir)
        write_batch_state(batch_dir / BATCH_STATE_FILENAME, batch_state)
    return experiment_dir


def run_prepare(args: argparse.Namespace, experiment_dir: Path) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = experiment_dir / PREDICTIONS_FILENAME

    if (experiment_dir / BATCH_STATE_FILENAME).exists():
        raise FileExistsError(
            f"Batch state already exists at {experiment_dir / BATCH_STATE_FILENAME}. Refusing to prepare over an existing submitted experiment."
        )
    if predictions_path.exists():
        raise FileExistsError(
            f"Predictions already exist at {predictions_path}. Refusing to overwrite an existing experiment directory."
        )
    if any(path.name.startswith(BATCH_DIR_PREFIX) for path in experiment_dir.iterdir()):
        raise FileExistsError(
            f"Batch subdirectories already exist in {experiment_dir}. Refusing to overwrite prepared batch inputs."
        )
    if (experiment_dir / REQUESTS_FILENAME).exists() or (experiment_dir / INPUT_ROWS_FILENAME).exists():
        raise FileExistsError(
            f"Batch input files already exist in {experiment_dir}. Refusing to overwrite prepared batch inputs."
        )

    _config, rows = prepare_input_rows(args, experiment_dir)
    prepare_batch_directories(args, experiment_dir, rows)
    return experiment_dir


def run_submit_batch(args: argparse.Namespace, experiment_dir: Path) -> tuple[Path, Path, BatchState]:
    batch_dir = resolve_target_batch_dir(experiment_dir, args.batch_index)
    request_path = batch_dir / REQUESTS_FILENAME
    state_path = batch_dir / BATCH_STATE_FILENAME
    if not request_path.exists():
        raise FileNotFoundError(f"No batch requests found at {request_path}. Run 'prepare' first.")
    if state_path.exists():
        raise FileExistsError(
            f"Batch state already exists at {state_path}. This batch has already been submitted."
        )

    if args.dry_run:
        return experiment_dir, batch_dir, BatchState(
            batch_id="dry-run",
            input_file_id="dry-run",
            created_at=datetime.now().isoformat(timespec="seconds"),
            status="prepared",
        )

    client = ensure_openai_client(args.api_key, args.api_base)
    input_file_id = upload_batch_file(client, request_path)
    batch_state = submit_batch_job(client, args, input_file_id, batch_dir)
    write_batch_state(state_path, batch_state)
    return experiment_dir, batch_dir, batch_state


def summarize_batch_statuses(statuses: list[str | None]) -> str | None:
    if not statuses:
        return None
    if len(statuses) == 1:
        return statuses[0]

    counts: dict[str, int] = {}
    for status in statuses:
        label = status or "unknown"
        counts[label] = counts.get(label, 0) + 1
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    return f"{summary} (total={len(statuses)})"


def run_status(args: argparse.Namespace, experiment_dir: Path) -> tuple[Path, str | None]:
    batch_dirs = resolve_batch_dirs(experiment_dir)
    statuses: list[str | None] = []

    client = None if args.dry_run else ensure_openai_client(args.api_key, args.api_base)
    for batch_dir in batch_dirs:
        state_path = batch_dir / BATCH_STATE_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(f"No batch state found at {state_path}.")

        if args.dry_run:
            statuses.append(load_batch_state(state_path).status)
            continue

        refreshed_state, _batch = refresh_batch_state(client, load_batch_state(state_path))
        write_batch_state(state_path, refreshed_state)
        statuses.append(refreshed_state.status)

    return experiment_dir, summarize_batch_statuses(statuses)


def run_download(args: argparse.Namespace, experiment_dir: Path) -> tuple[Path, int, list[dict[str, Any]]]:
    batch_dirs = resolve_batch_dirs(experiment_dir)
    client = ensure_openai_client(args.api_key, args.api_base)
    new_rows: list[dict[str, Any]] = []
    for batch_dir in batch_dirs:
        state_path = batch_dir / BATCH_STATE_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(f"No batch state found at {state_path}.")

        state, batch = refresh_batch_state(client, load_batch_state(state_path))
        write_batch_state(state_path, state)

        status = getattr(batch, "status", None)
        if status != "completed":
            raise RuntimeError(
                f"Batch {state.batch_id} in {batch_dir} is not ready for download. Current status: {status!r}."
            )
        if not state.output_file_id:
            raise RuntimeError(f"Batch {state.batch_id} in {batch_dir} completed without an output_file_id.")

        raw_output_text = download_file_text(client, state.output_file_id)
        (batch_dir / RAW_OUTPUT_FILENAME).write_text(raw_output_text, encoding="utf-8")

        if state.error_file_id:
            (batch_dir / RAW_ERROR_FILENAME).write_text(
                download_file_text(client, state.error_file_id),
                encoding="utf-8",
            )

        input_rows = load_input_rows(batch_dir / INPUT_ROWS_FILENAME)
        batch_prediction_rows = build_prediction_rows(
            input_rows_by_custom_id=input_rows,
            raw_output_text=raw_output_text,
        )
        write_jsonl_rows(batch_dir / PREDICTIONS_FILENAME, batch_prediction_rows)
        new_rows.extend(batch_prediction_rows)

    new_rows = sorted(new_rows, key=lambda row: int(row["dataset_index"]))
    base_predictions_path = experiment_dir / BASE_PREDICTIONS_FILENAME
    merged_rows = new_rows
    if base_predictions_path.exists():
        merged_rows = merge_prediction_rows(load_jsonl_rows(base_predictions_path), new_rows)
    written = write_jsonl_rows(experiment_dir / PREDICTIONS_FILENAME, merged_rows)
    invalid_rows = [row for row in merged_rows if prediction_failed(row)]
    return experiment_dir, written, invalid_rows


def run_complete(args: argparse.Namespace, experiment_dir: Path) -> Path:
    source_predictions_path = experiment_dir / PREDICTIONS_FILENAME
    source_input_rows_path = experiment_dir / INPUT_ROWS_FILENAME
    source_config_path = experiment_dir / CONFIG_FILENAME
    if not source_predictions_path.exists():
        raise FileNotFoundError(f"No predictions found at {source_predictions_path}. Run 'download' first.")
    if not source_input_rows_path.exists():
        raise FileNotFoundError(f"No input rows found at {source_input_rows_path}.")
    if not source_config_path.exists():
        raise FileNotFoundError(f"No config found at {source_config_path}.")

    source_predictions = load_jsonl_rows(source_predictions_path)
    completed_rows = [row for row in source_predictions if not prediction_failed(row)]
    failed_rows = [row for row in source_predictions if prediction_failed(row)]
    if not failed_rows:
        raise RuntimeError(f"No failed predictions found in {source_predictions_path}.")

    source_config = load_json(source_config_path)
    completion_args = clone_args_for_completion(args, source_config)
    completion_dir = create_experiment_dir(
        Path(args.output_root),
        source_config["dataset"],
        source_config["model_id"],
    )

    source_input_rows = load_input_rows(source_input_rows_path)
    failed_input_rows = [source_input_rows[row["custom_id"]] for row in failed_rows]

    completion_config = dict(source_config)
    completion_config.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_items": len(failed_input_rows),
            "num_batches_requested": args.num_batches,
            "num_batches_actual": actual_num_batches(args.num_batches, len(failed_input_rows)),
            "completed_items_from_parent": len(completed_rows),
            "failed_items_from_parent": len(failed_input_rows),
            "parent_experiment_dir": str(experiment_dir),
        }
    )
    write_json(completion_dir / CONFIG_FILENAME, completion_config)
    write_jsonl_rows(completion_dir / BASE_PREDICTIONS_FILENAME, completed_rows)
    write_jsonl_rows(completion_dir / INPUT_ROWS_FILENAME, failed_input_rows)
    batch_dirs = prepare_batch_directories(completion_args, completion_dir, failed_input_rows)

    if args.dry_run:
        return completion_dir

    client = ensure_openai_client(args.api_key, args.api_base)
    for batch_dir in batch_dirs:
        input_file_id = upload_batch_file(client, batch_dir / REQUESTS_FILENAME)
        batch_state = submit_batch_job(client, completion_args, input_file_id, batch_dir)
        write_batch_state(batch_dir / BATCH_STATE_FILENAME, batch_state)
    return completion_dir


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = resolve_experiment_dir(args)

    if args.action == "prepare":
        output_dir = run_prepare(args, experiment_dir)
        print(f"Prepared batch inputs in {output_dir}")
        return

    if args.action == "submit":
        output_dir = run_submit(args, experiment_dir)
        if args.dry_run:
            print(f"Prepared batch inputs in {output_dir} (dry run, no OpenAI job submitted)")
            return

        batch_dirs = resolve_batch_dirs(output_dir)
        if len(batch_dirs) == 1:
            state = load_batch_state(output_dir / BATCH_STATE_FILENAME)
            print(f"Submitted OpenAI batch job {state.batch_id} in {output_dir}")
            return

        print(f"Submitted {len(batch_dirs)} OpenAI batch jobs in {output_dir}")
        for batch_dir in batch_dirs:
            state = load_batch_state(batch_dir / BATCH_STATE_FILENAME)
            print(f"{batch_dir.name}: {state.batch_id}")
        return

    if args.action == "submit-batch":
        output_dir, batch_dir, state = run_submit_batch(args, experiment_dir)
        if args.dry_run:
            print(f"Validated prepared batch inputs for {batch_dir} (dry run, no OpenAI job submitted)")
            return

        print(f"Submitted OpenAI batch job {state.batch_id} for {batch_dir} in {output_dir}")
        return

    if args.action == "status":
        output_dir, status = run_status(args, experiment_dir)
        print(f"Batch status for {output_dir}: {status}")
        return

    if args.action == "complete":
        output_dir = run_complete(args, experiment_dir)
        if args.dry_run:
            print(f"Prepared completion batch inputs in {output_dir} (dry run, no OpenAI job submitted)")
            return

        batch_dirs = resolve_batch_dirs(output_dir)
        if len(batch_dirs) == 1:
            state = load_batch_state(output_dir / BATCH_STATE_FILENAME)
            print(f"Submitted completion batch job {state.batch_id} in {output_dir}")
            return

        print(f"Submitted {len(batch_dirs)} completion batch jobs in {output_dir}")
        for batch_dir in batch_dirs:
            state = load_batch_state(batch_dir / BATCH_STATE_FILENAME)
            print(f"{batch_dir.name}: {state.batch_id}")
        return

    output_dir, num_rows, invalid_rows = run_download(args, experiment_dir)
    print(
        f"Downloaded batch results to {output_dir / PREDICTIONS_FILENAME} "
        f"({num_rows} rows, {len(invalid_rows)} invalid responses)"
    )
    for row in invalid_rows:
        print(
            json.dumps(
                {
                    "custom_id": row.get("custom_id"),
                    "dataset_index": row.get("dataset_index"),
                    "sound_id": row.get("sound_id"),
                    "status_code": row.get("status_code"),
                    "error": row.get("error"),
                    "raw_response": row.get("raw_response"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
