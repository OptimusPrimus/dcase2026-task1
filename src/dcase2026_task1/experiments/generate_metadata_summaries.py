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
    DEFAULT_PROCESSED_METADATA_ROOT,
    BSDDataset,
)

DEFAULT_MODEL_ID = "gpt-5.4-mini"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_OUTPUT_ROOT = DEFAULT_PROCESSED_METADATA_ROOT
REQUESTS_FILENAME = "batch_requests.jsonl"
INPUT_ROWS_FILENAME = "input_rows.jsonl"
BATCH_STATE_FILENAME = "batch_state.json"
RAW_OUTPUT_FILENAME = "batch_output.jsonl"
RAW_ERROR_FILENAME = "batch_errors.jsonl"
PREDICTIONS_FILENAME = "predictions.jsonl"
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
    parser.add_argument("action", choices=["submit", "status", "download"], nargs="?", default="submit")
    parser.add_argument("--dataset", choices=["BSD10k", "BSD2k", "BSD35k-CS"], default="BSD10k")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--enable-reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--max-items", type=int, default=None)
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
    prompt = """Briefly summarize the audio content indicated by the provided title, tags, and description.

Include only details useful for understanding what is likely heard. Describe the content affirmatively: state what the audio is or contains, without explaining what it is not or contrasting it with alternatives. Do not add unsupported details. Ignore where in the metadata the information appears, as well as promotional, licensing, attribution, administrative, and technical information.

Return only a concise summary of one or two sentences."""

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
    if args.action != "submit":
        raise ValueError("--experiment-dir is required for status and download actions.")
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


def write_batch_state(path: Path, state: BatchState) -> None:
    write_json(path, state.__dict__)


def load_batch_state(path: Path) -> BatchState:
    return BatchState(**load_json(path))


def build_request_body(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model_id,
        "input": prompt,
        "max_output_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
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


def prepare_input_rows(args: argparse.Namespace, experiment_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    dataset = BSDDataset(root=dataset_root, dataset_name=args.dataset, load_audio=False)
    limit = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)
    if args.dry_run and args.max_items is None:
        limit = min(limit, 5)

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
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(experiment_dir / CONFIG_FILENAME, config)

    rows: list[dict[str, Any]] = []
    with (experiment_dir / INPUT_ROWS_FILENAME).open("w", encoding="utf-8") as input_handle, (
        experiment_dir / REQUESTS_FILENAME
    ).open("w", encoding="utf-8") as request_handle:
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
                "target_class": item["class"],
                "prompt": prompt,
            }
            rows.append(row)
            input_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            request_handle.write(
                json.dumps(build_request_record(custom_id, build_request_body(args, prompt)), ensure_ascii=False) + "\n"
            )

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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
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


def materialize_predictions(
    input_rows_by_custom_id: dict[str, dict[str, Any]],
    raw_output_text: str,
    predictions_path: Path,
) -> int:
    output_rows_by_custom_id: dict[str, dict[str, Any]] = {}
    for line in raw_output_text.splitlines():
        if not line.strip():
            continue
        batch_row = json.loads(line)
        custom_id = batch_row.get("custom_id")
        if not isinstance(custom_id, str):
            continue
        output_rows_by_custom_id[custom_id] = batch_row

    written = 0
    ordered_input_rows = sorted(
        input_rows_by_custom_id.values(),
        key=lambda row: int(row["dataset_index"]),
    )
    with predictions_path.open("w", encoding="utf-8") as handle:
        for input_row in ordered_input_rows:
            batch_row = output_rows_by_custom_id.get(input_row["custom_id"], {})
            response = batch_row.get("response") or {}
            response_body = response.get("body") or {}
            row = {
                **input_row,
                "batch_request_id": batch_row.get("id"),
                "status_code": response.get("status_code"),
                "raw_response": extract_output_text(response_body) or None,
                "reasoning": extract_reasoning_summary(response_body),
                "error": batch_row.get("error"),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def run_submit(args: argparse.Namespace, experiment_dir: Path) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    requests_path = experiment_dir / REQUESTS_FILENAME
    state_path = experiment_dir / BATCH_STATE_FILENAME
    predictions_path = experiment_dir / PREDICTIONS_FILENAME

    if state_path.exists():
        raise FileExistsError(
            f"Batch state already exists at {state_path}. Use 'status' or 'download' with --experiment-dir instead of resubmitting."
        )
    if predictions_path.exists():
        raise FileExistsError(
            f"Predictions already exist at {predictions_path}. Refusing to overwrite an existing experiment directory."
        )

    prepare_input_rows(args, experiment_dir)
    if args.dry_run:
        return experiment_dir

    client = ensure_openai_client(args.api_key, args.api_base)
    input_file_id = upload_batch_file(client, requests_path)
    batch_state = submit_batch_job(client, args, input_file_id, experiment_dir)
    write_batch_state(state_path, batch_state)
    return experiment_dir


def run_status(args: argparse.Namespace, experiment_dir: Path) -> tuple[Path, str | None]:
    state_path = experiment_dir / BATCH_STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"No batch state found at {state_path}.")

    if args.dry_run:
        state = load_batch_state(state_path)
        return experiment_dir, state.status

    client = ensure_openai_client(args.api_key, args.api_base)
    refreshed_state, _batch = refresh_batch_state(client, load_batch_state(state_path))
    write_batch_state(state_path, refreshed_state)
    return experiment_dir, refreshed_state.status


def run_download(args: argparse.Namespace, experiment_dir: Path) -> tuple[Path, int]:
    state_path = experiment_dir / BATCH_STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"No batch state found at {state_path}.")

    client = ensure_openai_client(args.api_key, args.api_base)
    state, batch = refresh_batch_state(client, load_batch_state(state_path))
    write_batch_state(state_path, state)

    status = getattr(batch, "status", None)
    if status != "completed":
        raise RuntimeError(
            f"Batch {state.batch_id} is not ready for download. Current status: {status!r}."
        )
    if not state.output_file_id:
        raise RuntimeError(f"Batch {state.batch_id} completed without an output_file_id.")

    raw_output_text = download_file_text(client, state.output_file_id)
    raw_output_path = experiment_dir / RAW_OUTPUT_FILENAME
    raw_output_path.write_text(raw_output_text, encoding="utf-8")

    if state.error_file_id:
        raw_error_path = experiment_dir / RAW_ERROR_FILENAME
        raw_error_path.write_text(download_file_text(client, state.error_file_id), encoding="utf-8")

    input_rows = load_input_rows(experiment_dir / INPUT_ROWS_FILENAME)
    written = materialize_predictions(
        input_rows_by_custom_id=input_rows,
        raw_output_text=raw_output_text,
        predictions_path=experiment_dir / PREDICTIONS_FILENAME,
    )
    return experiment_dir, written


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = resolve_experiment_dir(args)

    if args.action == "submit":
        output_dir = run_submit(args, experiment_dir)
        if args.dry_run:
            print(f"Prepared batch inputs in {output_dir} (dry run, no OpenAI job submitted)")
            return

        state = load_batch_state(output_dir / BATCH_STATE_FILENAME)
        print(f"Submitted OpenAI batch job {state.batch_id} in {output_dir}")
        return

    if args.action == "status":
        output_dir, status = run_status(args, experiment_dir)
        print(f"Batch status for {output_dir}: {status}")
        return

    output_dir, num_rows = run_download(args, experiment_dir)
    print(f"Downloaded batch results to {output_dir / PREDICTIONS_FILENAME} ({num_rows} rows)")


if __name__ == "__main__":
    main()
