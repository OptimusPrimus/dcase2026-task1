from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from tqdm import tqdm

from dcase2026_task1.data.splits import DEFAULT_BSD_SPLIT_SEED
from dcase2026_task1.experiments.ensemble_predictions import (
    BSD10K_LOGITS_FILENAME,
    build_bsd10k_eval_records,
    softmax,
)
from dcase2026_task1.experiments.generate_class_probabilities import (
    BASE_PREDICTIONS_FILENAME,
    BATCH_DIR_PREFIX,
    BATCH_STATE_FILENAME,
    CONFIG_FILENAME,
    DEFAULT_COMPLETION_WINDOW,
    DEFAULT_MODEL_ID,
    INPUT_ROWS_FILENAME,
    PREDICTIONS_FILENAME,
    REQUESTS_FILENAME,
    actual_num_batches,
    ensure_openai_client,
    load_batch_state,
    prepare_batch_directories,
    resolve_batch_dirs,
    run_complete,
    run_download,
    run_status,
    run_submit_batch,
    submit_batch_job,
    upload_batch_file,
    write_batch_state,
    write_json,
    write_jsonl_rows,
)
from dcase2026_task1.experiments.training import (
    DEFAULT_BSD10K_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_TRAINING_OUTPUT_ROOT,
    resolve_record_file_id,
)

DEFAULT_OUTPUT_ROOT = "outputs/experiments"
DEFAULT_ENSEMBLE_DIR = (
    "ensemble_20260613_140743_BSD10k_beats_b958ff06__"
    "20260613_145533_BSD10k_m2d_fef98c63__"
    "20260613_160638_BSD10k_clap_6dd7f8b4__"
    "20260613_160721_BSD10k_clap_4d86ad60"
)


CLASS_SELECTION_RULES: dict[str, dict[str, Any]] = {
    "m-sp": {
        "name": "Music / Solo percussion",
        "definition": (
            "A musical excerpt or rhythmic performance using only percussion instruments. "
            "Includes drum passages, rhythmic percussion patterns, percussion loops, and solo "
            "unpitched percussion performances."
        ),
        "select_when": (
            "The clip contains intentional rhythmic percussion over time, such as a drum loop, "
            "drum passage, tabla/conga rhythm, cymbal pattern, or other solo percussion performance."
        ),
        "reject_when": (
            "The clip is a single isolated hit, one-shot sample, dry drum sample, bell hit, or "
            "xylophone note; use `is-p`. If non-percussive instruments or accompaniment are present, "
            "use `m-m`."
        ),
    },

    "m-si": {
        "name": "Music / Solo instrument",
        "definition": (
            "A musical excerpt performed by exactly one non-percussive instrument or one singing voice. "
            "The content is a musical phrase, melody, chord sequence, or solo performance rather than "
            "a dry sample."
        ),
        "select_when": (
            "The clip is mainly solo piano, guitar, violin, flute, saxophone, solo singing, or another "
            "single melodic instrument performing a phrase, melody, sequence of notes, or chords."
        ),
        "reject_when": (
            "There is more than one instrument, accompaniment, backing track, duet, layered production, "
            "or ensemble; use `m-m`. If it is only a single note, articulation, scale demo, or dry "
            "sample-library recording, use the appropriate `is-*` label."
        ),
    },

    "m-m": {
        "name": "Music / Multiple instruments",
        "definition": (
            "A musical excerpt with more than one instrument, voice layer, accompaniment, ensemble, "
            "produced arrangement, or multi-part musical texture."
        ),
        "select_when": (
            "The clip is a band, orchestra, duet, backing track, song arrangement, EDM/music production, "
            "score, multi-instrument loop, or any music where multiple musical sources are active."
        ),
        "reject_when": (
            "Only one non-percussive instrument or one solo voice is present; use `m-si`. Only solo "
            "percussion is present; use `m-sp`. A dry isolated instrument sample belongs under `is-*`."
        ),
    },

    "is-p": {
        "name": "Instrument sample / Percussion",
        "definition": (
            "An isolated percussion instrument sample, hit, note, articulation, or short demonstration. "
            "Covers idiophones and membranophones such as drums, snares, gongs, bells, and xylophones."
        ),
        "select_when": (
            "The clip is a short kick, snare, tom, cymbal, gong, bell, xylophone, mallet hit, or other "
            "single percussion sample or brief sample-library percussion articulation."
        ),
        "reject_when": (
            "The percussion forms a rhythmic passage, beat, loop, groove, or musical performance over time; "
            "use `m-sp`. If percussion is part of a larger arrangement, use `m-m`."
        ),
    },

    "is-s": {
        "name": "Instrument sample / String",
        "definition": (
            "An isolated sample, note, articulation, short scale, or dry demonstration from a string "
            "instrument."
        ),
        "select_when": (
            "The clip is a dry guitar, violin, viola, cello, double bass, harp, mandolin, plucked string, "
            "bowed string, or other string-instrument sample."
        ),
        "reject_when": (
            "The string instrument performs a complete musical phrase or solo excerpt; use `m-si`. "
            "If accompanied by other instruments, use `m-m`. If the sound is synthesized to imitate "
            "strings but clearly electronic/sample-synth based, consider `is-e`."
        ),
    },

    "is-w": {
        "name": "Instrument sample / Wind",
        "definition": (
            "An isolated sample, note, articulation, short scale, or dry demonstration from a wind "
            "instrument, including woodwinds and brass."
        ),
        "select_when": (
            "The clip is a flute, clarinet, saxophone, oboe, bassoon, trumpet, trombone, horn, tuba, "
            "or other aerophone sample."
        ),
        "reject_when": (
            "The wind instrument performs a longer musical phrase or solo excerpt; use `m-si`. "
            "If accompanied by other instruments, use `m-m`."
        ),
    },

    "is-k": {
        "name": "Instrument sample / Keyboard instruments",
        "definition": (
            "An isolated sample, note, chord, articulation, or short dry demonstration from a piano, "
            "organ, harpsichord, or other non-synthesized keyboard instrument."
        ),
        "select_when": (
            "The clip is a dry piano note or chord, organ sample, harpsichord sample, electric piano "
            "sample, or other keyboard-instrument sample."
        ),
        "reject_when": (
            "The keyboard plays a musical phrase, melody, progression, or performance; use `m-si`. "
            "If layered with other instruments, use `m-m`. If the sound is a synth patch rather than "
            "a piano/organ/keyboard instrument sample, use `is-e`."
        ),
    },

    "is-e": {
        "name": "Instrument sample / Synths and electronic",
        "definition": (
            "An isolated tonal musical instrument sample produced by synthesis or electronic means. "
            "Includes synth notes, bass patches, lead patches, pad notes, stabs, and electronic "
            "instrument articulations."
        ),
        "select_when": (
            "The clip is a synth bass note, synth lead note, pad chord, electronic stab, electronic "
            "instrument sample, or other short tonal sound intended to be used musically."
        ),
        "reject_when": (
            "The sound is a non-musical UI beep, notification, laser, whoosh, sci-fi effect, cartoon "
            "effect, or designed sound effect; use `fx-el`. If it is a full electronic music loop or "
            "arrangement, use `m-m`."
        ),
    },

    "sp-s": {
        "name": "Speech / Solo speech",
        "definition": (
            "A recording dominated by one natural, unprocessed speaker using spoken language. "
            "Excludes singing and non-speech vocal sounds."
        ),
        "select_when": (
            "The clip is narration, monologue, lecture, podcast speech, audiobook reading, script "
            "reading, or another clear single-speaker recording."
        ),
        "reject_when": (
            "Multiple speakers, dialogue, crowd chatter, or overlapping voices are present; use `sp-c`. "
            "Phone/radio/robotic/TTS/vocoded speech should use `sp-p`. Singing belongs under music, "
            "usually `m-si` or `m-m`."
        ),
    },

    "sp-c": {
        "name": "Speech / Conversation or crowd",
        "definition": (
            "Speech involving multiple people, dialogue, conversation, public chatter, or overlapping "
            "human voices."
        ),
        "select_when": (
            "The clip is an interview, meeting, discussion, dialogue, playground chatter, people talking "
            "in public, conversational crowd speech, or overlapping spoken voices."
        ),
        "reject_when": (
            "Only one clear speaker is present; use `sp-s`. The crowd is mostly non-speech ambience "
            "without intelligible or dominant talking; use `ss-i` for indoor ambience or `ss-u` for "
            "urban/outdoor ambience. Processed, phone, radio, or synthetic speech should use `sp-p`."
        ),
    },

    "sp-p": {
        "name": "Speech / Processed or synthetic",
        "definition": (
            "Speech that is transmitted through a device, heavily processed, robotic, vocoded, degraded, "
            "or synthesized."
        ),
        "select_when": (
            "The clip contains phone speech, radio speech, walkie-talkie speech, intercom speech, TTS, "
            "robotic voice, AI/synthetic speech, vocoded speech, or strongly filtered/distorted speech."
        ),
        "reject_when": (
            "The voice is natural and unprocessed; use `sp-s` for one speaker or `sp-c` for multiple "
            "speakers. Non-speech human sounds such as coughing, crying, breathing, or laughter belong "
            "under `fx-h` unless spoken language dominates."
        ),
    },

    "fx-o": {
        "name": "Sound effects / Objects and household appliances",
        "definition": (
            "An isolated foreground sound event from everyday objects, small tools, clothing, weapons, "
            "domestic items, or household appliances."
        ),
        "select_when": (
            "The clip is keys, typing, door movement, dishes, zipper, switch, button, small tool, clothes, "
            "iron, microwave beep, domestic appliance action, object impact, or similar foreground object sound."
        ),
        "reject_when": (
            "The source is a transportation vehicle; use `fx-v`. It is an industrial or larger machine "
            "such as a drill, chainsaw, gear, or lawn mower; use `fx-m`. It is continuous room ambience "
            "rather than a single object event; use `ss-i`."
        ),
    },

    "fx-v": {
        "name": "Sound effects / Vehicles",
        "definition": (
            "An isolated foreground sound produced by a transportation vehicle or vehicle component."
        ),
        "select_when": (
            "The clip is a car passing by, car braking or screeching, wipers, motorcycle, bike, airplane, "
            "train, boat, ship, siren vehicle, vehicle engine, or other clear vehicle sound."
        ),
        "reject_when": (
            "Vehicles are only part of a continuous traffic bed or city ambience; use `ss-u`. "
            "A non-transport machine or stationary engine-like mechanism should use `fx-m`."
        ),
    },

    "fx-m": {
        "name": "Sound effects / Machines and engines",
        "definition": (
            "An isolated foreground sound from a mechanical, industrial, engine-like, or motorized source, "
            "excluding transportation vehicles and small domestic appliances."
        ),
        "select_when": (
            "The clip is a drill, lawn mower, gear mechanism, electric chainsaw, generator, compressor, "
            "factory machine, hydraulic mechanism, or similar non-vehicle machine sound."
        ),
        "reject_when": (
            "The machine is clearly part of a vehicle; use `fx-v`. It is a small household appliance or "
            "everyday object; use `fx-o`. It is a continuous industrial interior ambience; use `ss-i`."
        ),
    },

    "fx-h": {
        "name": "Sound effects / Human sounds and actions",
        "definition": (
            "Isolated non-speech sounds produced by the human body or human physical actions."
        ),
        "select_when": (
            "The clip is breathing, heartbeat, sneezing, coughing, crying, laughing, clapping, footsteps, "
            "walking, chewing, hand movement, or another non-speech human action."
        ),
        "reject_when": (
            "Spoken language dominates; use `sp-s`, `sp-c`, or `sp-p`. Singing belongs under music. "
            "A crowd ambience with human presence but no foreground body action should usually be `ss-i` "
            "or `ss-u`."
        ),
    },

    "fx-a": {
        "name": "Sound effects / Animals",
        "definition": (
            "An isolated or foreground animal vocalization or animal-generated sound."
        ),
        "select_when": (
            "The clip is a cat meow or purr, dog bark, sheep sound, bird call, insect buzz, horse sound, "
            "animal walking, growl, or another clear animal cue."
        ),
        "reject_when": (
            "Animals are only incidental within a broader natural environment; use `ss-n`. "
            "A continuous habitat recording dominated by ambience rather than a single animal cue should "
            "also use `ss-n`."
        ),
    },

    "fx-n": {
        "name": "Sound effects / Natural elements and explosions",
        "definition": (
            "An isolated foreground sound event caused by natural elements, physical natural processes, "
            "or explosions."
        ),
        "select_when": (
            "The clip is a single wind gust, fire flame burst, ice crack, rock fall, stones, water splash, "
            "thunder crack, impact from natural material, or explosion."
        ),
        "reject_when": (
            "The clip is continuous nature ambience such as rain bed, river ambience, forest ambience, "
            "seaside ambience, or weather atmosphere; use `ss-n`. Isolated animal sounds should use `fx-a`."
        ),
    },

    "fx-ex": {
        "name": "Sound effects / Experimental",
        "definition": (
            "A foreground sound effect that is abstract, heavily manipulated, unusually recorded, reversed, "
            "distorted, granular, noisy, or otherwise experimental."
        ),
        "select_when": (
            "The clip is a reversed sound, weird effect, abstract processed texture, glitch, unusual "
            "recording technique, extreme manipulation, or nonstandard designed effect that is not easily "
            "recognized as a normal object, machine, animal, natural event, UI sound, or musical sample."
        ),
        "reject_when": (
            "The sound is a recognizable UI, notification, laser, whoosh, cartoon, sci-fi, or animation "
            "effect; use `fx-el`. If it is tonal and intended as a musical synth sample, use `is-e`. "
            "If it is a continuous artificial ambience, use `ss-s`."
        ),
    },

    "fx-el": {
        "name": "Sound effects / Electronic or designed",
        "definition": (
            "A foreground computer-made, synthesized, or designed non-musical sound effect, often used "
            "for interfaces, animation, games, sci-fi, or transitions."
        ),
        "select_when": (
            "The clip is a UI click, alert, notification, beep, laser, whoosh, boink, cartoon effect, "
            "arcade sound, game effect, futuristic effect, or other discrete electronic/designed SFX."
        ),
        "reject_when": (
            "The sound is a tonal synth instrument sample meant for music; use `is-e`. "
            "A full electronic music loop or arrangement should use `m-m`. A continuous synthetic "
            "environment or drone bed should use `ss-s`. Highly abstract processing without a recognizable "
            "designed-effect function may be `fx-ex`."
        ),
    },

    "ss-n": {
        "name": "Soundscape / Nature",
        "definition": (
            "A continuous ambient field recording or environmental bed from a natural habitat or outdoor "
            "natural setting."
        ),
        "select_when": (
            "The clip is forest ambience, jungle ambience, seaside, river with surrounding nature, rain bed, "
            "wind/weather ambience, farmland ambience, or a natural outdoor environment with multiple "
            "ongoing background events."
        ),
        "reject_when": (
            "The clip is a single isolated animal sound; use `fx-a`. A single splash, gust, crack, fire, "
            "rock fall, thunder hit, or explosion should use `fx-n`. Urban outdoor ambience belongs under `ss-u`."
        ),
    },

    "ss-i": {
        "name": "Soundscape / Indoors",
        "definition": (
            "A continuous ambient recording from an enclosed or indoor space, with room tone or multiple "
            "background events."
        ),
        "select_when": (
            "The clip is closed room ambience, room tone, office ambience, restaurant/bar ambience, mall "
            "interior, indoor crowd bed, factory interior ambience, or another indoor environmental recording."
        ),
        "reject_when": (
            "The clip is a single foreground object, appliance, machine, or human action; use the appropriate "
            "`fx-*` label. Outdoor city ambience belongs under `ss-u`. Multiple people speaking clearly may "
            "belong under `sp-c` if speech dominates."
        ),
    },

    "ss-u": {
        "name": "Soundscape / Urban",
        "definition": (
            "A continuous outdoor soundscape from a city, transport hub exterior, road, market, or other "
            "human-made outdoor environment."
        ),
        "select_when": (
            "The clip is city ambience, busy road ambience, traffic bed, station exterior, market ambience, "
            "outside airport ambience, street atmosphere, or outdoor public space with human-made background sound."
        ),
        "reject_when": (
            "The clip is one foreground vehicle pass-by, horn, brake, or engine event; use `fx-v`. "
            "Indoor public ambience belongs under `ss-i`. Natural outdoor habitats belong under `ss-n`."
        ),
    },

    "ss-s": {
        "name": "Soundscape / Synthetic or artificial",
        "definition": (
            "A continuous synthesized, computer-made, fictional, or artificially designed ambient environment."
        ),
        "select_when": (
            "The clip is sci-fi ambience, fantasy environment, imaginary-place ambience, synthetic drone "
            "atmosphere, artificial environmental bed, or computer-made soundscape."
        ),
        "reject_when": (
            "The clip is a discrete UI sound, laser, whoosh, alert, cartoon sound, or other foreground "
            "designed effect; use `fx-el`. If it is a tonal synth instrument sample, use `is-e`. "
            "If it is a full electronic music excerpt, use `m-m`."
        ),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OpenAI batch prompts that ask an LLM to select likely BSD10k classes "
            "from metadata and an audio classifier's top predictions."
        )
    )
    parser.add_argument(
        "action",
        choices=["prepare", "submit", "submit-batch", "status", "download", "complete"],
        nargs="?",
        default="submit",
    )
    parser.add_argument("--dataset", choices=["BSD10k"], default="BSD10k")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--dataset-root", "--bsd10k-root", dest="dataset_root", default=None)
    parser.add_argument("--prediction-dir", default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--prediction-filename", default=BSD10K_LOGITS_FILENAME)
    parser.add_argument("--prediction-output-root", default=str(DEFAULT_TRAINING_OUTPUT_ROOT))
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
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_BSD_SPLIT_SEED)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--candidate-top-k", type=int, default=5)
    parser.add_argument("--max-candidate-classes", type=int, default=10)
    return parser


def create_experiment_dir(output_root: Path, dataset_name: str, split_name: str, model_id: str) -> Path:
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{split_name}_"
        f"audio_classifier_classes_{model_id.replace('/', '_')}_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def resolve_experiment_dir(args: argparse.Namespace) -> Path:
    if args.experiment_dir is not None:
        return Path(args.experiment_dir)
    if args.action not in {"prepare", "submit"}:
        raise ValueError("--experiment-dir is required for submit-batch, status, download, and complete actions.")
    return create_experiment_dir(Path(args.output_root), args.dataset, args.split, args.model_id)


def resolve_prediction_dir(prediction_dir: str, prediction_output_root: str | Path) -> Path:
    path = Path(prediction_dir).expanduser()
    if path.exists():
        return path

    candidate = Path(prediction_output_root).expanduser() / prediction_dir
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve prediction directory {prediction_dir!r} directly or under {prediction_output_root}."
    )


def read_split_config(prediction_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = prediction_dir / "ensemble_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        split = payload.get("split")
        if isinstance(split, dict):
            return {
                "fold": int(split.get("fold", args.fold)),
                "n_splits": int(split.get("n_splits", args.n_splits)),
                "validation_size": float(split.get("validation_size", args.validation_size)),
                "split_seed": int(split.get("split_seed", args.split_seed)),
            }

    return {
        "fold": args.fold,
        "n_splits": args.n_splits,
        "validation_size": args.validation_size,
        "split_seed": args.split_seed,
    }


def load_logits_npz(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "label_names" not in data:
            raise ValueError(f"Prediction file {path} does not contain label_names.")
        label_names = [str(label) for label in data["label_names"].tolist()]
        logits_by_file_id = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in data.files
            if key != "label_names"
        }
    return label_names, logits_by_file_id


def select_candidate_predictions(
    logits: np.ndarray,
    label_names: list[str],
    *,
    min_probability: float,
    top_k: int,
    max_classes: int,
) -> list[dict[str, float | str]]:
    if top_k < 1:
        raise ValueError("--candidate-top-k must be at least 1.")
    if max_classes < 1:
        raise ValueError("--max-candidate-classes must be at least 1.")

    probabilities = softmax(logits)
    ranked_indices = np.argsort(probabilities)[::-1]
    selected_indices: list[int] = []
    for rank, label_index in enumerate(ranked_indices):
        probability = float(probabilities[label_index])
        if rank < top_k or probability >= min_probability:
            selected_indices.append(int(label_index))
        if len(selected_indices) >= max_classes:
            break

    return [
        {"label": label_names[index], "probability": round(float(probabilities[index]), 6)}
        for index in selected_indices
    ]


def format_candidate_rules(candidate_labels: list[str]) -> str:
    sections: list[str] = []
    for label in candidate_labels:
        rule = CLASS_SELECTION_RULES.get(label)
        if rule is None:
            sections.append(
                f"- `{label}`: No detailed rule is available. Use only if the metadata and audio score strongly support it."
            )
            continue
        sections.append(
            "\n".join(
                [
                    f"- `{label}` - {rule['name']}",
                    f"  Definition: {rule['definition']}",
                    f"  Select when: {rule['select_when']}",
                    f"  Reject when: {rule['reject_when']}",
                ]
            )
        )
    return "\n".join(sections)


def build_prompt(item: dict[str, Any], candidate_predictions: list[dict[str, float | str]]) -> str:
    candidate_labels = [str(prediction["label"]) for prediction in candidate_predictions]
    predictions_json = json.dumps(candidate_predictions, ensure_ascii=False, indent=2)
    labels_json = json.dumps(candidate_labels, ensure_ascii=False)

    return f"""Your task is to correct the predictions of an audio classifier.
* Look at the audio classifier's predictions and the audio clip's metadata below.
* Use the class definitions and rules below to correct the predictions.
* Return only a JSON dictionary with one entry for every plausible candidate label.
* The dictionary keys are the class labels, the entries your confidence.
* The label must be one of the candidate labels.
* The confidence must be a number from 0 to 1 reflecting your confidence that the label applies.
* Include every candidate that is likely enough to be a correct label, not just the top one.
* Exclude candidates that are merely plausible but contradicted by the metadata.
* Sort from most likely to least likely.
* Do not invent labels, do not include `other`, and do not write explanations.

Candidate class definitions and selection rules:
{format_candidate_rules(candidate_labels)}

Clip metadata:
{json.dumps( { "title": item.get("title") or "", "tags": item.get("tags") or "", "description": item.get("description") or "", }, ensure_ascii=False, indent=2, )}

Audio classifier predictions:
{predictions_json}
"""


def prepare_input_rows(args: argparse.Namespace, experiment_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_dir = resolve_prediction_dir(args.prediction_dir, args.prediction_output_root)
    prediction_path = prediction_dir / args.prediction_filename
    label_names, logits_by_file_id = load_logits_npz(prediction_path)
    split_config = read_split_config(prediction_dir, args)
    if int(split_config.get("split_seed", DEFAULT_BSD_SPLIT_SEED)) != DEFAULT_BSD_SPLIT_SEED:
        raise ValueError(
            f"Unsupported split_seed={split_config['split_seed']}. "
            f"Expected {DEFAULT_BSD_SPLIT_SEED}."
        )

    dataset_root = Path(args.dataset_root or DEFAULT_BSD10K_ROOT).expanduser()
    val_records, test_records = build_bsd10k_eval_records(dataset_root, split_config)
    records = val_records if args.split == "validation" else test_records
    limit = len(records) if args.max_items is None else min(len(records), args.max_items)

    rows: list[dict[str, Any]] = []
    for split_index, item in enumerate(tqdm(records[:limit], desc="Preparing requests", unit="item")):
        file_id = resolve_record_file_id(item)
        if file_id not in logits_by_file_id:
            raise KeyError(f"Prediction file {prediction_path} is missing logits for file_id={file_id!r}.")

        candidate_predictions = select_candidate_predictions(
            logits_by_file_id[file_id],
            label_names,
            min_probability=args.candidate_threshold,
            top_k=args.candidate_top_k,
            max_classes=args.max_candidate_classes,
        )
        prompt = build_prompt(item, candidate_predictions)
        dataset_index = int(item["dataset_index"])
        row = {
            "custom_id": f"{args.split}-dataset-index-{dataset_index}",
            "dataset_index": dataset_index,
            "split": args.split,
            "split_index": split_index,
            "file_id": file_id,
            "sound_id": item.get("sound_id"),
            "source_dataset": item["source_dataset"],
            "audio_path": item["audio_path"],
            "title": item.get("title", ""),
            "tags": item.get("tags", ""),
            "description": item.get("description", ""),
            "target_class_idx": int(item.get("class_idx") or -1),
            "target_class": item.get("class"),
            "audio_classifier_predictions": candidate_predictions,
            "prompt": prompt,
        }
        rows.append(row)

    config = {
        "dataset": args.dataset,
        "split": args.split,
        "dataset_root": str(dataset_root),
        "prediction_dir": str(prediction_dir),
        "prediction_filename": args.prediction_filename,
        "prediction_path": str(prediction_path),
        "candidate_threshold": args.candidate_threshold,
        "candidate_top_k": args.candidate_top_k,
        "max_candidate_classes": args.max_candidate_classes,
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
        "split_config": split_config,
        "label_names": label_names,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(experiment_dir / CONFIG_FILENAME, config)
    write_jsonl_rows(experiment_dir / INPUT_ROWS_FILENAME, rows)
    return config, rows


def run_prepare_local(args: argparse.Namespace, experiment_dir: Path) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if (experiment_dir / BATCH_STATE_FILENAME).exists():
        raise FileExistsError(f"Batch state already exists at {experiment_dir / BATCH_STATE_FILENAME}.")
    if (experiment_dir / PREDICTIONS_FILENAME).exists():
        raise FileExistsError(f"Predictions already exist at {experiment_dir / PREDICTIONS_FILENAME}.")
    if any(path.name.startswith(BATCH_DIR_PREFIX) for path in experiment_dir.iterdir()):
        raise FileExistsError(f"Batch subdirectories already exist in {experiment_dir}.")
    if (experiment_dir / REQUESTS_FILENAME).exists() or (experiment_dir / INPUT_ROWS_FILENAME).exists():
        raise FileExistsError(f"Batch input files already exist in {experiment_dir}.")

    _config, rows = prepare_input_rows(args, experiment_dir)
    prepare_batch_directories(args, experiment_dir, rows)
    return experiment_dir


def run_submit_local(args: argparse.Namespace, experiment_dir: Path) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if (experiment_dir / BATCH_STATE_FILENAME).exists():
        raise FileExistsError(f"Batch state already exists at {experiment_dir / BATCH_STATE_FILENAME}.")
    if (experiment_dir / PREDICTIONS_FILENAME).exists():
        raise FileExistsError(f"Predictions already exist at {experiment_dir / PREDICTIONS_FILENAME}.")

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


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = resolve_experiment_dir(args)

    if args.action == "prepare":
        output_dir = run_prepare_local(args, experiment_dir)
        print(f"Prepared batch inputs in {output_dir}")
        return

    if args.action == "submit":
        output_dir = run_submit_local(args, experiment_dir)
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

    if args.action == "download":
        output_dir, written, invalid_rows = run_download(args, experiment_dir)
        print(f"Wrote {written} prediction rows to {output_dir / PREDICTIONS_FILENAME}")
        if invalid_rows:
            print(f"{len(invalid_rows)} rows need retry. Run 'complete' with --experiment-dir {output_dir}.")
        return

    if args.action == "complete":
        completion_dir = run_complete(args, experiment_dir)
        if args.dry_run:
            print(f"Prepared retry batch inputs in {completion_dir} (dry run, no OpenAI job submitted)")
            return
        print(f"Submitted retry batch job(s) in {completion_dir}")
        print(f"Parent predictions are preserved in {completion_dir / BASE_PREDICTIONS_FILENAME}")
        return

    raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
