from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4
import soundfile as sf
from tqdm import tqdm

def audio_metadata(path):
    path = Path(path)

    # Reads only header metadata, not full audio data
    info = sf.info(path)

    return {
        "file": str(path),
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration_sec": round(info.duration, 3),
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,   # often contains bit depth info
    }

from dcase2026_task1.data.datasets import (
    DEFAULT_BSD10K_ROOT,
    DEFAULT_BSD35K_ROOT,
    BSDDataset,
)
from dcase2026_task1.models import GenerativeModel, ModelInput, ModelOutput, OpenAIModel, QwenModel


@dataclass(frozen=True)
class CandidateClass:
    class_idx: int
    class_name: str
    description_top_level: str
    description_second_level: str
    description: str
    examples: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a metadata-only text classification experiment over one BSD dataset."
    )
    parser.add_argument("--dataset", choices=["BSD10k", "BSD35k-CS"], default="BSD10k")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--model", choices=["qwen", "openai"], default="openai")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Reserved dtype setting for the Qwen backend.",
    )
    parser.add_argument("--device", default="auto", help="Reserved device setting for the Qwen backend.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--enable-reasoning", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default="outputs/experiments")
    return parser


def resolve_dataset_root(dataset_name: str, explicit_root: str | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    if dataset_name == "BSD10k":
        return DEFAULT_BSD10K_ROOT
    return DEFAULT_BSD35K_ROOT


def build_candidate_classes(dataset: BSDDataset) -> list[CandidateClass]:
    by_class_idx: dict[int, CandidateClass] = {}
    for record in dataset.records:
        class_idx = int(record["class_idx"])
        if class_idx in by_class_idx:
            continue
        by_class_idx[class_idx] = CandidateClass(
            class_idx=class_idx,
            class_name=str(record["class"]),
            description_top_level=str(record["description_top_level"]),
            description_second_level=str(record["description_second_level"]),
            description=str(record["description_text"]),
            examples=str(record["description_examples"])
        )
    return [by_class_idx[idx] for idx in sorted(by_class_idx)]


def build_prompt(item: dict[str, Any], candidate_classes: list[CandidateClass]) -> str:
    class_lines = [
        (
            f"{index}. {candidate.class_name} "
            f"({candidate.description_top_level} -> {candidate.description_second_level}): "
            f"{candidate.description}"
            f"Examples: {candidate.examples}"
        )
        for index, candidate in enumerate(candidate_classes, start=1)
    ]

    prompt = """
You are an audio metadata classifier.

Task:
Classify the audio clip into exactly ONE label from the allowed classes below using metadata only.

Rules:
- Return ONLY the class label (example: `fx-o`) or `unknown`.
- Do not return explanations.
- Choose the dominant/intended content, not incidental or background sounds.
- If metadata is missing or ambiguous return `unknown`.
- Prefer the most specific valid class.

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
Return exactly one label or `unknown`.
"""

    return (
        prompt + "\n"
        "Clip metadata:\n"
        f'- title="{item.get("title", "")}"\n'
        f'- tags="{item.get("tags", "")}"\n'
        f'- description="{item.get("description", "")}"\n'
        f'- duration={audio_metadata(item["audio_path"])["duration_sec"]} sec\n'
    )


def parse_prediction(
    raw_response: str,
    candidate_classes: list[CandidateClass],
) -> tuple[int | None, str | None]:
    import re

    option_match = re.search(r"\b(\d+)\b", raw_response)
    if option_match is None:
        return None, raw_response.strip() or None
    option_index = int(option_match.group(1))
    if not 1 <= option_index <= len(candidate_classes):
        return None, str(option_index)
    return candidate_classes[option_index - 1].class_idx, str(option_index)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("batch_size must be >= 1.")
    iterator = iter(items)
    batches: list[list[Any]] = []
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return batches
        batches.append(batch)


def create_experiment_dir(output_root: Path, dataset_name: str, model_name: str) -> Path:
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{model_name}_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def load_text_model(args: argparse.Namespace) -> GenerativeModel:
    if args.model == "openai":
        return OpenAIModel(
            model_id=args.model_id or "gpt-5.4-mini",
            api_key=args.api_key,
            base_url=args.api_base,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            enable_reasoning=args.enable_reasoning,
        )
    raise ValueError(f"Unsupported model backend: {args.model}")


def resolve_model_id(args: argparse.Namespace) -> str | None:
    if args.model_id is not None:
        return args.model_id
    if args.model == "qwen":
        return "Qwen/Qwen3.6-27B"
    if args.model == "openai":
        return "gpt-5.4-mini"
    return None


def run_experiment(args: argparse.Namespace) -> Path:
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    dataset = BSDDataset(root=dataset_root, dataset_name=args.dataset, load_audio=False)
    candidate_classes = build_candidate_classes(dataset)
    model = None if args.dry_run else load_text_model(args)
    output_root = Path(args.output_root)
    experiment_dir = create_experiment_dir(output_root, args.dataset, args.model)

    limit = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)
    if args.dry_run and args.max_items is None:
        limit = min(limit, 5)

    config = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "model": args.model,
        "model_id": resolve_model_id(args),
        "api_base": args.api_base,
        "max_new_tokens": args.max_new_tokens,
        "enable_reasoning": args.enable_reasoning,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
        "num_items": limit,
        "candidate_classes": [asdict(candidate) for candidate in candidate_classes],
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    predictions_path = experiment_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for batch_indices in tqdm(chunked(list(range(limit)), args.batch_size), desc="Batches", unit="batch"):
            batch_items = [dataset[index] for index in batch_indices]
            batch_inputs = [ModelInput(prompt=build_prompt(item, candidate_classes)) for item in batch_items]
            batch_outputs = (
                [ModelOutput(text="") for _ in batch_inputs]
                if args.dry_run
                else model.generate_batch_outputs(batch_inputs)
            )

            for index, item, model_input, model_output in zip(
                batch_indices,
                batch_items,
                batch_inputs,
                batch_outputs,
                strict=True,
            ):
                raw_response = model_output.text
                predicted_class_idx, parsed_label = parse_prediction(raw_response, candidate_classes)
                row = {
                    "dataset_index": index,
                    "sound_id": item["sound_id"],
                    "source_dataset": item["source_dataset"],
                    "audio_path": item["audio_path"],
                    "title": item["title"],
                    "tags": item["tags"],
                    "description": item["description"],
                    "target_class_idx": int(item["class_idx"]),
                    "target_class": item["class"],
                    "prompt": model_input.prompt,
                    "raw_response": raw_response or None,
                    "reasoning": model_output.reasoning_summary,
                    "parsed_label": parsed_label,
                    "predicted_class_idx": predicted_class_idx,
                    "dry_run": args.dry_run,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    return experiment_dir


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = run_experiment(args)
    print(f"Wrote experiment outputs to {experiment_dir}")


if __name__ == "__main__":
    main()
