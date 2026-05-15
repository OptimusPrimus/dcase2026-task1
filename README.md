# DCASE 2026 Task 1

Minimal project structure for working with the `BSD35k-CS` and `BSD10k` datasets.

## Layout

- `src/dcase2026_task1/data/datasets.py`: PyTorch datasets
- `src/dcase2026_task1/models/<model_name>/base.py`: model runtime implementation
- `src/dcase2026_task1/experiments/text_metadata_classification.py`: metadata-only text classification experiment
- `src/dcase2026_task1/experiments/audio_tagging.py`: audio-input tagging experiment with per-class probabilities

## Quick start

Run the metadata-only experiment against `BSD10k` with the Qwen backend:

```bash
python -m dcase2026_task1.experiments.text_metadata_classification --dataset BSD10k --model qwen
```

Dry-run the same experiment without submitting model requests:

```bash
python -m dcase2026_task1.experiments.text_metadata_classification --dataset BSD10k --model qwen --dry-run --max-items 5
```

The legacy `train` module now forwards to the same experiment:

```bash
python -m dcase2026_task1.train --dataset BSD10k --model qwen
```

Run an AudioSet audio-tagging model on raw audio and write probabilities for every tag:

```bash
python -m dcase2026_task1.experiments.audio_tagging --dataset BSD10k --max-items 5
```

Use a different Hugging Face checkpoint if you want a BEATs-based or other AudioSet-compatible tagger:

```bash
python -m dcase2026_task1.experiments.audio_tagging --dataset BSD10k --model-id <huggingface-model-id>
```

Default dataset roots:

- `~/data/BSD35k-CS`
- `~/data/BSD10k`

Each root is expected to contain:

- `audio/`
- `metadata/`
  - `BSD35k-CS_metadata.csv` or `BSD10k_metadata.csv`
  - `BST_description.csv` or `BTS_description.csv`

Install the runtime dependencies before running experiments:

```bash
python -m pip install -e '.[dev]'
```
