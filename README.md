# DCASE 2026 Task 1

Minimal project structure for working with the `BSD35k-CS` and `BSD10k` datasets.

## Layout

- `src/dcase2026_task1/data/datasets.py`: PyTorch datasets
- `src/dcase2026_task1/tasks/`: model-agnostic task definitions
- `src/dcase2026_task1/models/<model_name>/base.py`: model runtime implementation
- `src/dcase2026_task1/models/<model_name>/<skill>.py`: task skill implementation for a model
- `src/dcase2026_task1/cli.py`: argument parser and quick inspection entrypoint

Current tasks:

- `classification`
- `audio_captioning`
- `metadata_summarization`

## Quick start

```bash
python -m dcase2026_task1.cli --limit 3
```

```bash
python -m dcase2026_task1.train --max-test-items 10
```

```bash
python -m dcase2026_task1.caption_dataset --max-items 10
```

```bash
python -m dcase2026_task1.metadata_summary_dataset --max-items 10
```

Default dataset roots:

- `~/data/BSD35k-CS`
- `~/data/BSD10k`

Each root is expected to contain:

- `audio/`
- `metadata/`
  - `BSD35k-CS_metadata.csv` or `BSD10k_metadata.csv`
  - `BST_description.csv` or `BTS_description.csv`

## Evaluation

`dcase2026_task1.train` builds a combined dataset, creates five stratified folds, takes one fold as test, and splits the remaining four folds into train and validation with a 20% stratified validation split.

The current setup skips training and evaluates a pretrained audio-language model on the test split. The first backend is `Audio Flamingo 3` through Hugging Face Transformers.

Install the optional runtime before using the model adapter:

```bash
python -m pip install -e '.[dev]'
```
