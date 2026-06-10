# DCASE 2026 Task 1

## Setup

This repository requires Python 3.10 or newer.

1. Create the conda environment:

```bash
conda create -n dcase2026-task1 python=3.10
```

2. Activate it:

```bash
conda activate dcase2026-task1
```

3. Install the package in editable mode:

```bash
python -m pip install -e '.[dev]'
```

4. Place the datasets in the default locations, or pass custom paths to the training script:

- `~/data/BSD10k`
- `~/data/BSD35k-CS`

Each dataset root is expected to contain:

- `audio/`
- `metadata/`

The training script uses the official `beats_iter3plus_as2m` checkpoint from `~/checkpoints` by default.

## Example: Train With BEATs

```bash
python -m dcase2026_task1.experiments.training \
  --wandb-project=dcase2026-task1 \
  --wandb-mode=online \
  --learning_rate=3e-05 \
  --lr_decay_start_epoch=1 \
  --max_epochs=10 \
  --min_learning_rate=1e-06 \
  --warmup_epochs=1 \
  --weight_decay=0.01
```

To enable the metadata decoder:

```bash
python -m dcase2026_task1.experiments.training \
  --decoder-model=bart \
  --decoder-pretrained-model-name=facebook/bart-base
```

## Example: Start a Sweep

```bash
wandb sweep sweeps/training_learning_rate.yaml
```

Start a W&B agent for the created sweep:

```bash
wandb agent dcase2026-task1/<SWEEP_ID>
```
