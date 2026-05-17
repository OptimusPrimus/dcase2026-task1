from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

DEFAULT_BSD_SPLIT_SEED = 566182


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class ExperimentSplit:
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    clean_train_size: int
    noisy_train_size: int
    split_seed: int = DEFAULT_BSD_SPLIT_SEED


def build_stratified_folds(
    labels: list[int],
    n_splits: int = 5,
    validation_size: float = 0.2,
    seed: int = 42,
) -> list[FoldSplit]:
    if len(labels) == 0:
        raise ValueError("Cannot build folds for an empty label list.")

    label_array = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldSplit] = []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(label_array)), label_array)):
        trainval_labels = label_array[trainval_idx]
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed+11,
        )
        train_idx_rel, val_idx_rel = next(
            sss.split(np.zeros(len(trainval_labels)), trainval_labels)
        )
        train_indices = trainval_idx[train_idx_rel].tolist()
        val_indices = trainval_idx[val_idx_rel].tolist()
        test_indices = test_idx.tolist()
        folds.append(
            FoldSplit(
                fold=fold,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
        )

    return folds


def load_records_by_dataset_name(dataset_name: str, root: Path) -> list[dict[str, Any]]:
    from dcase2026_task1.data.datasets import BSDDataset

    dataset = BSDDataset(
        root=root,
        dataset_name=dataset_name,
        load_audio=False,
    )
    records = list(dataset.records)
    if dataset_name == "BSD35k-CS":
        records = [record for record in records if not str(record["class"]).endswith("-other")]
    return records


def build_experiment_split(
    bsd10k_root: Path,
    bsd35k_root: Path | None,
    include_bsd35k_cs: bool,
    fold: int,
    n_splits: int,
    validation_size: float,
) -> ExperimentSplit:
    clean_records = load_records_by_dataset_name("BSD10k", bsd10k_root)
    noisy_records = (
        load_records_by_dataset_name("BSD35k-CS", bsd35k_root)
        if include_bsd35k_cs and bsd35k_root is not None
        else []
    )

    splits = build_stratified_folds(
        labels=[int(record["class_idx"]) for record in clean_records],
        n_splits=n_splits,
        validation_size=validation_size,
        seed=DEFAULT_BSD_SPLIT_SEED,
    )
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}], got {fold}.")
    fold_split = splits[fold]

    train_records = [clean_records[index] for index in fold_split.train_indices]
    train_records.extend(noisy_records)
    val_records = [clean_records[index] for index in fold_split.val_indices]
    test_records = [clean_records[index] for index in fold_split.test_indices]
    return ExperimentSplit(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        clean_train_size=len(fold_split.train_indices),
        noisy_train_size=len(noisy_records),
    )


def get_experiment_records(
    bsd10k_root: Path,
    bsd35k_root: Path | None,
    include_bsd35k_cs: bool,
    fold: int,
    n_splits: int,
    validation_size: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split = build_experiment_split(
        bsd10k_root=bsd10k_root,
        bsd35k_root=bsd35k_root,
        include_bsd35k_cs=include_bsd35k_cs,
        fold=fold,
        n_splits=n_splits,
        validation_size=validation_size,
    )
    return split.train_records, split.val_records, split.test_records
