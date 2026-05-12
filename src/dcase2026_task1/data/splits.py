from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


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
            random_state=seed,
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
