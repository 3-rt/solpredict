from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from solpredict.featurize import smiles_to_descriptors, smiles_to_fingerprint

DescriptorValue = float | int


@dataclass(slots=True)
class FeaturizedDataset:
    fingerprints: NDArray[np.int8]
    descriptors: list[dict[str, DescriptorValue]]
    valid_mask: NDArray[np.bool_]
    cache_path: Path
    cache_hit: bool


def load_esol(path: str | Path) -> pd.DataFrame:
    """Load the ESOL dataset and normalize the expected column names."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        lowered = column.lower()
        if lowered == "smiles":
            rename_map[column] = "smiles"
        elif "measured log solubility" in lowered:
            rename_map[column] = "log_solubility"
        elif lowered in {"compound id", "compound_id", "name"}:
            rename_map[column] = "name"

    frame = frame.rename(columns=rename_map)
    required = {"smiles", "log_solubility"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"ESOL CSV missing required columns: {missing}")

    columns = ["smiles", "log_solubility"]
    if "name" in frame.columns:
        columns.append("name")
    return frame.loc[:, columns].copy()


def build_feature_cache_key(
    frame: pd.DataFrame,
    *,
    fp_radius: int,
    fp_nbits: int,
) -> str:
    """Derive a stable cache key from the dataset contents and fingerprint params."""
    payload = frame.loc[:, ["smiles", "log_solubility"]].to_csv(index=False).encode("utf-8")
    suffix = f"|{fp_radius}|{fp_nbits}".encode()
    return sha256(payload + suffix).hexdigest()[:16]


def featurize_dataset(
    frame: pd.DataFrame,
    *,
    cache_dir: str | Path,
    fp_radius: int,
    fp_nbits: int,
) -> FeaturizedDataset:
    """Featurize the dataset, persisting reusable results under the cache dir."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = build_feature_cache_key(frame, fp_radius=fp_radius, fp_nbits=fp_nbits)
    cache_path = cache_root / f"fingerprints-{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as cached:
            cached_descriptors = cast(
                list[dict[str, DescriptorValue]],
                list(cached["descriptors"].tolist()),
            )
            return FeaturizedDataset(
                fingerprints=cast(NDArray[np.int8], cached["fingerprints"]),
                descriptors=cached_descriptors,
                valid_mask=cast(NDArray[np.bool_], cached["valid_mask"]),
                cache_path=cache_path,
                cache_hit=True,
            )

    fingerprints: list[NDArray[np.int8]] = []
    descriptors: list[dict[str, DescriptorValue]] = []
    valid_mask: list[bool] = []

    for smiles in frame["smiles"]:
        fingerprint = smiles_to_fingerprint(smiles, radius=fp_radius, n_bits=fp_nbits)
        descriptor_map = smiles_to_descriptors(smiles)
        if fingerprint is None or descriptor_map is None:
            valid_mask.append(False)
            continue
        valid_mask.append(True)
        fingerprints.append(fingerprint)
        descriptors.append(cast(dict[str, DescriptorValue], descriptor_map))

    fingerprint_array: NDArray[np.int8]
    if fingerprints:
        fingerprint_array = np.asarray(fingerprints, dtype=np.int8)
    else:
        fingerprint_array = np.empty((0, fp_nbits), dtype=np.int8)

    result = FeaturizedDataset(
        fingerprints=fingerprint_array,
        descriptors=descriptors,
        valid_mask=np.asarray(valid_mask, dtype=bool),
        cache_path=cache_path,
        cache_hit=False,
    )
    np.savez_compressed(
        cache_path,
        fingerprints=result.fingerprints,
        descriptors=np.asarray(result.descriptors, dtype=object),
        valid_mask=result.valid_mask,
    )
    return result


def split_holdout(
    fingerprints: NDArray[np.int8],
    targets: NDArray[np.float64],
    *,
    test_size: float,
    random_seed: int,
) -> tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.float64], NDArray[np.float64]]:
    """Perform the deterministic train/test split used by the pipeline."""
    x_train, x_test, y_train, y_test = train_test_split(
        fingerprints,
        targets,
        test_size=test_size,
        random_state=random_seed,
    )
    return x_train, x_test, y_train, y_test
