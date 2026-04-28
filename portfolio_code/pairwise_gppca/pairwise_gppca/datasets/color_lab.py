from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..types import SubjectPreferenceData, empirical_pairwise_scores


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / ".." / "color_preference_lab" / "data"


def _default_names_dir() -> Path:
    return Path(__file__).resolve().parents[2] / ".." / "color_preference_lab" / "postprocess"


def load_color_lab_pairwise_data(
    data_dir: str | Path | None = None,
    names_dir: str | Path | None = None,
    session: int = 1,
    name_list_file: str | Path | None = None,
) -> list[SubjectPreferenceData]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("`torch` is required to load the color_lab_pairwise dataset.") from exc

    data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    names_dir = Path(names_dir) if names_dir is not None else _default_names_dir()
    name_list_path = Path(name_list_file) if name_list_file is not None else names_dir / f"name_list{session}.csv"

    if not name_list_path.exists():
        raise FileNotFoundError(
            f"Missing participant list: {name_list_path}. "
            "The adapter is available, but the raw color preference data is not bundled in this portfolio."
        )
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Missing raw color preference data directory: {data_dir}. "
            "Place `*_result.pt` and `*_response.pt` files there."
        )

    names = pd.read_csv(name_list_path, header=None).iloc[:, 0].astype(str).tolist()
    subjects: list[SubjectPreferenceData] = []

    for subject_id, name in enumerate(names, start=1):
        result_path = data_dir / f"{name}_result.pt"
        response_path = data_dir / f"{name}_response.pt"
        if not result_path.exists() or not response_path.exists():
            raise FileNotFoundError(f"Missing raw color preference files for participant `{name}`.")

        x = torch.load(result_path).detach().cpu().numpy().astype(np.float64)
        response = torch.load(response_path).detach().cpu().numpy().astype(np.int64)
        winners = response[:, 0]
        losers = response[:, 1]
        initial_targets = empirical_pairwise_scores(winners, losers, x.shape[0])

        subjects.append(
            SubjectPreferenceData(
                subject_id=subject_id,
                x=x,
                winner_indices=winners,
                loser_indices=losers,
                initial_targets=initial_targets,
                label=name,
                dataset_name="color_lab_pairwise",
            )
        )

    return subjects
