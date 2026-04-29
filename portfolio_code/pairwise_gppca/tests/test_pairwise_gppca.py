from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pairwise_gppca import KernelParameters, PairwiseGPPCA, load_dataset


class PairwiseGPPCTest(unittest.TestCase):
    def test_golden_ratio_loader_creates_induced_pairwise_data(self) -> None:
        subjects = load_dataset(
            "golden_ratio_induced",
            data_dir=Path(__file__).resolve().parents[1] / "data" / "golden_ratio",
            num_subjects=3,
            min_score_gap=1.0,
            max_comparisons=80,
            random_state=1,
        )
        self.assertEqual(len(subjects), 3)
        self.assertTrue(all(subject.comparison_count > 0 for subject in subjects))
        self.assertTrue(all(subject.x.shape[1] == 1 for subject in subjects))

    def test_thurstone_loader_reads_real_pairwise_data(self) -> None:
        subjects = load_dataset(
            "thurstone_pairwise",
            data_dir=Path(__file__).resolve().parents[1] / "data" / "thurstone" / "data10",
            session_count=1,
            subject_count=2,
        )
        self.assertEqual(len(subjects), 2)
        self.assertEqual(subjects[0].x.shape[1], 6)
        self.assertTrue(subjects[0].comparison_count > 0)

    def test_model_smoke_fit_and_predict_on_golden_ratio(self) -> None:
        subjects = load_dataset(
            "golden_ratio_induced",
            data_dir=Path(__file__).resolve().parents[1] / "data" / "golden_ratio",
            num_subjects=3,
            min_score_gap=1.0,
            max_comparisons=60,
            random_state=2,
        )
        model = PairwiseGPPCA(
            subjects=subjects,
            params=KernelParameters(
                length=0.25,
                preference_noise=1.0,
                newton_max_iter=6,
                newton_tolerance=1e-5,
            ),
            latent_dim=2,
            basis_size=10,
            jitter=1e-5,
            random_state=0,
        ).fit(epochs=3, step_z=0.01, step_w1=0.05, step_w2=0.05)

        grid = np.linspace(0.2, 2.8, 25)[:, None]
        mean, cov = model.predict_single(grid)
        reconstructed_mean, reconstructed_cov = model.predict_reconstructed(grid)

        self.assertEqual(mean.shape, (3, 25))
        self.assertEqual(cov.shape, (3, 25, 25))
        self.assertEqual(reconstructed_mean.shape, (3, 25))
        self.assertEqual(reconstructed_cov.shape, (3, 25, 25))
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(reconstructed_mean).all())


if __name__ == "__main__":
    unittest.main()
