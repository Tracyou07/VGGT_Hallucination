from __future__ import annotations

import unittest
import hashlib

import torch

from pre_experiments.conditional_hierarchical_vrfm.basis import (
    canonical_basis_sha256,
    expand_residual,
    split_hierarchical_coefficients,
    temporal_dct_basis,
)


class TemporalBasisTests(unittest.TestCase):
    def test_basis_is_deterministic_and_orthonormal(self) -> None:
        first = temporal_dct_basis()
        second = temporal_dct_basis()
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(
            first.T @ first, torch.eye(32), atol=1e-5, rtol=1e-5
        )

    def test_expand_and_hierarchical_split_have_exact_shapes(self) -> None:
        coefficients = torch.zeros(4, 32, 2048)
        residual = expand_residual(coefficients, temporal_dct_basis())
        self.assertEqual(residual.shape, (4, 500, 2048))
        global_part, local_part = split_hierarchical_coefficients(coefficients)
        self.assertEqual(global_part.shape, (4, 4, 2048))
        self.assertEqual(local_part.shape, (4, 28, 2048))

    def test_split_allows_the_full_fixed_rank_as_global(self) -> None:
        coefficients = torch.zeros(1, 32, 2048)
        global_part, local_part = split_hierarchical_coefficients(
            coefficients, global_rank=32
        )
        self.assertEqual(global_part.shape, (1, 32, 2048))
        self.assertEqual(local_part.shape, (1, 0, 2048))

    def test_canonical_digest_binds_exact_cpu_float32_bytes(self) -> None:
        basis = temporal_dct_basis()
        expected = hashlib.sha256(basis.numpy().tobytes(order="C")).hexdigest()
        self.assertEqual(canonical_basis_sha256(), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cpu_and_cuda_use_identical_canonical_basis_bytes(self) -> None:
        cpu = temporal_dct_basis()
        cuda_roundtrip = temporal_dct_basis(device=torch.device("cuda")).cpu()
        self.assertTrue(torch.equal(cpu, cuda_roundtrip))


if __name__ == "__main__":
    unittest.main()
