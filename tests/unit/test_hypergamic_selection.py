"""Unit tests for Phase 4, Step 3: Hypergamic Assortative Selection & Residual Orthogonality Gating."""

import pytest

from quant.analytics.hypergamic_selection import (
    HypergamicConfig,
    HypergamicSelectionError,
    InvalidCohortException,
    MatingDeadlockException,
    MatingPair,
    OffspringResult,
)


class TestDomainEntitiesAndInvariants:
    """Test domain dataclasses, invariants, and validation rules."""

    def test_hypergamic_config_defaults_and_validation(self) -> None:
        """Nominal defaults and boundary invariant validation in HypergamicConfig."""
        cfg = HypergamicConfig()
        assert cfg.alpha_ratio == 0.25
        assert cfg.orthogonality_threshold == 0.30
        assert cfg.max_mating_attempts == 5
        assert cfg.relaxation_factor == 0.80
        assert cfg.tournament_size == 3
        assert cfg.crossover_distribution_index == 15.0
        assert cfg.alpha_risk_inheritance_prob == 0.75
        assert cfg.aspirant_repr_inheritance_prob == 0.65
        assert cfg.elitism_count == 2

        # Reject out-of-bounds alpha_ratio
        with pytest.raises(HypergamicSelectionError, match="alpha_ratio must be in"):
            HypergamicConfig(alpha_ratio=0.0)
        with pytest.raises(HypergamicSelectionError, match="alpha_ratio must be in"):
            HypergamicConfig(alpha_ratio=1.0)

        # Reject out-of-bounds orthogonality_threshold
        with pytest.raises(HypergamicSelectionError, match="orthogonality_threshold must be in"):
            HypergamicConfig(orthogonality_threshold=-0.1)
        with pytest.raises(HypergamicSelectionError, match="orthogonality_threshold must be in"):
            HypergamicConfig(orthogonality_threshold=1.05)

        # Reject non-positive max_mating_attempts
        with pytest.raises(HypergamicSelectionError, match="max_mating_attempts must be >= 1"):
            HypergamicConfig(max_mating_attempts=0)

        # Reject out-of-bounds relaxation_factor
        with pytest.raises(HypergamicSelectionError, match="relaxation_factor must be in"):
            HypergamicConfig(relaxation_factor=0.0)
        with pytest.raises(HypergamicSelectionError, match="relaxation_factor must be in"):
            HypergamicConfig(relaxation_factor=1.5)

        # Reject non-positive tournament_size
        with pytest.raises(HypergamicSelectionError, match="tournament_size must be >= 1"):
            HypergamicConfig(tournament_size=0)

        # Reject non-positive crossover_distribution_index
        with pytest.raises(HypergamicSelectionError, match="crossover_distribution_index must be > 0"):
            HypergamicConfig(crossover_distribution_index=0.0)

        # Reject negative elitism_count
        with pytest.raises(HypergamicSelectionError, match="elitism_count must be >= 0"):
            HypergamicConfig(elitism_count=-1)

    def test_mating_pair_and_offspring_result_immutability(self) -> None:
        """MatingPair and OffspringResult are immutable frozen dataclasses."""
        pair = MatingPair(
            alpha_id="alpha_1",
            aspirant_id="asp_2",
            residual_correlation=0.12,
            is_relaxed=False,
            relaxation_level=0,
        )
        assert pair.alpha_id == "alpha_1"
        assert pair.residual_correlation == 0.12

        with pytest.raises(Exception):
            pair.alpha_id = "alpha_2"  # type: ignore[misc]

        res = OffspringResult(
            offspring_chromosomes=(),
            mating_pairs=(pair,),
            alpha_ids=("alpha_1",),
            aspirant_ids=("asp_2",),
            elite_ids=(),
            rejection_count=0,
            relaxation_count=0,
        )
        assert len(res.mating_pairs) == 1
        with pytest.raises(Exception):
            res.rejection_count = 5  # type: ignore[misc]
