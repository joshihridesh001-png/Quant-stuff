"""Unit tests for evolutionary strategy chromosome architecture and vector codec.

Purpose:
    Validates parameter bounds, scale-free unit hypercube encoding/decoding,
    invariants by construction (timescale ordering and regime simplex),
    discrete quantization midpoint stability, phenotypic distance metric,
    backward-compatible dictionary serialization, typed factory adapters,
    and deterministic error diagnostics.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from quant.analytics.chromosomes import (
    ERR_EVO_CHROM_BOUNDS,
    ERR_EVO_CHROM_CORRUPT,
    ERR_EVO_CHROM_DIM,
    ERR_EVO_CHROM_NON_FINITE,
    GENE_REGISTRY,
    ChromosomeVectorCodec,
    GameTheoryChromosome,
    GeneSpec,
    InferenceChromosome,
    RepresentationChromosome,
    RiskChromosome,
    ScaleType,
    StrategyChromosome,
)
from quant.analytics.labeling import TripleBarrierConfig
from quant.analytics.meta_labeling import MetaLabelConfig
from quant.analytics.minimax_regret import MinimaxRegretConfig
from quant.analytics.payoff_matrix import StackelbergConfig
from quant.analytics.regimes import RegimeConfig


class TestGeneRegistry:
    """Test suite for declarative gene metadata registry."""

    def test_registry_dimension_is_twenty(self) -> None:
        """Verify registry contains exactly D = 20 genes."""
        assert len(GENE_REGISTRY) == 20

    def test_gene_names_are_unique(self) -> None:
        """Verify no duplicate gene names exist in registry."""
        names = [spec.name for spec in GENE_REGISTRY]
        assert len(names) == len(set(names))

    def test_gene_block_distribution(self) -> None:
        """Verify balanced gene distribution across chromosome blocks."""
        blocks: dict[str, int] = {}
        for spec in GENE_REGISTRY:
            blocks[spec.block] = blocks.get(spec.block, 0) + 1

        assert blocks["repr"] == 4
        assert blocks["game"] == 6
        assert blocks["infer"] == 6
        assert blocks["risk"] == 4

    def test_gene_specs_satisfy_invariants(self) -> None:
        """Verify all declared gene bounds and default values satisfy invariants."""
        for spec in GENE_REGISTRY:
            assert spec.lower_bound < spec.upper_bound
            assert spec.step_size > 0.0
            assert spec.lower_bound <= spec.default_value <= spec.upper_bound
            if spec.scale_type == ScaleType.LOGARITHMIC:
                assert spec.lower_bound > 0.0

    def test_gene_spec_invalid_bounds_rejected(self) -> None:
        """Verify ValueError raised when lower_bound >= upper_bound."""
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_BOUNDS):
            GeneSpec(
                name="bad_gene",
                block="risk",
                scale_type=ScaleType.LINEAR,
                lower_bound=1.0,
                upper_bound=1.0,
                default_value=1.0,
                description="Invalid equal bounds",
            )

    def test_gene_spec_invalid_log_bound_rejected(self) -> None:
        """Verify ValueError raised when logarithmic gene has non-positive lower_bound."""
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_BOUNDS):
            GeneSpec(
                name="bad_log_gene",
                block="repr",
                scale_type=ScaleType.LOGARITHMIC,
                lower_bound=0.0,
                upper_bound=10.0,
                default_value=5.0,
                description="Invalid zero lower bound for log gene",
            )

    def test_gene_spec_invalid_step_size_rejected(self) -> None:
        """Verify ValueError raised when step_size is non-positive."""
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_BOUNDS):
            GeneSpec(
                name="bad_step_gene",
                block="infer",
                scale_type=ScaleType.DISCRETE_INT,
                lower_bound=1.0,
                upper_bound=10.0,
                default_value=5.0,
                step_size=-1.0,
                description="Negative step size",
            )


class TestChromosomeVectorCodec:
    """Test suite for continuous unit hypercube encoding and decoding."""

    @pytest.fixture
    def codec(self) -> ChromosomeVectorCodec:
        return ChromosomeVectorCodec()

    @pytest.fixture
    def default_chromosome(self) -> StrategyChromosome:
        return StrategyChromosome()

    def test_codec_dimension(self, codec: ChromosomeVectorCodec) -> None:
        """Verify codec dimension is 20."""
        assert codec.dimension == 20

    def test_codec_invalid_registry_dimension(self) -> None:
        """Verify ValueError on incomplete registry."""
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_DIM):
            ChromosomeVectorCodec(registry=GENE_REGISTRY[:10])

    def test_encode_decode_round_trip_lossless(
        self,
        codec: ChromosomeVectorCodec,
        default_chromosome: StrategyChromosome,
    ) -> None:
        """Verify round-trip encoding and decoding preserves parameter values to high precision."""
        u = codec.encode(default_chromosome)
        assert u.shape == (20,)
        assert np.all(u >= 0.0)
        assert np.all(u <= 1.0)

        decoded = codec.decode(u)

        # Check Representation
        assert math.isclose(
            decoded.representation.tau_slow,
            default_chromosome.representation.tau_slow,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.representation.tau_ratio,
            default_chromosome.representation.tau_ratio,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.representation.alpha_decay,
            default_chromosome.representation.alpha_decay,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.representation.fractional_d,
            default_chromosome.representation.fractional_d,
            rel_tol=1e-8,
        )

        # Check Game Theory
        assert math.isclose(
            decoded.game_theory.ambiguity_temp,
            default_chromosome.game_theory.ambiguity_temp,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.game_theory.risk_aversion,
            default_chromosome.game_theory.risk_aversion,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.game_theory.predatory_intensity,
            default_chromosome.game_theory.predatory_intensity,
            rel_tol=1e-8,
        )

        # Check Inference (exact integer match for discrete genes)
        assert decoded.inference.execution_horizon == default_chromosome.inference.execution_horizon
        assert decoded.inference.holding_period == default_chromosome.inference.holding_period
        assert math.isclose(
            decoded.inference.hyperbolic_decay,
            default_chromosome.inference.hyperbolic_decay,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.inference.profit_take_mult,
            default_chromosome.inference.profit_take_mult,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.inference.stop_loss_mult,
            default_chromosome.inference.stop_loss_mult,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.inference.meta_label_thresh,
            default_chromosome.inference.meta_label_thresh,
            rel_tol=1e-8,
        )

        # Check Risk
        assert math.isclose(
            decoded.risk.vol_target,
            default_chromosome.risk.vol_target,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.risk.max_weight,
            default_chromosome.risk.max_weight,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.risk.max_drawdown_limit,
            default_chromosome.risk.max_drawdown_limit,
            rel_tol=1e-8,
        )
        assert math.isclose(
            decoded.risk.turnover_budget,
            default_chromosome.risk.turnover_budget,
            rel_tol=1e-8,
        )

    def test_logarithmic_scale_geometric_midpoint(self, codec: ChromosomeVectorCodec) -> None:
        """Verify u = 0.5 decodes to geometric mean for logarithmic genes."""
        u = np.full(20, 0.5)
        decoded = codec.decode(u)

        # tau_slow bounds: 14400, 604800
        expected_tau_slow = math.exp(0.5 * (math.log(14400.0) + math.log(604800.0)))
        assert math.isclose(decoded.representation.tau_slow, expected_tau_slow, rel_tol=1e-6)

        # ambiguity_temp bounds: 0.05, 5.0
        expected_temp = math.exp(0.5 * (math.log(0.05) + math.log(5.0)))
        assert math.isclose(decoded.game_theory.ambiguity_temp, expected_temp, rel_tol=1e-6)

    def test_discrete_quantization_midpoint_stability(self, codec: ChromosomeVectorCodec) -> None:
        """Verify discrete gene quantization is immune to small perturbations."""
        chrom = StrategyChromosome(inference=InferenceChromosome(execution_horizon=5))
        u = codec.encode(chrom)

        # Perturb u around bucket midpoint by +/- 0.02 (bucket width is 1/20 = 0.05)
        exec_idx = 10  # index of execution_horizon in GENE_REGISTRY
        u_perturbed_up = u.copy()
        u_perturbed_up[exec_idx] += 0.02
        u_perturbed_down = u.copy()
        u_perturbed_down[exec_idx] -= 0.02

        decoded_up = codec.decode(u_perturbed_up)
        decoded_down = codec.decode(u_perturbed_down)

        assert decoded_up.inference.execution_horizon == 5
        assert decoded_down.inference.execution_horizon == 5

    def test_boundary_capping_and_clipping(self, codec: ChromosomeVectorCodec) -> None:
        """Verify u = 0.0 and u = 1.0 decode to exact physical boundary values."""
        u_zeros = np.zeros(20)
        decoded_min = codec.decode(u_zeros)

        assert decoded_min.inference.execution_horizon == 1
        assert decoded_min.inference.holding_period == 5
        assert math.isclose(decoded_min.representation.tau_slow, 14400.0, rel_tol=1e-6)
        assert math.isclose(decoded_min.risk.max_weight, 0.05, rel_tol=1e-6)

        u_ones = np.ones(20)
        decoded_max = codec.decode(u_ones)

        assert decoded_max.inference.execution_horizon == 20
        assert decoded_max.inference.holding_period == 100
        assert math.isclose(decoded_max.representation.tau_slow, 604800.0, rel_tol=1e-6)
        assert math.isclose(decoded_max.risk.max_weight, 0.50, rel_tol=1e-6)

    def test_out_of_bounds_inputs_clipped(self, codec: ChromosomeVectorCodec) -> None:
        """Verify values < 0 or > 1 are clipped safely without raising exceptions."""
        u_exceeded = np.full(20, 1.5)
        decoded_exceeded = codec.decode(u_exceeded)
        assert decoded_exceeded.inference.execution_horizon == 20
        assert math.isclose(decoded_exceeded.risk.max_weight, 0.50, rel_tol=1e-6)

        u_negative = np.full(20, -0.5)
        decoded_negative = codec.decode(u_negative)
        assert decoded_negative.inference.execution_horizon == 1
        assert math.isclose(decoded_negative.risk.max_weight, 0.05, rel_tol=1e-6)

    def test_timescale_ordering_invariant(self, codec: ChromosomeVectorCodec) -> None:
        """Verify tau_fast < tau_slow is unconditionally guaranteed for 100 random unit vectors."""
        np.random.seed(42)
        for _ in range(100):
            u_rand = np.random.uniform(0.0, 1.0, size=20)
            decoded = codec.decode(u_rand)
            assert decoded.representation.tau_fast < decoded.representation.tau_slow
            assert decoded.representation.tau_ratio <= 0.50

    def test_regime_simplex_probability_invariant(self, codec: ChromosomeVectorCodec) -> None:
        """Verify scenario priors sum to 1.0 and remain strictly positive across 100 random vectors."""
        np.random.seed(42)
        for _ in range(100):
            u_rand = np.random.uniform(0.0, 1.0, size=20)
            decoded = codec.decode(u_rand)
            priors = decoded.game_theory.regime_priors
            assert priors.shape == (3,)
            assert math.isclose(float(np.sum(priors)), 1.0, rel_tol=1e-8)
            assert np.all(priors > 0.0)

    def test_invalid_vector_dimensions_rejected(self, codec: ChromosomeVectorCodec) -> None:
        """Verify ValueError raised on incorrect vector dimension."""
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_DIM):
            codec.decode(np.zeros(19))
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_DIM):
            codec.decode(np.zeros(21))

    def test_non_finite_vector_rejected(self, codec: ChromosomeVectorCodec) -> None:
        """Verify ValueError raised on NaN or Inf in chromosome vector."""
        u_nan = np.zeros(20)
        u_nan[5] = np.nan
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_NON_FINITE):
            codec.decode(u_nan)

        u_inf = np.zeros(20)
        u_inf[12] = np.inf
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_NON_FINITE):
            codec.decode(u_inf)


class TestPhenotypicDistance:
    """Test suite for phenotypic behavioral distance metric."""

    @pytest.fixture
    def codec(self) -> ChromosomeVectorCodec:
        return ChromosomeVectorCodec()

    def test_identity_distance_is_zero(self, codec: ChromosomeVectorCodec) -> None:
        """Verify d(c, c) == 0.0."""
        c = StrategyChromosome()
        assert codec.compute_phenotypic_distance(c, c) == 0.0

    def test_symmetry(self, codec: ChromosomeVectorCodec) -> None:
        """Verify d(c1, c2) == d(c2, c1)."""
        c1 = StrategyChromosome(risk=RiskChromosome(max_weight=0.10))
        c2 = StrategyChromosome(risk=RiskChromosome(max_weight=0.45))
        d12 = codec.compute_phenotypic_distance(c1, c2)
        d21 = codec.compute_phenotypic_distance(c2, c1)
        assert math.isclose(d12, d21, rel_tol=1e-9)

    def test_positivity_for_distinct_chromosomes(self, codec: ChromosomeVectorCodec) -> None:
        """Verify d(c1, c2) > 0.0 for distinct chromosomes."""
        c1 = StrategyChromosome()
        c2 = StrategyChromosome(risk=RiskChromosome(vol_target=0.35))
        assert codec.compute_phenotypic_distance(c1, c2) > 0.0

    def test_gauge_invariance_for_logits(self, codec: ChromosomeVectorCodec) -> None:
        """Verify two chromosomes differing only by constant logit shift have zero simplex distance."""
        # Both [0, 0, 0] and [1, 1, 1] yield priors [1/3, 1/3, 1/3]
        c1 = StrategyChromosome(
            game_theory=GameTheoryChromosome(logit_bull=0.0, logit_bear=0.0, logit_panic=0.0)
        )
        c2 = StrategyChromosome(
            game_theory=GameTheoryChromosome(logit_bull=1.5, logit_bear=1.5, logit_panic=1.5)
        )
        # Because phenotypic distance extracts actual probabilities p in Delta^3, distance is 0
        dist = codec.compute_phenotypic_distance(c1, c2)
        assert math.isclose(dist, 0.0, abs_tol=1e-8)


class TestStrategyChromosomeSerialization:
    """Test suite for dictionary conversion, JSON serialization, and legacy backward compatibility."""

    def test_to_dict_types_are_json_serializable(self) -> None:
        """Verify to_dict() produces strictly native Python types serializable by json.dumps."""
        chrom = StrategyChromosome()
        cdict = chrom.to_dict()

        # Check blocks exist
        assert "chromosome_repr" in cdict
        assert "chromosome_game" in cdict
        assert "chromosome_infer" in cdict
        assert "chromosome_risk" in cdict

        # Must not raise TypeError: Object of type float64 is not JSON serializable
        json_str = json.dumps(cdict)
        assert isinstance(json_str, str)

        # Verify all elements in to_dict are standard Python types
        for block_name, block_dict in cdict.items():
            for key, val in block_dict.items():
                if isinstance(val, list):
                    for item in val:
                        assert type(item) in (float, int, str, bool), f"{key} item is {type(item)}"
                else:
                    assert type(val) in (
                        float,
                        int,
                        str,
                        bool,
                    ), f"{block_name}.{key} is {type(val)}"

    def test_from_dict_lossless_round_trip(self) -> None:
        """Verify chrom == from_dict(to_dict(chrom))."""
        original = StrategyChromosome(
            representation=RepresentationChromosome(
                tau_slow=100000.0,
                tau_ratio=0.15,
                alpha_decay=0.70,
                fractional_d=0.35,
            ),
            game_theory=GameTheoryChromosome(
                ambiguity_temp=2.5,
                risk_aversion=3.0,
                predatory_intensity=0.40,
                logit_bull=0.5,
                logit_bear=-0.5,
                logit_panic=-1.5,
            ),
            inference=InferenceChromosome(
                execution_horizon=8,
                hyperbolic_decay=0.80,
                profit_take_mult=3.0,
                stop_loss_mult=1.5,
                holding_period=30,
                meta_label_thresh=0.65,
            ),
            risk=RiskChromosome(
                vol_target=0.20,
                max_weight=0.25,
                max_drawdown_limit=0.12,
                turnover_budget=0.60,
            ),
        )

        cdict = original.to_dict()
        reconstructed = StrategyChromosome.from_dict(cdict)

        assert math.isclose(reconstructed.representation.tau_slow, original.representation.tau_slow)
        assert math.isclose(
            reconstructed.representation.tau_ratio, original.representation.tau_ratio
        )
        assert math.isclose(
            reconstructed.representation.alpha_decay, original.representation.alpha_decay
        )
        assert math.isclose(
            reconstructed.representation.fractional_d, original.representation.fractional_d
        )
        assert math.isclose(
            reconstructed.game_theory.ambiguity_temp, original.game_theory.ambiguity_temp
        )
        assert reconstructed.inference.execution_horizon == original.inference.execution_horizon
        assert reconstructed.inference.holding_period == original.inference.holding_period
        assert math.isclose(reconstructed.risk.max_weight, original.risk.max_weight)

    def test_legacy_dictionary_backward_compatibility(self) -> None:
        """Verify legacy seed dictionaries decode seamlessly with alias translation."""
        legacy_payload: dict[str, Any] = {
            "chromosome_repr": {
                "tau_fast": 7200.0,
                "tau_slow": 86400.0,
                "alpha": 0.60,
            },
            "chromosome_game": {
                "risk_aversion_lambda": 2.0,
                "belief_prior": [0.50, 0.30, 0.20],
            },
            "chromosome_infer": {
                "model_depth": 3,
                "sensitivity": 0.55,
            },
            "chromosome_risk": {
                "vol_target": 0.18,
                "max_drawdown_limit": 0.08,
            },
        }

        chrom = StrategyChromosome.from_dict(legacy_payload)

        # Aliases mapped
        assert math.isclose(chrom.representation.tau_slow, 86400.0)
        assert math.isclose(chrom.representation.tau_fast, 7200.0)
        assert math.isclose(chrom.representation.tau_ratio, 7200.0 / 86400.0)
        assert math.isclose(chrom.representation.alpha_decay, 0.60)
        assert math.isclose(chrom.game_theory.risk_aversion, 2.0)
        assert math.isclose(chrom.inference.meta_label_thresh, 0.55)

        # Missing fields populated with institutional defaults
        assert chrom.representation.fractional_d == 0.40
        assert chrom.inference.execution_horizon == 5
        assert chrom.risk.max_weight == 0.30

        # Priors preserved in decoded simplex
        priors = chrom.game_theory.regime_priors
        assert math.isclose(priors[0], 0.50, rel_tol=1e-2)
        assert math.isclose(priors[1], 0.30, rel_tol=1e-2)
        assert math.isclose(priors[2], 0.20, rel_tol=1e-2)

    def test_empty_dictionary_fallback_to_defaults(self) -> None:
        """Verify empty dictionary creates valid chromosome with all baseline defaults."""
        chrom = StrategyChromosome.from_dict({})
        assert chrom.representation.tau_slow == 86400.0
        assert chrom.game_theory.risk_aversion == 1.0
        assert chrom.inference.execution_horizon == 5
        assert chrom.risk.max_weight == 0.30

    def test_corrupt_dictionary_type_rejected(self) -> None:
        """Verify ValueError raised when passing non-dict object."""
        with pytest.raises(ValueError, match=ERR_EVO_CHROM_CORRUPT):
            StrategyChromosome.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


class TestFactoryAdapters:
    """Test suite for typed analytics configuration factory adapters."""

    @pytest.fixture
    def chromosome(self) -> StrategyChromosome:
        return StrategyChromosome(
            game_theory=GameTheoryChromosome(
                ambiguity_temp=1.5,
                risk_aversion=2.5,
                predatory_intensity=0.35,
            ),
            inference=InferenceChromosome(
                execution_horizon=7,
                hyperbolic_decay=0.60,
                profit_take_mult=2.5,
                stop_loss_mult=1.8,
                holding_period=25,
                meta_label_thresh=0.60,
            ),
            risk=RiskChromosome(max_weight=0.25),
        )

    def test_to_stackelberg_config(self, chromosome: StrategyChromosome) -> None:
        """Verify StackelbergConfig adapter maps parameters correctly."""
        cfg = chromosome.to_stackelberg_config()
        assert isinstance(cfg, StackelbergConfig)
        assert cfg.execution_horizon == 7
        assert cfg.hyperbolic_decay_rate == 0.60
        assert cfg.risk_aversion == 2.5
        assert cfg.predatory_shading_intensity == 0.35
        assert cfg.max_weight_per_asset == 0.25
        assert cfg.gross_leverage_limit == 1.0

    def test_to_regime_config(self, chromosome: StrategyChromosome) -> None:
        """Verify RegimeConfig adapter maps parameters correctly."""
        cfg = chromosome.to_regime_config()
        assert isinstance(cfg, RegimeConfig)
        assert cfg.n_regimes == 3
        assert cfg.temperature_scale == 1.5

    def test_to_minimax_config(self, chromosome: StrategyChromosome) -> None:
        """Verify MinimaxRegretConfig adapter maps parameters correctly."""
        cfg = chromosome.to_minimax_config()
        assert isinstance(cfg, MinimaxRegretConfig)
        assert cfg.max_iterations == 25
        assert cfg.gradient_tolerance == 1e-6

    def test_to_triple_barrier_config(self, chromosome: StrategyChromosome) -> None:
        """Verify TripleBarrierConfig adapter maps parameters correctly."""
        cfg = chromosome.to_triple_barrier_config()
        assert isinstance(cfg, TripleBarrierConfig)
        assert cfg.profit_multiplier == 2.5
        assert cfg.stop_multiplier == 1.8
        assert cfg.horizon_bars == 25

    def test_to_meta_label_config(self, chromosome: StrategyChromosome) -> None:
        """Verify MetaLabelConfig adapter maps parameters correctly."""
        cfg = chromosome.to_meta_label_config()
        assert isinstance(cfg, MetaLabelConfig)
        assert cfg.kelly_fraction == 0.60
        assert cfg.max_leverage == 1.0
