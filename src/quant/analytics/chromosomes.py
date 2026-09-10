"""Evolutionary strategy chromosome encoding, parameter registry, and vector codec.

Purpose:
    Provides the declarative chromosome architecture and scale-free vector codec for
    the evolutionary strategy population engine. Encodes strategy hyperparameters across
    representation, game theory, inference, and risk into a normalized unit hypercube
    u in [0, 1]^D, enabling scale-invariant genetic crossover and mutation.

Dependencies:
    - numpy: Vectorized array operations and numerical clipping.
    - standard library: dataclasses, enum, math, typing.
    - quant.analytics.payoff_matrix: StackelbergConfig factory target.
    - quant.analytics.regimes: RegimeConfig factory target.
    - quant.analytics.minimax_regret: MinimaxRegretConfig factory target.
    - quant.analytics.labeling: TripleBarrierConfig factory target.
    - quant.analytics.meta_labeling: MetaLabelingConfig factory target.

Structural Relationship:
    - Encapsulated by Genotype domain model (quant.domain.models.Genotype).
    - Persisted into PostgreSQL / SQLite JSON columns via DBGenotype.
    - Consumed by Step 2 (Pareto fitness & novelty search), Step 3 (hypergamic mating),
      and Step 4 (evolutionary generation lifecycle).

Invariants:
    - Total registry dimension is exactly D = 20 genes across 4 blocks.
    - Every encoded unit vector satisfies 0.0 <= u_i <= 1.0.
    - Timescale ordering tau_fast < tau_slow is unconditionally guaranteed by construction
      via ratio parameterization tau_fast = tau_ratio * tau_slow with tau_ratio in [0.02, 0.50].
    - Regime scenario priors strictly satisfy sum(p_j) == 1.0 and p_j > 0.0 via softmax logits.
    - Discrete genes are encoded at bucket midpoints u(k) = (k + 0.5) / K to guarantee exact
      lossless round-trip stability against IEEE-754 floating-point drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from quant.analytics.labeling import TripleBarrierConfig
from quant.analytics.meta_labeling import MetaLabelConfig
from quant.analytics.minimax_regret import MinimaxRegretConfig
from quant.analytics.payoff_matrix import StackelbergConfig
from quant.analytics.regimes import RegimeConfig

# Deterministic Diagnostic Codes (Rule 2: Deterministic Diagnostics)
ERR_EVO_CHROM_BOUNDS = "ERR-EVO-CHROM-001: Gene value outside declared physical bounds"
ERR_EVO_CHROM_INVARIANT = "ERR-EVO-CHROM-002: Relational or simplex invariant violation"
ERR_EVO_CHROM_DIM = "ERR-EVO-CHROM-003: Vector dimension mismatch with gene registry"
ERR_EVO_CHROM_NON_FINITE = "ERR-EVO-CHROM-004: Non-finite numerical element encountered"
ERR_EVO_CHROM_CORRUPT = "ERR-EVO-CHROM-005: Unparseable or corrupt chromosome dictionary"


class ScaleType(StrEnum):
    """Scaling transformation rule mapping physical values to the unit hypercube [0, 1].

    Purpose:
        Distinguishes linear, logarithmic, softmax logit, and discrete integer parameters
        to ensure scale-invariant mutation and crossover sensitivity across diverse units.
    """

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    SOFTMAX_LOGIT = "softmax_logit"
    DISCRETE_INT = "discrete_int"


@dataclass(frozen=True)
class GeneSpec:
    """Declarative specification for a single chromosome parameter.

    Purpose:
        Declares parameter bounds, scaling laws, default values, and discretization
        steps to enable polymorphic encoding and decoding across genetic space.

    Attributes:
        name: Unique identifier string for the gene.
        block: Categorical chromosome block ('repr', 'game', 'infer', 'risk').
        scale_type: ScaleType transformation rule.
        lower_bound: Minimum allowable physical parameter value.
        upper_bound: Maximum allowable physical parameter value.
        default_value: Baseline institutional default value.
        description: Plain-English explanation of parameter function.
        step_size: Discrete integer increment step (applicable for DISCRETE_INT).
    """

    name: str
    block: str
    scale_type: ScaleType
    lower_bound: float
    upper_bound: float
    default_value: float
    description: str
    step_size: float = 1.0

    def __post_init__(self) -> None:
        """Validate invariant metadata bounds."""
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                f"{ERR_EVO_CHROM_BOUNDS}: lower_bound ({self.lower_bound}) "
                f"must be strictly less than upper_bound ({self.upper_bound}) for gene '{self.name}'"
            )
        if self.scale_type == ScaleType.LOGARITHMIC and self.lower_bound <= 0.0:
            raise ValueError(
                f"{ERR_EVO_CHROM_BOUNDS}: logarithmic gene '{self.name}' "
                f"must have lower_bound > 0.0, got {self.lower_bound}"
            )
        if self.step_size <= 0.0:
            raise ValueError(
                f"{ERR_EVO_CHROM_BOUNDS}: step_size must be strictly positive, got {self.step_size}"
            )


# Static Declarative Gene Registry (D = 20 genes)
GENE_REGISTRY: list[GeneSpec] = [
    # Block 1: Representation (4 genes)
    GeneSpec(
        name="tau_slow",
        block="repr",
        scale_type=ScaleType.LOGARITHMIC,
        lower_bound=14400.0,  # 4 hours
        upper_bound=604800.0,  # 7 days
        default_value=86400.0,  # 24 hours
        description="Slow sentiment decay half-life in seconds",
    ),
    GeneSpec(
        name="tau_ratio",
        block="repr",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.02,  # 2% of tau_slow
        upper_bound=0.50,  # 50% of tau_slow
        default_value=0.10,  # 10% of tau_slow
        description="Fast-to-slow decay half-life ratio governing tau_fast = tau_ratio * tau_slow",
    ),
    GeneSpec(
        name="alpha_decay",
        block="repr",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.0,
        upper_bound=1.0,
        default_value=0.50,
        description="Fast versus slow sentiment kernel blending weight",
    ),
    GeneSpec(
        name="fractional_d",
        block="repr",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.05,
        upper_bound=0.95,
        default_value=0.40,
        description="Memory-preserving fractional differentiation degree d",
    ),
    # Block 2: Game Theory & Regimes (6 genes)
    GeneSpec(
        name="ambiguity_temp",
        block="game",
        scale_type=ScaleType.LOGARITHMIC,
        lower_bound=0.05,
        upper_bound=5.0,
        default_value=1.0,
        description="Worst-case adversary ambiguity temperature tau",
    ),
    GeneSpec(
        name="risk_aversion",
        block="game",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.1,
        upper_bound=10.0,
        default_value=1.0,
        description="Portfolio variance risk aversion penalty gamma",
    ),
    GeneSpec(
        name="predatory_intensity",
        block="game",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.0,
        upper_bound=1.0,
        default_value=0.25,
        description="Market maker predatory quote-shading intensity theta_pred",
    ),
    GeneSpec(
        name="logit_bull",
        block="game",
        scale_type=ScaleType.SOFTMAX_LOGIT,
        lower_bound=-2.0,
        upper_bound=2.0,
        default_value=0.0,
        description="Unconstrained logit for bull regime scenario prior",
    ),
    GeneSpec(
        name="logit_bear",
        block="game",
        scale_type=ScaleType.SOFTMAX_LOGIT,
        lower_bound=-2.0,
        upper_bound=2.0,
        default_value=0.0,
        description="Unconstrained logit for bear regime scenario prior",
    ),
    GeneSpec(
        name="logit_panic",
        block="game",
        scale_type=ScaleType.SOFTMAX_LOGIT,
        lower_bound=-2.0,
        upper_bound=2.0,
        default_value=-1.0,
        description="Unconstrained logit for panic regime scenario prior",
    ),
    # Block 3: Inference & Execution (6 genes)
    GeneSpec(
        name="execution_horizon",
        block="infer",
        scale_type=ScaleType.DISCRETE_INT,
        lower_bound=1.0,
        upper_bound=20.0,
        default_value=5.0,
        step_size=1.0,
        description="Order slicing horizon in discrete bars H",
    ),
    GeneSpec(
        name="hyperbolic_decay",
        block="infer",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.01,
        upper_bound=2.0,
        default_value=0.50,
        description="Hyperbolic front-loading trajectory curvature kappa",
    ),
    GeneSpec(
        name="profit_take_mult",
        block="infer",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.5,
        upper_bound=5.0,
        default_value=2.0,
        description="Dynamic volatility profit-taking barrier multiple k_pt",
    ),
    GeneSpec(
        name="stop_loss_mult",
        block="infer",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.5,
        upper_bound=5.0,
        default_value=2.0,
        description="Dynamic volatility stop-loss barrier multiple k_sl",
    ),
    GeneSpec(
        name="holding_period",
        block="infer",
        scale_type=ScaleType.DISCRETE_INT,
        lower_bound=5.0,
        upper_bound=100.0,
        default_value=20.0,
        step_size=5.0,
        description="Maximum vertical barrier duration in bars T_hold",
    ),
    GeneSpec(
        name="meta_label_thresh",
        block="infer",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.40,
        upper_bound=0.85,
        default_value=0.50,
        description="Minimum probability hurdle for directional Kelly bet sizing",
    ),
    # Block 4: Risk Management (4 genes)
    GeneSpec(
        name="vol_target",
        block="risk",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.05,
        upper_bound=0.40,
        default_value=0.15,
        description="Target annualized portfolio volatility sigma_target",
    ),
    GeneSpec(
        name="max_weight",
        block="risk",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.05,
        upper_bound=0.50,
        default_value=0.30,
        description="Maximum single-asset portfolio allocation cap w_max",
    ),
    GeneSpec(
        name="max_drawdown_limit",
        block="risk",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.02,
        upper_bound=0.30,
        default_value=0.10,
        description="Maximum allowable peak-to-trough equity drawdown limit",
    ),
    GeneSpec(
        name="turnover_budget",
        block="risk",
        scale_type=ScaleType.LINEAR,
        lower_bound=0.05,
        upper_bound=1.0,
        default_value=0.50,
        description="Maximum per-rebalance portfolio turnover fraction",
    ),
]

# Quick lookup mappings
_GENE_MAP: dict[str, GeneSpec] = {spec.name: spec for spec in GENE_REGISTRY}


@dataclass(frozen=True)
class RepresentationChromosome:
    """Sub-chromosome governing news sentiment feature decay and fractional memory.

    Attributes:
        tau_slow: Slow sentiment decay half-life in seconds [14400, 604800].
        tau_ratio: Ratio r_tau in [0.02, 0.50] defining tau_fast = r_tau * tau_slow.
        alpha_decay: Fast versus slow blending weight in [0.0, 1.0].
        fractional_d: Fractional differentiation degree d in [0.05, 0.95].
    """

    tau_slow: float = 86400.0
    tau_ratio: float = 0.10
    alpha_decay: float = 0.50
    fractional_d: float = 0.40

    @property
    def tau_fast(self) -> float:
        """Fast sentiment decay half-life in seconds (invariant: tau_fast < tau_slow)."""
        return float(self.tau_ratio * self.tau_slow)


@dataclass(frozen=True)
class GameTheoryChromosome:
    """Sub-chromosome governing adversary ambiguity and game-theoretic payoffs.

    Attributes:
        ambiguity_temp: Adversary worst-case ambiguity temperature tau in [0.05, 5.0].
        risk_aversion: Portfolio variance risk penalty gamma in [0.1, 10.0].
        predatory_intensity: Quote-shading drag theta_pred in [0.0, 1.0].
        logit_bull: Unconstrained logit for bull scenario prior in [-2.0, 2.0].
        logit_bear: Unconstrained logit for bear scenario prior in [-2.0, 2.0].
        logit_panic: Unconstrained logit for panic scenario prior in [-2.0, 2.0].
    """

    ambiguity_temp: float = 1.0
    risk_aversion: float = 1.0
    predatory_intensity: float = 0.25
    logit_bull: float = 0.0
    logit_bear: float = 0.0
    logit_panic: float = -1.0

    @property
    def regime_priors(self) -> np.ndarray:
        """Normalized scenario prior probability simplex p in Delta^3 (sum == 1, p_j > 0)."""
        logits = np.array([self.logit_bull, self.logit_bear, self.logit_panic], dtype=np.float64)
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        priors: np.ndarray = exp_logits / np.sum(exp_logits)
        return priors


@dataclass(frozen=True)
class InferenceChromosome:
    """Sub-chromosome governing signal thresholds and order-slicing execution.

    Attributes:
        execution_horizon: Slicing horizon bars H in {1, ..., 20}.
        hyperbolic_decay: Front-loading curvature kappa in [0.01, 2.0].
        profit_take_mult: Profit-taking volatility multiple k_pt in [0.5, 5.0].
        stop_loss_mult: Stop-loss volatility multiple k_sl in [0.5, 5.0].
        holding_period: Maximum vertical barrier duration in bars T_hold in {5, ..., 100}.
        meta_label_thresh: Minimum Kelly meta-labeling confidence hurdle in [0.40, 0.85].
    """

    execution_horizon: int = 5
    hyperbolic_decay: float = 0.50
    profit_take_mult: float = 2.0
    stop_loss_mult: float = 2.0
    holding_period: int = 20
    meta_label_thresh: float = 0.50


@dataclass(frozen=True)
class RiskChromosome:
    """Sub-chromosome governing capital exposure limits and drawdown stops.

    Attributes:
        vol_target: Annualized portfolio volatility target in [0.05, 0.40].
        max_weight: Maximum single-asset concentration cap w_max in [0.05, 0.50].
        max_drawdown_limit: Maximum allowable peak-to-trough equity drawdown in [0.02, 0.30].
        turnover_budget: Maximum per-rebalance turnover fraction in [0.05, 1.0].
    """

    vol_target: float = 0.15
    max_weight: float = 0.30
    max_drawdown_limit: float = 0.10
    turnover_budget: float = 0.50


@dataclass(frozen=True)
class StrategyChromosome:
    """Complete candidate strategy chromosome container g_k = <g_repr, g_game, g_infer, g_risk>.

    Purpose:
        Serves as the unified domain object encapsulating all algorithmic hyperparameters.
        Provides bidirectional mapping to database JSON dictionaries and typed factory
        adapters to instantiate Phase 1-3 analytics engines.
    """

    representation: RepresentationChromosome = RepresentationChromosome()
    game_theory: GameTheoryChromosome = GameTheoryChromosome()
    inference: InferenceChromosome = InferenceChromosome()
    risk: RiskChromosome = RiskChromosome()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize chromosome into native Python dictionaries for DBGenotype JSON columns.

        Guarantees:
            - All numbers are explicitly cast to native Python float or int, preventing
              numpy.float64 serialization crashes during database or API JSON encoding.
            - Includes both modern keys and legacy aliases (e.g. risk_aversion_lambda, belief_prior)
              for complete backward compatibility.
        """
        priors = self.game_theory.regime_priors
        return {
            "chromosome_repr": {
                "tau_slow": float(self.representation.tau_slow),
                "tau_ratio": float(self.representation.tau_ratio),
                "tau_fast": float(self.representation.tau_fast),
                "alpha": float(self.representation.alpha_decay),
                "alpha_decay": float(self.representation.alpha_decay),
                "fractional_d": float(self.representation.fractional_d),
            },
            "chromosome_game": {
                "ambiguity_temp": float(self.game_theory.ambiguity_temp),
                "risk_aversion": float(self.game_theory.risk_aversion),
                "risk_aversion_lambda": float(self.game_theory.risk_aversion),
                "predatory_intensity": float(self.game_theory.predatory_intensity),
                "logit_bull": float(self.game_theory.logit_bull),
                "logit_bear": float(self.game_theory.logit_bear),
                "logit_panic": float(self.game_theory.logit_panic),
                "belief_prior": [float(p) for p in priors],
            },
            "chromosome_infer": {
                "execution_horizon": int(self.inference.execution_horizon),
                "hyperbolic_decay": float(self.inference.hyperbolic_decay),
                "profit_take_mult": float(self.inference.profit_take_mult),
                "stop_loss_mult": float(self.inference.stop_loss_mult),
                "holding_period": int(self.inference.holding_period),
                "meta_label_thresh": float(self.inference.meta_label_thresh),
                "model_depth": 3,
                "sensitivity": float(self.inference.meta_label_thresh),
            },
            "chromosome_risk": {
                "vol_target": float(self.risk.vol_target),
                "max_weight": float(self.risk.max_weight),
                "max_drawdown_limit": float(self.risk.max_drawdown_limit),
                "turnover_budget": float(self.risk.turnover_budget),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyChromosome:
        """Reconstruct StrategyChromosome from database JSON dictionary or legacy DTO.

        Handles:
            - Nested dictionaries ('chromosome_repr', etc.) or flattened dictionary.
            - Legacy key aliases: 'risk_aversion_lambda' -> 'risk_aversion', 'alpha' -> 'alpha_decay'.
            - Conversion of legacy 'belief_prior' probability vectors into zero-centered logits.
            - Automatic substitution of institutional default values for any omitted parameters.
        """
        if not isinstance(data, dict):
            raise ValueError(f"{ERR_EVO_CHROM_CORRUPT}: Expected dict, got {type(data)}")

        # Extract nested blocks if present, otherwise treat as flat dict
        c_repr = data.get("chromosome_repr", data)
        c_game = data.get("chromosome_game", data)
        c_infer = data.get("chromosome_infer", data)
        c_risk = data.get("chromosome_risk", data)

        # Parse Representation
        tau_slow = float(c_repr.get("tau_slow", _GENE_MAP["tau_slow"].default_value))
        if "tau_ratio" in c_repr:
            tau_ratio = float(c_repr["tau_ratio"])
        elif "tau_fast" in c_repr and tau_slow > 0.0:
            tau_ratio = float(np.clip(float(c_repr["tau_fast"]) / tau_slow, 0.02, 0.50))
        else:
            tau_ratio = float(_GENE_MAP["tau_ratio"].default_value)

        alpha_decay = float(
            c_repr.get("alpha_decay", c_repr.get("alpha", _GENE_MAP["alpha_decay"].default_value))
        )
        fractional_d = float(c_repr.get("fractional_d", _GENE_MAP["fractional_d"].default_value))

        repr_chrom = RepresentationChromosome(
            tau_slow=float(np.clip(tau_slow, 14400.0, 604800.0)),
            tau_ratio=float(np.clip(tau_ratio, 0.02, 0.50)),
            alpha_decay=float(np.clip(alpha_decay, 0.0, 1.0)),
            fractional_d=float(np.clip(fractional_d, 0.05, 0.95)),
        )

        # Parse Game Theory
        ambiguity_temp = float(
            c_game.get("ambiguity_temp", _GENE_MAP["ambiguity_temp"].default_value)
        )
        risk_aversion = float(
            c_game.get(
                "risk_aversion",
                c_game.get("risk_aversion_lambda", _GENE_MAP["risk_aversion"].default_value),
            )
        )
        predatory_intensity = float(
            c_game.get("predatory_intensity", _GENE_MAP["predatory_intensity"].default_value)
        )

        if "logit_bull" in c_game and "logit_bear" in c_game and "logit_panic" in c_game:
            logit_bull = float(c_game["logit_bull"])
            logit_bear = float(c_game["logit_bear"])
            logit_panic = float(c_game["logit_panic"])
        elif "belief_prior" in c_game and isinstance(c_game["belief_prior"], (list, tuple)):
            priors = np.array(c_game["belief_prior"], dtype=np.float64)
            if len(priors) != 3 or np.any(priors <= 0.0):
                logit_bull, logit_bear, logit_panic = 0.0, 0.0, -1.0
            else:
                raw_logits = np.log(priors)
                zero_mean = raw_logits - np.mean(raw_logits)
                logit_bull = float(np.clip(zero_mean[0], -2.0, 2.0))
                logit_bear = float(np.clip(zero_mean[1], -2.0, 2.0))
                logit_panic = float(np.clip(zero_mean[2], -2.0, 2.0))
        else:
            logit_bull = float(_GENE_MAP["logit_bull"].default_value)
            logit_bear = float(_GENE_MAP["logit_bear"].default_value)
            logit_panic = float(_GENE_MAP["logit_panic"].default_value)

        game_chrom = GameTheoryChromosome(
            ambiguity_temp=float(np.clip(ambiguity_temp, 0.05, 5.0)),
            risk_aversion=float(np.clip(risk_aversion, 0.1, 10.0)),
            predatory_intensity=float(np.clip(predatory_intensity, 0.0, 1.0)),
            logit_bull=float(np.clip(logit_bull, -2.0, 2.0)),
            logit_bear=float(np.clip(logit_bear, -2.0, 2.0)),
            logit_panic=float(np.clip(logit_panic, -2.0, 2.0)),
        )

        # Parse Inference
        execution_horizon = int(
            c_infer.get("execution_horizon", _GENE_MAP["execution_horizon"].default_value)
        )
        hyperbolic_decay = float(
            c_infer.get("hyperbolic_decay", _GENE_MAP["hyperbolic_decay"].default_value)
        )
        profit_take_mult = float(
            c_infer.get("profit_take_mult", _GENE_MAP["profit_take_mult"].default_value)
        )
        stop_loss_mult = float(
            c_infer.get("stop_loss_mult", _GENE_MAP["stop_loss_mult"].default_value)
        )
        holding_period = int(
            c_infer.get("holding_period", _GENE_MAP["holding_period"].default_value)
        )
        meta_label_thresh = float(
            c_infer.get(
                "meta_label_thresh",
                c_infer.get("sensitivity", _GENE_MAP["meta_label_thresh"].default_value),
            )
        )

        infer_chrom = InferenceChromosome(
            execution_horizon=int(np.clip(execution_horizon, 1, 20)),
            hyperbolic_decay=float(np.clip(hyperbolic_decay, 0.01, 2.0)),
            profit_take_mult=float(np.clip(profit_take_mult, 0.5, 5.0)),
            stop_loss_mult=float(np.clip(stop_loss_mult, 0.5, 5.0)),
            holding_period=int(np.clip(holding_period, 5, 100)),
            meta_label_thresh=float(np.clip(meta_label_thresh, 0.40, 0.85)),
        )

        # Parse Risk
        vol_target = float(c_risk.get("vol_target", _GENE_MAP["vol_target"].default_value))
        max_weight = float(c_risk.get("max_weight", _GENE_MAP["max_weight"].default_value))
        max_drawdown_limit = float(
            c_risk.get("max_drawdown_limit", _GENE_MAP["max_drawdown_limit"].default_value)
        )
        turnover_budget = float(
            c_risk.get("turnover_budget", _GENE_MAP["turnover_budget"].default_value)
        )

        risk_chrom = RiskChromosome(
            vol_target=float(np.clip(vol_target, 0.05, 0.40)),
            max_weight=float(np.clip(max_weight, 0.05, 0.50)),
            max_drawdown_limit=float(np.clip(max_drawdown_limit, 0.02, 0.30)),
            turnover_budget=float(np.clip(turnover_budget, 0.05, 1.0)),
        )

        return cls(
            representation=repr_chrom,
            game_theory=game_chrom,
            inference=infer_chrom,
            risk=risk_chrom,
        )

    # Typed Factory Adapters to Phase 1-3 Engine Configurations
    def to_stackelberg_config(self) -> StackelbergConfig:
        """Instantiate StackelbergConfig parameterizing Phase 3 StackelbergPayoffEngine."""
        return StackelbergConfig(
            execution_horizon=self.inference.execution_horizon,
            hyperbolic_decay_rate=self.inference.hyperbolic_decay,
            risk_aversion=self.game_theory.risk_aversion,
            predatory_shading_intensity=self.game_theory.predatory_intensity,
            max_weight_per_asset=self.risk.max_weight,
            gross_leverage_limit=1.0,
        )

    def to_regime_config(self) -> RegimeConfig:
        """Instantiate RegimeConfig parameterizing Phase 3 CausalBayesianRegimeFilter."""
        return RegimeConfig(
            n_regimes=3,
            temperature_scale=self.game_theory.ambiguity_temp,
        )

    def to_minimax_config(self) -> MinimaxRegretConfig:
        """Instantiate MinimaxRegretConfig parameterizing Phase 3 VectorizedNewtonSolver."""
        return MinimaxRegretConfig(
            max_iterations=25,
            gradient_tolerance=1e-6,
            step_tolerance=1e-8,
        )

    def to_triple_barrier_config(self) -> TripleBarrierConfig:
        """Instantiate TripleBarrierConfig parameterizing Phase 2 dynamic labeling."""
        return TripleBarrierConfig(
            profit_multiplier=self.inference.profit_take_mult,
            stop_multiplier=self.inference.stop_loss_mult,
            horizon_bars=self.inference.holding_period,
        )

    def to_meta_label_config(self) -> MetaLabelConfig:
        """Instantiate MetaLabelConfig parameterizing Phase 2 Kelly bet sizing."""
        return MetaLabelConfig(
            kelly_fraction=self.inference.meta_label_thresh,
            max_leverage=1.0,
        )


class ChromosomeVectorCodec:
    """Vector encoder, decoder, and phenotypic distance evaluator for StrategyChromosome.

    Purpose:
        Maps StrategyChromosome bidirectionally between structured domain representations
        and a 20-dimensional scale-free unit hypercube u in [0, 1]^20.

    Mathematical Guarantees:
        - Scale Invariance: Logarithmic scaling ensures identical mutation step sensitivity
          across multi-order-of-magnitude ranges (half-lives, temperature).
        - Midpoint Quantization: Discrete parameters are centered at u(k) = (k + 0.5) / K,
          providing a +/- 2.5% safety buffer against floating-point round-trip distortion.
        - Invariance by Construction: Encoded vectors cannot violate ordering or simplex
          constraints upon decoding.
        - Gauge-Fixed Phenotypic Metric: Evaluates distance using decoded regime probabilities
          p in Delta^3 instead of unconstrained logits, preventing artificial diversity.
    """

    def __init__(self, registry: list[GeneSpec] | None = None) -> None:
        self.registry = registry if registry is not None else GENE_REGISTRY
        self.dimension = len(self.registry)
        if self.dimension != 20:
            raise ValueError(
                f"{ERR_EVO_CHROM_DIM}: Expected D = 20 genes in registry, got {self.dimension}"
            )

    def encode(self, chromosome: StrategyChromosome) -> np.ndarray:
        """Encode StrategyChromosome into continuous unit hypercube vector u in [0, 1]^20.

        Args:
            chromosome: Structured domain chromosome instance.

        Returns:
            np.ndarray of shape (20,) with elements strictly in [0.0, 1.0].
        """
        u = np.zeros(self.dimension, dtype=np.float64)

        for i, gene in enumerate(self.registry):
            # Extract raw physical value from sub-chromosomes
            if gene.block == "repr":
                val = getattr(chromosome.representation, gene.name)
            elif gene.block == "game":
                val = getattr(chromosome.game_theory, gene.name)
            elif gene.block == "infer":
                val = getattr(chromosome.inference, gene.name)
            elif gene.block == "risk":
                val = getattr(chromosome.risk, gene.name)
            else:
                raise ValueError(f"{ERR_EVO_CHROM_CORRUPT}: Unknown gene block '{gene.block}'")

            # Scale to unit interval [0, 1]
            if gene.scale_type == ScaleType.LINEAR or gene.scale_type == ScaleType.SOFTMAX_LOGIT:
                u[i] = (val - gene.lower_bound) / (gene.upper_bound - gene.lower_bound)
            elif gene.scale_type == ScaleType.LOGARITHMIC:
                log_val = math.log(max(val, 1e-12))
                log_min = math.log(gene.lower_bound)
                log_max = math.log(gene.upper_bound)
                u[i] = (log_val - log_min) / (log_max - log_min)
            elif gene.scale_type == ScaleType.DISCRETE_INT:
                n_steps = int(round((gene.upper_bound - gene.lower_bound) / gene.step_size)) + 1
                step_idx = int(round((val - gene.lower_bound) / gene.step_size))
                # Bucket midpoint centering: (k + 0.5) / K
                u[i] = (step_idx + 0.5) / n_steps
            else:
                raise ValueError(
                    f"{ERR_EVO_CHROM_CORRUPT}: Unsupported scale type {gene.scale_type}"
                )

        return np.clip(u, 0.0, 1.0)

    def decode(self, u: np.ndarray) -> StrategyChromosome:
        """Decode continuous unit hypercube vector u in [0, 1]^20 into StrategyChromosome.

        Args:
            u: 1D numpy array of shape (20,) representing the genetic chromosome.

        Returns:
            Decoded, constraint-guaranteed StrategyChromosome.

        Raises:
            ValueError: If u has incorrect dimension or non-finite elements.
        """
        if not isinstance(u, np.ndarray) or u.ndim != 1 or u.shape[0] != self.dimension:
            raise ValueError(
                f"{ERR_EVO_CHROM_DIM}: Expected 1D array of shape ({self.dimension},), "
                f"got {getattr(u, 'shape', type(u))}"
            )
        if not np.all(np.isfinite(u)):
            raise ValueError(f"{ERR_EVO_CHROM_NON_FINITE}: Vector contains NaN or Inf elements")

        # Project into valid unit hypercube
        u_clipped = np.clip(u, 0.0, 1.0)
        decoded_values: dict[str, Any] = {}

        for i, gene in enumerate(self.registry):
            u_i = float(u_clipped[i])

            if gene.scale_type == ScaleType.LINEAR or gene.scale_type == ScaleType.SOFTMAX_LOGIT:
                phys_val = gene.lower_bound + u_i * (gene.upper_bound - gene.lower_bound)
            elif gene.scale_type == ScaleType.LOGARITHMIC:
                log_min = math.log(gene.lower_bound)
                log_max = math.log(gene.upper_bound)
                phys_val = math.exp(log_min + u_i * (log_max - log_min))
            elif gene.scale_type == ScaleType.DISCRETE_INT:
                n_steps = int(round((gene.upper_bound - gene.lower_bound) / gene.step_size)) + 1
                # Bounded quantization
                step_idx = min(n_steps - 1, max(0, int(math.floor(u_i * n_steps))))
                phys_val = gene.lower_bound + step_idx * gene.step_size
            else:
                raise ValueError(
                    f"{ERR_EVO_CHROM_CORRUPT}: Unsupported scale type {gene.scale_type}"
                )

            decoded_values[gene.name] = phys_val

        # Zero-center softmax logits to enforce gauge symmetry
        logit_bull = decoded_values["logit_bull"]
        logit_bear = decoded_values["logit_bear"]
        logit_panic = decoded_values["logit_panic"]
        mean_logit = (logit_bull + logit_bear + logit_panic) / 3.0

        return StrategyChromosome(
            representation=RepresentationChromosome(
                tau_slow=float(decoded_values["tau_slow"]),
                tau_ratio=float(decoded_values["tau_ratio"]),
                alpha_decay=float(decoded_values["alpha_decay"]),
                fractional_d=float(decoded_values["fractional_d"]),
            ),
            game_theory=GameTheoryChromosome(
                ambiguity_temp=float(decoded_values["ambiguity_temp"]),
                risk_aversion=float(decoded_values["risk_aversion"]),
                predatory_intensity=float(decoded_values["predatory_intensity"]),
                logit_bull=float(np.clip(logit_bull - mean_logit, -2.0, 2.0)),
                logit_bear=float(np.clip(logit_bear - mean_logit, -2.0, 2.0)),
                logit_panic=float(np.clip(logit_panic - mean_logit, -2.0, 2.0)),
            ),
            inference=InferenceChromosome(
                execution_horizon=int(round(decoded_values["execution_horizon"])),
                hyperbolic_decay=float(decoded_values["hyperbolic_decay"]),
                profit_take_mult=float(decoded_values["profit_take_mult"]),
                stop_loss_mult=float(decoded_values["stop_loss_mult"]),
                holding_period=int(round(decoded_values["holding_period"])),
                meta_label_thresh=float(decoded_values["meta_label_thresh"]),
            ),
            risk=RiskChromosome(
                vol_target=float(decoded_values["vol_target"]),
                max_weight=float(decoded_values["max_weight"]),
                max_drawdown_limit=float(decoded_values["max_drawdown_limit"]),
                turnover_budget=float(decoded_values["turnover_budget"]),
            ),
        )

    def extract_phenotype_vector(self, chromosome: StrategyChromosome) -> np.ndarray:
        """Extract scale-normalized phenotype vector for behavioral distance calculations.

        Replaces unconstrained softmax logits with actual decoded regime probabilities
        p in Delta^3, ensuring that distance metrics measure true behavioral diversity
        rather than gauge shifts.

        Returns:
            np.ndarray of shape (20,) with normalized elements in [0.0, 1.0].
        """
        pheno = np.zeros(self.dimension, dtype=np.float64)
        priors = chromosome.game_theory.regime_priors

        for i, gene in enumerate(self.registry):
            if gene.name == "logit_bull":
                pheno[i] = priors[0]
            elif gene.name == "logit_bear":
                pheno[i] = priors[1]
            elif gene.name == "logit_panic":
                pheno[i] = priors[2]
            else:
                if gene.block == "repr":
                    val = getattr(chromosome.representation, gene.name)
                elif gene.block == "game":
                    val = getattr(chromosome.game_theory, gene.name)
                elif gene.block == "infer":
                    val = getattr(chromosome.inference, gene.name)
                else:
                    val = getattr(chromosome.risk, gene.name)

                pheno[i] = (val - gene.lower_bound) / (gene.upper_bound - gene.lower_bound)

        return np.clip(pheno, 0.0, 1.0)

    def compute_phenotypic_distance(
        self,
        c1: StrategyChromosome,
        c2: StrategyChromosome,
    ) -> float:
        """Compute normalized Root-Mean-Square Euclidean distance in phenotypic space.

        Returns:
            Scalar float d in [0.0, 1.0].
        """
        v1 = self.extract_phenotype_vector(c1)
        v2 = self.extract_phenotype_vector(c2)
        diff = v1 - v2
        dist: float = float(np.sqrt(np.mean(diff * diff)))
        return dist
