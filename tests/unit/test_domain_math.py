"""Unit tests for domain mathematical algorithms: decay kernel and fitness metrics."""

from quant.services.event_service import compute_decay_kernel
from quant.services.genotype_service import compute_multiobjective_fitness


def test_decay_kernel_at_time_zero() -> None:
    # When delta_t = 0, exp(0) = 1, (1 + 0)^-beta = 1
    # kappa(0, u) = alpha * 1 + (1 - alpha) * 1 = 1.0
    kernel_val = compute_decay_kernel(delta_t_seconds=0.0, urgency=0.5, alpha=0.6)
    assert abs(kernel_val - 1.0) < 1e-6


def test_decay_kernel_monotonic_decrease() -> None:
    k_1h = compute_decay_kernel(delta_t_seconds=3600.0, urgency=0.5)
    k_1d = compute_decay_kernel(delta_t_seconds=86400.0, urgency=0.5)
    k_7d = compute_decay_kernel(delta_t_seconds=604800.0, urgency=0.5)

    assert 1.0 > k_1h > k_1d > k_7d > 0.0


def test_decay_kernel_urgency_accelerates_decay() -> None:
    # High urgency (fast transmission) should cause faster decay than low urgency
    k_low_urgency = compute_decay_kernel(delta_t_seconds=1800.0, urgency=0.1)
    k_high_urgency = compute_decay_kernel(delta_t_seconds=1800.0, urgency=0.9)

    assert k_low_urgency > k_high_urgency


def test_multiobjective_fitness_properties() -> None:
    # Baseline fitness
    f_baseline = compute_multiobjective_fitness(
        deflated_sharpe=1.5,
        max_drawdown=0.05,
        regret_score=0.8,
        novelty_score=0.4,
    )

    # Higher DSR must increase fitness
    f_higher_dsr = compute_multiobjective_fitness(
        deflated_sharpe=2.5,
        max_drawdown=0.05,
        regret_score=0.8,
        novelty_score=0.4,
    )
    assert f_higher_dsr > f_baseline

    # Higher MaxDD must heavily penalize fitness
    f_high_drawdown = compute_multiobjective_fitness(
        deflated_sharpe=1.5,
        max_drawdown=0.40,
        regret_score=0.8,
        novelty_score=0.4,
    )
    assert f_baseline > f_high_drawdown
