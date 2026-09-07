"""Tests for NSGA-II non-dominated Pareto sorting, crowding distance, and novelty search."""

from uuid import uuid4

from quant.domain.models import Genotype, GenotypeCohort
from quant.services.genotype_service import (
    GenotypeService,
    calculate_crowding_distance,
    fast_non_dominated_sort,
)


def test_fast_non_dominated_sort_dominance() -> None:
    # 4 individuals with 2 objectives (both maximized)
    # Individual 0: [10, 10] (dominates everyone)
    # Individual 1: [8, 5]
    # Individual 2: [5, 8]
    # Individual 3: [2, 2] (dominated by everyone)
    objectives = [
        [10.0, 10.0],
        [8.0, 5.0],
        [5.0, 8.0],
        [2.0, 2.0],
    ]

    fronts = fast_non_dominated_sort(objectives)
    assert len(fronts) == 3
    assert fronts[0] == [0]  # Non-dominated elite front
    assert set(fronts[1]) == {1, 2}  # Second trade-off front
    assert fronts[2] == [3]  # Dominated front


def test_calculate_crowding_distance_boundary_infinite() -> None:
    objectives = [
        [1.0, 5.0],
        [3.0, 3.0],
        [5.0, 1.0],
    ]
    front = [0, 1, 2]
    distances = calculate_crowding_distance(front, objectives)

    # Boundary points should have infinite distance to prioritize diversity
    assert distances[0] == float("inf")
    assert distances[2] == float("inf")
    assert distances[1] > 0.0


def test_novelty_aspirant_selection() -> None:
    # Mock GenotypeService
    service = GenotypeService(genotype_repo=None)  # type: ignore[arg-type]

    elite = [
        Genotype(
            id=uuid4(),
            generation=0,
            cohort=GenotypeCohort.ALPHA,
            chromosome_repr={"tau_fast": 3600.0, "tau_slow": 86400.0},
            chromosome_game={},
            chromosome_infer={},
            chromosome_risk={},
        )
    ]

    # Aspirant A has parameters identical to elite (low novelty)
    # Aspirant B has distant parameters (high novelty)
    asp_a = Genotype(
        id=uuid4(),
        generation=0,
        cohort=GenotypeCohort.ASPIRANT,
        chromosome_repr={"tau_fast": 3650.0, "tau_slow": 86500.0},
        chromosome_game={},
        chromosome_infer={},
        chromosome_risk={},
    )
    asp_b = Genotype(
        id=uuid4(),
        generation=0,
        cohort=GenotypeCohort.ASPIRANT,
        chromosome_repr={"tau_fast": 14400.0, "tau_slow": 345600.0},
        chromosome_game={},
        chromosome_infer={},
        chromosome_risk={},
    )

    selected = service.select_aspirant_by_novelty([asp_a, asp_b], elite)
    assert selected is not None
    assert selected.id == asp_b.id
