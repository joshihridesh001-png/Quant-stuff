"""Integration tests for SQLAlchemy repositories against an async database."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from quant.domain.models import Asset, EventCentrality, Genotype, GenotypeCohort, NewsEvent
from quant.infrastructure.repositories.asset_repository import SqlAlchemyAssetRepository
from quant.infrastructure.repositories.event_repository import SqlAlchemyEventRepository
from quant.infrastructure.repositories.genotype_repository import SqlAlchemyGenotypeRepository


@pytest.mark.asyncio
async def test_asset_repository_crud(db_session: AsyncSession) -> None:
    repo = SqlAlchemyAssetRepository(db_session)

    # 1. Add Asset
    asset = Asset(ticker="NVDA", name="NVIDIA Corporation", sector="Semiconductors")
    created = await repo.add(asset)
    assert created.ticker == "NVDA"

    # 2. Query Asset by Ticker
    fetched = await repo.get_by_ticker("NVDA")
    assert fetched is not None
    assert fetched.ticker == "NVDA"
    assert fetched.sector == "Semiconductors"

    # 3. List Active Assets
    all_active = await repo.list_active()
    assert len(all_active) == 1
    assert all_active[0].ticker == "NVDA"


@pytest.mark.asyncio
async def test_event_repository_crud(db_session: AsyncSession) -> None:
    asset_repo = SqlAlchemyAssetRepository(db_session)
    event_repo = SqlAlchemyEventRepository(db_session)

    asset = await asset_repo.add(Asset(ticker="AAPL", name="Apple Inc.", sector="Technology"))

    # 1. Add Event
    event = NewsEvent(
        headline="Apple unveils new M-series neural processing architecture",
        raw_text="Full article body discussing breakthrough low-power tensor acceleration...",
        timestamp=datetime.now(UTC),
        sentiment_polarity=0.85,
        sentiment_subjectivity=0.20,
        sentiment_novelty=0.90,
        urgency=0.75,
        source="REUTERS",
    )
    centrality = EventCentrality(event_id=event.id, asset_id=asset.id, centrality=0.95)

    created_event = await event_repo.add(event, [centrality])
    assert created_event.id == event.id

    # 2. Retrieve Event by ID
    fetched_event = await event_repo.get_by_id(event.id)
    assert fetched_event is not None
    assert fetched_event.headline == event.headline
    assert fetched_event.sentiment_polarity == 0.85

    # 3. Retrieve Windowed Events for Asset
    start_time = datetime(2020, 1, 1, tzinfo=UTC)
    end_time = datetime(2030, 1, 1, tzinfo=UTC)
    windowed = await event_repo.get_events_for_asset(asset.id, start_time, end_time)

    assert len(windowed) == 1
    ev, weight = windowed[0]
    assert ev.id == event.id
    assert abs(weight - 0.95) < 1e-6


@pytest.mark.asyncio
async def test_genotype_repository_crud(db_session: AsyncSession) -> None:
    repo = SqlAlchemyGenotypeRepository(db_session)

    # 1. Add Genotypes
    alpha_genotype = Genotype(
        generation=1,
        cohort=GenotypeCohort.ALPHA,
        chromosome_repr={"tau_fast": 3600.0, "tau_slow": 86400.0},
        chromosome_game={"lambda": 1.5},
        chromosome_infer={"threshold": 0.6},
        chromosome_risk={"vol_target": 0.12},
    )
    aspirant_genotype = Genotype(
        generation=1,
        cohort=GenotypeCohort.ASPIRANT,
        chromosome_repr={"tau_fast": 7200.0, "tau_slow": 43200.0},
        chromosome_game={"lambda": 2.0},
        chromosome_infer={"threshold": 0.4},
        chromosome_risk={"vol_target": 0.15},
    )

    await repo.add(alpha_genotype)
    await repo.add(aspirant_genotype)

    # 2. Update Fitness for Alpha
    await repo.update_evaluation(
        genotype_id=alpha_genotype.id,
        fitness_score=2.85,
        deflated_sharpe=2.10,
        max_drawdown=0.06,
        regret_score=0.90,
    )

    # 3. Fetch Genotype by ID
    fetched = await repo.get_by_id(alpha_genotype.id)
    assert fetched is not None
    assert fetched.fitness_score == 2.85
    assert fetched.deflated_sharpe == 2.10

    # 4. Fetch Alpha Cohort
    alphas = await repo.get_alpha_cohort(limit=10)
    assert len(alphas) == 1
    assert alphas[0].id == alpha_genotype.id
    assert alphas[0].cohort == GenotypeCohort.ALPHA
