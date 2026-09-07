"""SQLAlchemy implementation of Genotype chromosome repository."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from quant.domain.interfaces import IGenotypeRepository
from quant.domain.models import Genotype, GenotypeCohort
from quant.infrastructure.database.models import DBGenotype


class SqlAlchemyGenotypeRepository(IGenotypeRepository):
    """Asynchronous repository for evolutionary strategy chromosomes."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _to_domain(self, row: DBGenotype) -> Genotype:
        return Genotype(
            id=UUID(row.id),
            generation=row.generation,
            cohort=GenotypeCohort(row.cohort),
            chromosome_repr=row.chromosome_repr,
            chromosome_game=row.chromosome_game,
            chromosome_infer=row.chromosome_infer,
            chromosome_risk=row.chromosome_risk,
            fitness_score=row.fitness_score,
            deflated_sharpe=row.deflated_sharpe,
            max_drawdown=row.max_drawdown,
            regret_score=row.regret_score,
            novelty_score=row.novelty_score,
            created_at=row.created_at,
        )

    async def add(self, genotype: Genotype) -> Genotype:
        db_genotype = DBGenotype(
            id=str(genotype.id),
            generation=genotype.generation,
            cohort=genotype.cohort.value,
            chromosome_repr=genotype.chromosome_repr,
            chromosome_game=genotype.chromosome_game,
            chromosome_infer=genotype.chromosome_infer,
            chromosome_risk=genotype.chromosome_risk,
            fitness_score=genotype.fitness_score,
            deflated_sharpe=genotype.deflated_sharpe,
            max_drawdown=genotype.max_drawdown,
            regret_score=genotype.regret_score,
            novelty_score=genotype.novelty_score,
            created_at=genotype.created_at,
        )
        self.session.add(db_genotype)
        await self.session.flush()
        return genotype

    async def get_by_id(self, genotype_id: UUID) -> Genotype | None:
        query = select(DBGenotype).where(DBGenotype.id == str(genotype_id))
        result = await self.session.execute(query)
        row = result.scalar_one_or_none()
        if not row:
            return None
        return self._to_domain(row)

    async def get_generation(self, generation: int) -> list[Genotype]:
        query = select(DBGenotype).where(DBGenotype.generation == generation)
        result = await self.session.execute(query)
        rows = result.scalars().all()
        return [self._to_domain(r) for r in rows]

    async def get_alpha_cohort(self, limit: int = 50) -> list[Genotype]:
        query = (
            select(DBGenotype)
            .where(DBGenotype.cohort == GenotypeCohort.ALPHA.value)
            .order_by(DBGenotype.fitness_score.desc().nullslast())
            .limit(limit)
        )
        result = await self.session.execute(query)
        rows = result.scalars().all()
        return [self._to_domain(r) for r in rows]

    async def update_evaluation(
        self,
        genotype_id: UUID,
        fitness_score: float,
        deflated_sharpe: float,
        max_drawdown: float,
        regret_score: float,
    ) -> None:
        stmt = (
            update(DBGenotype)
            .where(DBGenotype.id == str(genotype_id))
            .values(
                fitness_score=fitness_score,
                deflated_sharpe=deflated_sharpe,
                max_drawdown=max_drawdown,
                regret_score=regret_score,
            )
        )
        await self.session.execute(stmt)
        await self.session.flush()
