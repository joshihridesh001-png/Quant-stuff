"""SQLAlchemy implementation of Asset repository."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from quant.domain.interfaces import IAssetRepository
from quant.domain.models import Asset
from quant.infrastructure.database.models import DBAsset


class SqlAlchemyAssetRepository(IAssetRepository):
    """Asynchronous repository for Universe Asset entities."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, asset: Asset) -> Asset:
        db_asset = DBAsset(
            id=str(asset.id),
            ticker=asset.ticker.upper(),
            name=asset.name,
            sector=asset.sector,
            is_active=asset.is_active,
        )
        self.session.add(db_asset)
        await self.session.flush()
        return asset

    async def get_by_ticker(self, ticker: str) -> Asset | None:
        query = select(DBAsset).where(DBAsset.ticker == ticker.upper())
        result = await self.session.execute(query)
        db_asset = result.scalar_one_or_none()
        if not db_asset:
            return None
        return Asset(
            id=UUID(db_asset.id),
            ticker=db_asset.ticker,
            name=db_asset.name,
            sector=db_asset.sector,
            is_active=db_asset.is_active,
        )

    async def list_active(self) -> list[Asset]:
        query = select(DBAsset).where(DBAsset.is_active.is_(True))
        result = await self.session.execute(query)
        db_assets = result.scalars().all()
        return [
            Asset(
                id=UUID(row.id),
                ticker=row.ticker,
                name=row.name,
                sector=row.sector,
                is_active=row.is_active,
            )
            for row in db_assets
        ]
