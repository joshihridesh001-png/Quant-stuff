"""SQLAlchemy implementation of News Event and Causal Chaining repository."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from quant.domain.interfaces import IEventRepository
from quant.domain.models import EventCentrality, NewsEvent
from quant.infrastructure.database.models import DBEvent, DBEventCentrality


class SqlAlchemyEventRepository(IEventRepository):
    """Asynchronous repository handling news events and asset centrality mappings."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, event: NewsEvent, centralities: list[EventCentrality]) -> NewsEvent:
        db_event = DBEvent(
            id=str(event.id),
            timestamp=event.timestamp,
            headline=event.headline,
            raw_text=event.raw_text,
            dense_embedding=event.dense_embedding,
            sentiment_polarity=event.sentiment_polarity,
            sentiment_subjectivity=event.sentiment_subjectivity,
            sentiment_novelty=event.sentiment_novelty,
            urgency=event.urgency,
            source=event.source,
            created_at=event.created_at,
        )
        self.session.add(db_event)

        for c in centralities:
            db_centrality = DBEventCentrality(
                event_id=str(c.event_id),
                asset_id=str(c.asset_id),
                centrality=c.centrality,
            )
            self.session.add(db_centrality)

        await self.session.flush()
        return event

    async def add_batch(
        self, batch: list[tuple[NewsEvent, list[EventCentrality]]]
    ) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        for event, centralities in batch:
            db_event = DBEvent(
                id=str(event.id),
                timestamp=event.timestamp,
                headline=event.headline,
                raw_text=event.raw_text,
                dense_embedding=event.dense_embedding,
                sentiment_polarity=event.sentiment_polarity,
                sentiment_subjectivity=event.sentiment_subjectivity,
                sentiment_novelty=event.sentiment_novelty,
                urgency=event.urgency,
                source=event.source,
                created_at=event.created_at,
            )
            self.session.add(db_event)
            for c in centralities:
                db_centrality = DBEventCentrality(
                    event_id=str(c.event_id),
                    asset_id=str(c.asset_id),
                    centrality=c.centrality,
                )
                self.session.add(db_centrality)
            events.append(event)

        await self.session.flush()
        return events

    async def get_by_id(self, event_id: UUID) -> NewsEvent | None:
        query = select(DBEvent).where(DBEvent.id == str(event_id))
        result = await self.session.execute(query)
        db_event = result.scalar_one_or_none()
        if not db_event:
            return None

        return NewsEvent(
            id=UUID(db_event.id),
            headline=db_event.headline,
            raw_text=db_event.raw_text,
            timestamp=db_event.timestamp,
            dense_embedding=db_event.dense_embedding,
            sentiment_polarity=db_event.sentiment_polarity,
            sentiment_subjectivity=db_event.sentiment_subjectivity,
            sentiment_novelty=db_event.sentiment_novelty,
            urgency=db_event.urgency,
            source=db_event.source,
            created_at=db_event.created_at,
        )

    async def get_events_for_asset(
        self, asset_id: UUID, start_time: datetime, end_time: datetime
    ) -> list[tuple[NewsEvent, float]]:
        query = (
            select(DBEvent, DBEventCentrality.centrality)
            .join(DBEventCentrality, DBEvent.id == DBEventCentrality.event_id)
            .where(
                DBEventCentrality.asset_id == str(asset_id),
                DBEvent.timestamp >= start_time,
                DBEvent.timestamp <= end_time,
            )
            .order_by(DBEvent.timestamp.desc())
        )
        result = await self.session.execute(query)
        rows = result.all()

        events_with_weights: list[tuple[NewsEvent, float]] = []
        for db_event, weight in rows:
            event = NewsEvent(
                id=UUID(db_event.id),
                headline=db_event.headline,
                raw_text=db_event.raw_text,
                timestamp=db_event.timestamp,
                dense_embedding=db_event.dense_embedding,
                sentiment_polarity=db_event.sentiment_polarity,
                sentiment_subjectivity=db_event.sentiment_subjectivity,
                sentiment_novelty=db_event.sentiment_novelty,
                urgency=db_event.urgency,
                source=db_event.source,
                created_at=db_event.created_at,
            )
            events_with_weights.append((event, float(weight)))

        return events_with_weights
