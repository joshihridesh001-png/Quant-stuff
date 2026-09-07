"""Declarative SQLAlchemy ORM models matching domain entity specifications."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from quant.infrastructure.database.session import Base


def utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(UTC)


class DBAsset(Base):
    """Relational table for traded assets in target universe."""

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    sector: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    centralities = relationship(
        "DBEventCentrality", back_populates="asset", cascade="all, delete-orphan"
    )


class DBEvent(Base):
    """Relational table storing ingested news events and extracted vectors."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    headline: Mapped[str] = mapped_column(String(500), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    dense_embedding: Mapped[list[float]] = mapped_column(JSON, default=list, nullable=False)
    sentiment_polarity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sentiment_subjectivity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sentiment_novelty: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    urgency: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    source: Mapped[str] = mapped_column(String(50), default="GENERIC", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    centralities = relationship(
        "DBEventCentrality", back_populates="event", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_events_timestamp_urgency", "timestamp", "urgency"),)


class DBEventCentrality(Base):
    """Junction table connecting news events to universe assets with relevance weights."""

    __tablename__ = "event_centralities"

    event_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("events.id", ondelete="CASCADE"), primary_key=True
    )
    asset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assets.id", ondelete="CASCADE"), primary_key=True
    )
    centrality: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    event = relationship("DBEvent", back_populates="centralities")
    asset = relationship("DBAsset", back_populates="centralities")


class DBGenotype(Base):
    """Relational table storing candidate strategy chromosomes and fitness metrics."""

    __tablename__ = "genotypes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    generation: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    cohort: Mapped[str] = mapped_column(String(20), nullable=False)  # ALPHA, ASPIRANT
    chromosome_repr: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    chromosome_game: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    chromosome_infer: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    chromosome_risk: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    fitness_score: Mapped[float] = mapped_column(Float, nullable=True)
    deflated_sharpe: Mapped[float] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=True)
    regret_score: Mapped[float] = mapped_column(Float, nullable=True)
    novelty_score: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (Index("ix_genotypes_cohort_fitness", "cohort", "fitness_score"),)
