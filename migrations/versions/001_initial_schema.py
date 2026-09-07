"""001 Initial Schema for Assets, Events, Centralities, and Genotypes

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-09-07 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Assets Table
    op.create_table(
        "assets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("sector", sa.String(length=50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assets_ticker", "assets", ["ticker"], unique=True)

    # 2. Events Table
    op.create_table(
        "events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("headline", sa.String(length=500), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("dense_embedding", sa.JSON(), nullable=False),
        sa.Column("sentiment_polarity", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column(
            "sentiment_subjectivity", sa.Float(), nullable=False, server_default=sa.text("0.0")
        ),
        sa.Column("sentiment_novelty", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("urgency", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column(
            "source", sa.String(length=50), nullable=False, server_default=sa.text("'GENERIC'")
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_events_timestamp", "events", ["timestamp"])
    op.create_index("ix_events_timestamp_urgency", "events", ["timestamp", "urgency"])

    # 3. Event Centralities Junction Table
    op.create_table(
        "event_centralities",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("asset_id", sa.String(length=36), nullable=False),
        sa.Column("centrality", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id", "asset_id"),
    )

    # 4. Genotypes Table
    op.create_table(
        "genotypes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("cohort", sa.String(length=20), nullable=False),
        sa.Column("chromosome_repr", sa.JSON(), nullable=False),
        sa.Column("chromosome_game", sa.JSON(), nullable=False),
        sa.Column("chromosome_infer", sa.JSON(), nullable=False),
        sa.Column("chromosome_risk", sa.JSON(), nullable=False),
        sa.Column("fitness_score", sa.Float(), nullable=True),
        sa.Column("deflated_sharpe", sa.Float(), nullable=True),
        sa.Column("max_drawdown", sa.Float(), nullable=True),
        sa.Column("regret_score", sa.Float(), nullable=True),
        sa.Column("novelty_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_genotypes_generation", "genotypes", ["generation"])
    op.create_index("ix_genotypes_cohort_fitness", "genotypes", ["cohort", "fitness_score"])


def downgrade() -> None:
    op.drop_table("genotypes")
    op.drop_table("event_centralities")
    op.drop_table("events")
    op.drop_table("assets")
