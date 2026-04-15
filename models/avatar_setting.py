import uuid
from datetime import datetime

from sqlalchemy import (
    Text,
    Boolean,
    DateTime,
    func,
    text,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from db import Base

class AvatarSettings(Base):
    __tablename__ = "avatar_settings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    profile_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'default'::text"),
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )

    tone: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'neutral'::text"),
    )

    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)

    safety_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("''::text"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),  # app-side update timestamp (no trigger)
    )

    __table_args__ = (
        # Matches: avatar_settings_org_idx
        Index("avatar_settings_org_idx", "org_id"),

        # Matches: UNIQUE (org_id, profile_name)
        Index(
            "avatar_settings_org_id_profile_name_key",
            "org_id",
            "profile_name",
            unique=True,
        ),

        # Matches: UNIQUE (org_id) WHERE is_active = true
        Index(
            "avatar_settings_one_active_per_org",
            "org_id",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )
