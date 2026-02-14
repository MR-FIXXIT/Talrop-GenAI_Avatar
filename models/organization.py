import uuid
from datetime import datetime

from sqlalchemy import Text, DateTime, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base

class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)

    slug: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,  # matches organizations_slug_key
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'active'::text"),
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
        onupdate=func.now(),  # no trigger required
    )

    api_keys = relationship(
        "ApiKey",
        back_populates="organization",
        cascade="all, delete-orphan",
    )


    api_keys = relationship("ApiKey", back_populates="organization", cascade="all, delete-orphan")
