import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import Text, DateTime, func, text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base

class ApiKey(Base):
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,  # matches api_keys_org_idx
    )

    key_prefix: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,  # matches unique constraint
    )

    key_hash: Mapped[str] = mapped_column(Text, nullable=False)

    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    scopes: Mapped[List[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("ARRAY['chat'::text, 'upload'::text]"),
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'active'::text"),
        index=True,  # matches api_keys_status_idx
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Optional (only if you want ORM navigation):
    organization = relationship("Organization", back_populates="api_keys")

    __table_args__ = (
        # These are already covered by unique=True and index=True above,
        # but keeping explicit Index definitions is fine if you prefer:
        Index("api_keys_org_idx", "org_id"),
        Index("api_keys_status_idx", "status"),
    )
