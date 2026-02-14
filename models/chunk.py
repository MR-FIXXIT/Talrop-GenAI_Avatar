# models.py (or chunks.py)
import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import (
    Text,
    Integer,
    DateTime,
    func,
    text,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db import Base

class Chunk(Base):
    __tablename__ = "chunks"

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

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    metadata_: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",                          # keep DB column name
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        # btree indexes shown by \d
        Index("chunks_org_idx", "org_id"),
        Index("chunks_doc_idx", "document_id"),

        # UNIQUE (document_id, chunk_index)
        Index(
            "chunks_document_id_chunk_index_key",
            "document_id",
            "chunk_index",
            unique=True,
        ),

        # GIN full-text index: to_tsvector('english', content)
        Index(
            "chunks_fts_idx",
            text("to_tsvector('english'::regconfig, content)"),
            postgresql_using="gin",
        ),
    )