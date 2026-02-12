import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import Header, HTTPException, Depends
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db

pwd = CryptContext(schemes=["argon2"], deprecated="auto")


def generate_full_key() -> tuple[str, str]:
    prefix = "ak_" + secrets.token_urlsafe(8).replace("-", "").replace("_", "")
    secret = secrets.token_urlsafe(32)
    full_key = f"{prefix}.{secret}"
    return prefix, full_key


def create_api_key(db: Session, org_id: str, name: Optional[str] = None, scopes=None) -> str:
    scopes = scopes or ["chat", "upload"]

    prefix, full_key = generate_full_key()
    key_hash = pwd.hash(full_key)

    db.execute(
        text("""
            INSERT INTO api_keys (org_id, key_prefix, key_hash, name, scopes, status)
            VALUES (:org_id, :key_prefix, :key_hash, :name, :scopes, 'active')
        """),
        {
            "org_id": org_id,
            "key_prefix": prefix,
            "key_hash": key_hash,
            "name": name,
            "scopes": scopes,
        },
    )
    db.commit()

    return full_key  # show once


def require_tenant(
    x_api_key: str = Header(..., alias="X-API-KEY"),
    db: Session = Depends(get_db),
) -> dict:
    # basic format check
    if "." not in x_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key format")

    prefix = x_api_key.split(".", 1)[0]

    row = db.execute(
        text("""
            SELECT k.id, k.org_id, k.key_hash, k.scopes, k.status, o.status AS org_status
            FROM api_keys k
            JOIN organizations o ON o.id = k.org_id
            WHERE k.key_prefix = :prefix
            LIMIT 1
        """),
        {"prefix": prefix},
    ).mappings().first()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if row["status"] != "active":
        raise HTTPException(status_code=401, detail="API key revoked")

    if row["org_status"] != "active":
        raise HTTPException(status_code=403, detail="Organization not active")

    if not pwd.verify(x_api_key, row["key_hash"]):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # update last_used_at (optional but useful)
    db.execute(
        text("UPDATE api_keys SET last_used_at = :t WHERE id = :id"),
        {"t": datetime.now(timezone.utc), "id": row["id"]},
    )
    db.commit()

    return {"org_id": row["org_id"], "scopes": row["scopes"]}
