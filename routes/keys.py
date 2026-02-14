import os
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db
from auth import create_api_key

router = APIRouter(prefix="/keys", tags=["keys"])

ADMIN_BOOTSTRAP_TOKEN = os.environ.get("ADMIN_BOOTSTRAP_TOKEN")

@router.post("/create")
def keys_create(
    org_id: str,
    name: str | None = None,
    admin_token: str | None = Header(None, alias="X-ADMIN-TOKEN"),
    db: Session = Depends(get_db),
):
    if not ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_BOOTSTRAP_TOKEN not set")

    if admin_token != ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    exists = db.execute(
        text("SELECT 1 FROM organizations WHERE id = CAST(:id AS uuid) LIMIT 1"),
        {"id": org_id},
    ).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Organization not found")

    full_key = create_api_key(db, org_id=org_id, name=name)
    return {"api_key": full_key}
