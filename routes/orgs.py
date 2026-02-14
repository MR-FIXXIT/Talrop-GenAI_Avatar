from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from uuid import UUID

from db import get_db
from models import Organization
from schemas.orgs import OrgCreate
from utils.slug import slugify

router = APIRouter(prefix="/orgs", tags=["orgs"])

@router.get("")
def list_orgs(db: Session = Depends(get_db)):
    items = db.query(Organization).all()
    return {
        "count": len(items),
        "items": [
            {"id": str(o.id), "name": o.name, "slug": o.slug, "status": o.status}
            for o in items
        ],
    }

@router.post("/create")
def create_org(payload: OrgCreate, db: Session = Depends(get_db)):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    slug = (payload.slug or slugify(name)).strip()
    if not slug:
        raise HTTPException(status_code=400, detail="slug is required")

    org = Organization(name=name, slug=slug)

    try:
        db.add(org)
        db.commit()
        db.refresh(org)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Organization slug already exists")

    return {
        "id": str(org.id),
        "name": org.name,
        "slug": org.slug,
        "status": org.status,
        "created_at": org.created_at,
        "updated_at": org.updated_at,
    }