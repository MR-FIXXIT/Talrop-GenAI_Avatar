from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional

from db import get_db
from auth import require_tenant
from models.avatar_setting import AvatarSettings

router = APIRouter(prefix="/settings", tags=["settings"])

class AvatarSettingsPayload(BaseModel):
    tone: Optional[str] = Field("neutral", description="Behavioral tone of the responder")
    system_prompt: str = Field(..., description="The main behavior instructions for the LLM")
    safety_prompt: Optional[str] = Field("", description="Safety constraints added to prompts")
    profile_name: Optional[str] = Field("default", description="Profile configuration name")
    is_active: Optional[bool] = Field(True, description="Sets this profile as active")


@router.get("/avatar")
def get_avatar_settings(
    ctx: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Retrieve the active avatar/personality settings for the authenticated organization.
    """
    org_id = str(ctx["org_id"])
    
    # We query the active profile for the organization
    settings = db.query(AvatarSettings).filter(
        AvatarSettings.org_id == org_id,
        AvatarSettings.is_active == True
    ).first()
    
    if not settings:
        raise HTTPException(status_code=404, detail="No active avatar settings found for this organization.")
        
    return {
        "id": str(settings.id),
        "org_id": str(settings.org_id),
        "profile_name": settings.profile_name,
        "is_active": settings.is_active,
        "tone": settings.tone,
        "system_prompt": settings.system_prompt,
        "safety_prompt": settings.safety_prompt,
        "created_at": settings.created_at,
        "updated_at": settings.updated_at,
    }


@router.put("/avatar")
def upsert_avatar_settings(
    payload: AvatarSettingsPayload,
    ctx: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Upsert the avatar semantics for the organization. Overwrites existing profile 
    with the same name or creates a new one. Ensures only one profile is active at a time.
    """
    org_id = str(ctx["org_id"])
    
    # If this profile is going to be marked as active, we must deactivate the others 
    # to satisfy the postgresql_where unique index "avatar_settings_one_active_per_org".
    if payload.is_active:
        db.execute(
            text("UPDATE avatar_settings SET is_active = false WHERE org_id = CAST(:org_id AS uuid)"),
            {"org_id": org_id}
        )
    
    # Check if a profile with this exact name already exists for the org
    existing_profile = db.query(AvatarSettings).filter(
        AvatarSettings.org_id == org_id,
        AvatarSettings.profile_name == payload.profile_name
    ).first()
    
    if existing_profile:
        # Update existing properties
        existing_profile.is_active = payload.is_active
        existing_profile.tone = payload.tone
        existing_profile.system_prompt = payload.system_prompt
        existing_profile.safety_prompt = payload.safety_prompt
        
        db.commit()
        db.refresh(existing_profile)
        target = existing_profile
    else:
        # Create new configuration profile
        new_profile = AvatarSettings(
            org_id=org_id,
            profile_name=payload.profile_name,
            is_active=payload.is_active,
            tone=payload.tone,
            system_prompt=payload.system_prompt,
            safety_prompt=payload.safety_prompt,
        )
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        target = new_profile
        
    return {
        "message": "Avatar settings upserted successfully",
        "id": str(target.id),
        "profile_name": target.profile_name,
        "is_active": target.is_active
    }
