from pydantic import BaseModel

class OrgCreate(BaseModel):
    name: str
    slug: str | None = None  # if not provided, derive from name
