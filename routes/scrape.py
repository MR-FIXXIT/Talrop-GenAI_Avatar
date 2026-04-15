# routes/scrape.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from auth import require_tenant
from db import get_db
from rag.ingestion import ingest_and_index
from rag.scraping import scrape_static_url, scraped_page_to_uploadfile

router = APIRouter(tags=["documents"])

MAX_URLS_PER_REQUEST = 20
DELAY_BETWEEN_URLS_S = 0.5  # polite delay


def require_scope(ctx: dict, needed: str):
    scopes = ctx.get("scopes") or []
    if needed not in scopes:
        raise HTTPException(status_code=403, detail=f"Missing scope: {needed}")


class ScrapeRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., min_length=1, max_length=MAX_URLS_PER_REQUEST)
    # Optional: override static-only threshold
    min_text_chars: Optional[int] = Field(default=None, ge=100, le=20000)


@router.post("/scrape")
async def scrape_and_ingest(
    payload: ScrapeRequest,
    ctx=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    require_scope(ctx, "upload")  # or define a separate "scrape" scope
    org_id = str(ctx["org_id"])

    results = []

    for u in payload.urls:
        url = str(u)
        try:
            # Scrape (static-only) in threadpool to avoid blocking the event loop
            page = await run_in_threadpool(
                scrape_static_url,
                url,
                # keep defaults; allow overriding min_text_chars
                min_text_chars=(payload.min_text_chars or None) or 600,
            )

            upload_file = await run_in_threadpool(scraped_page_to_uploadfile, page)

            # Ingest using your unified ingestion pipeline
            ingest_result = await run_in_threadpool(
                ingest_and_index,
                org_id=org_id,
                file=upload_file,
            )

            results.append(
                {
                    "url": url,
                    "final_url": page.final_url,
                    "title": page.title,
                    "status": "ok",
                    "document_id": ingest_result.document_id,
                }
            )

        except HTTPException as e:
            results.append(
                {
                    "url": url,
                    "status": "error",
                    "reason": e.detail,
                }
            )
        except Exception as e:
            results.append(
                {
                    "url": url,
                    "status": "error",
                    "reason": str(e),
                }
            )

        if DELAY_BETWEEN_URLS_S > 0:
            await asyncio.sleep(DELAY_BETWEEN_URLS_S)

    return {"results": results}
