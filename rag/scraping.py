# rag/scraping.py
from __future__ import annotations

import ipaddress
import re
import socket
import tempfile
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers


# ---- Tunables (static-only) ----
DEFAULT_TIMEOUT_S = 15.0
MAX_BYTES = 2_000_000          # 2 MB cap per page
MAX_REDIRECTS = 5
MIN_TEXT_CHARS = 600           # below this => treat as dynamic/blocked/empty
ALLOWED_PORTS = {80, 443, None}


def _normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_text_from_html(html: str) -> tuple[str, str]:
    """
    Returns (title, text).
    Static-only extraction: strips scripts/styles, prefers <main>/<article>.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.get_text():
        title = _normalize_text(soup.title.get_text())

    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text(separator="\n")
    text = _normalize_text(text)

    return title, text


def _is_public_ip(ip: str) -> bool:
    try:
        obj = ipaddress.ip_address(ip)
    except ValueError:
        return False

    # Block anything that is not globally routable
    if (
        obj.is_private
        or obj.is_loopback
        or obj.is_link_local
        or obj.is_multicast
        or obj.is_reserved
        or obj.is_unspecified
    ):
        return False
    return True


def _validate_url_ssrf(url: str) -> None:
    """
    Basic SSRF defense:
    - only http/https
    - only allowed ports
    - host must resolve only to public IPs
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http/https URLs are allowed")

    if parsed.port not in ALLOWED_PORTS:
        raise HTTPException(status_code=400, detail="Only ports 80/443 are allowed")

    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail="Invalid URL host")

    # If host is already an IP literal, check it directly
    try:
        ipaddress.ip_address(host)
        if not _is_public_ip(host):
            raise HTTPException(status_code=400, detail="Blocked host (non-public IP)")
        return
    except ValueError:
        pass

    # DNS resolve and validate each A/AAAA
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Could not resolve host")

    resolved_any = False
    for family, _, _, _, sockaddr in infos:
        ip = sockaddr[0]
        resolved_any = True
        if not _is_public_ip(ip):
            raise HTTPException(status_code=400, detail="Blocked host (resolves to non-public IP)")

    if not resolved_any:
        raise HTTPException(status_code=400, detail="Could not resolve host to any IP")


def _safe_filename_from_url(url: str, fallback: str = "page.txt") -> str:
    p = urlparse(url)
    host = (p.hostname or "site").lower()
    path = (p.path or "/").strip("/")
    if not path:
        name = host
    else:
        # keep it filesystem-friendly
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path)[:80].strip("_")
        name = f"{host}_{slug}" if slug else host
    return f"{name}.txt" if name else fallback


@dataclass
class ScrapedPage:
    url: str
    final_url: str
    title: str
    text: str
    filename: str


def _fetch_html_static(url: str, *, timeout_s: float) -> tuple[str, str]:
    """
    Fetch HTML with manual redirect handling so each hop can be SSRF-validated.
    Returns (final_url, html).
    """
    _validate_url_ssrf(url)

    headers = {
        "User-Agent": "Talrop-GenBot/1.0 (admin@talrop.com)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }

    with httpx.Client(timeout=timeout_s, headers=headers, follow_redirects=False) as client:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            r = client.get(current)

            # Handle redirects manually
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location")
                if not loc:
                    raise HTTPException(status_code=400, detail="Redirect without Location header")
                next_url = urljoin(current, loc)
                _validate_url_ssrf(next_url)
                current = next_url
                continue

            if r.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"HTTP error {r.status_code}")

            ct = (r.headers.get("content-type") or "").lower()
            if "text/html" not in ct and "application/xhtml+xml" not in ct:
                raise HTTPException(status_code=400, detail="Non-HTML content type")

            content = r.content
            if content is None:
                raise HTTPException(status_code=400, detail="Empty response")

            if len(content) > MAX_BYTES:
                raise HTTPException(status_code=400, detail="Page too large")

            # Decode with best effort
            encoding = r.encoding or "utf-8"
            html = content.decode(encoding, errors="ignore")
            return current, html

    raise HTTPException(status_code=400, detail="Too many redirects")


def scrape_static_url(url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S, min_text_chars: int = MIN_TEXT_CHARS) -> ScrapedPage:
    final_url, html = _fetch_html_static(url, timeout_s=timeout_s)
    title, text = _extract_text_from_html(html)

    if len(text) < min_text_chars:
        raise HTTPException(
            status_code=400,
            detail="dynamic_or_empty_content (static-only scraper: extracted text too short)",
        )

    filename = _safe_filename_from_url(final_url)
    return ScrapedPage(
        url=url,
        final_url=final_url,
        title=title,
        text=text,
        filename=filename,
    )


def scraped_page_to_uploadfile(page: ScrapedPage) -> UploadFile:
    """
    Wrap scraped text as an UploadFile so it can be passed to ingest_and_index_text_file unchanged.
    """
    data = page.text.encode("utf-8", errors="ignore")

    spooled = tempfile.SpooledTemporaryFile(max_size=512 * 1024, mode="w+b")
    spooled.write(data)
    spooled.seek(0)

    headers = Headers({"content-type": "text/plain; charset=utf-8"})
    return UploadFile(spooled, filename=page.filename, size=len(data), headers=headers)
