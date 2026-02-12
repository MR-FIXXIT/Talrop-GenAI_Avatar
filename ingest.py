def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """
    Simple splitter:
    - takes chunk_size characters
    - overlaps by overlap characters
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)

    return chunks
    