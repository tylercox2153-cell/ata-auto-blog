import time, hashlib
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from config.settings import USER_AGENT, TIMEOUT, REQUEST_DELAY_SECONDS, CACHE_DIR

HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return Path(CACHE_DIR) / f"{h}.html"

def read_cache(url: str) -> str | None:
    p = _cache_path(url)
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else None

def write_cache(url: str, text: str):
    _cache_path(url).write_text(text, encoding="utf-8")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_html(url: str, use_cache=True) -> str:
    if use_cache:
        cached = read_cache(url)
        if cached:
            return cached
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    text = resp.text
    write_cache(url, text)
    time.sleep(REQUEST_DELAY_SECONDS)
    return text
