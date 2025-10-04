import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from config.settings import USER_AGENT

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _fetch_title(url: str) -> str:
    """Fetch a page and return a clean title or a fallback."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title = ""
    if soup.title and soup.title.text.strip():
        title = soup.title.text.strip()
    else:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        title = urlparse(url).path.rsplit("/", 1)[-1] or url

    return " ".join(title.split())

def get_shoots_from_urls(urls):
    """Return [{'title': str, 'url': str}] for the given ShootScoreboard URLs."""
    items = []
    for u in urls:
        try:
            t = _fetch_title(u)
        except Exception as e:
            print(f"[warn] could not fetch {u}: {e}")
            t = "(Shoot title)"
        items.append({"title": t, "url": u})
        time.sleep(1.5)  # polite delay between requests
    return items
