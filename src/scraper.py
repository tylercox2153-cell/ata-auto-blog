import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from config.settings import BASE_URL, USER_AGENT

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch(url: str) -> str:
    """Fetch a single page politely (demo only)."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    time.sleep(2)  # polite delay
    return resp.text

def parse_home(html: str):
    """Demo parser: first 20 links from homepage (we’ll replace later)."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        text = (a.get_text() or "").strip()
        href = a["href"].strip()
        if text and href and not href.startswith("#"):
            links.append({"text": text[:100], "url": href})
    return links[:20]

def main():
    print(f"[info] BASE_URL={BASE_URL}")
    html = fetch(BASE_URL)
    links = parse_home(html)
    print("[demo] Found links:")
    for i, lk in enumerate(links, 1):
        print(f"{i:02d}. {lk['text']} -> {urljoin(BASE_URL, lk['url'])}")

if __name__ == "__main__":
    main()
