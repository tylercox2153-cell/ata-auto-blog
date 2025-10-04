import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from config.settings import USER_AGENT

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    time.sleep(1.5)  # polite delay
    return r.text

def list_result_links(index_url: str, pattern_hint: str = "result"):
    html = fetch(index_url)
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        text = (a.get_text() or "").strip()
        if pattern_hint.lower() in href.lower() or pattern_hint.lower() in text.lower():
            links.append({"text": text[:100], "url": urljoin(index_url, href)})
    return links

def main():
    example_index = "https://shootata.com/"  # replace with approved page later
    out = list_result_links(example_index, pattern_hint="result")
    print(f"[demo] potential result links from: {example_index}")
    for i, lk in enumerate(out[:20], 1):
        print(f"{i:02d}. {lk['text']} -> {lk['url']}")

if __name__ == "__main__":
    main()
