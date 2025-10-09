from bs4 import BeautifulSoup

def parse_ata_event(html: str, url: str):
    soup = BeautifulSoup(html, "html.parser")
    name = None
    if soup.title and soup.title.text.strip():
        name = soup.title.text.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        name = h1.get_text(strip=True)

    event = {"url": url, "name": name, "club": None, "state": None, "start_date": None, "end_date": None}
    rows = []
    # TODO: tune selectors once you choose the specific ATA page template(s) you’ll use
    return {"event": event, "rows": rows}
