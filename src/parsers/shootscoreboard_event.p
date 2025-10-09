from bs4 import BeautifulSoup

def parse_shootscoreboard_event(html: str, url: str):
    """
    Returns:
    {
      "event": {"url": url, "name": "...", "club": None, "state": None, "start_date": None, "end_date": None},
      "rows": [
        {"shooter_name":"...", "profile_url": None, "ata_no": None,
         "discipline": None, "class": None, "yardage": None,
         "score": 98, "targets": 100}
      ]
    }
    """
    soup = BeautifulSoup(html, "html.parser")
    # Title / header
    name = None
    if soup.title and soup.title.text.strip():
        name = soup.title.text.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        name = h1.get_text(strip=True)

    event = {"url": url, "name": name, "club": None, "state": None, "start_date": None, "end_date": None}

    rows = []
    # VERY generic: if there are tables, try to read top row values that look like "Name 99 100"
    for tr in soup.select("table tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.select("td")]
        if len(tds) >= 2:
            # heuristic: if a column looks like a number out of 100, treat as score
            score = None
            for tok in tds:
                if tok.isdigit() and 0 <= int(tok) <= 100:
                    score = int(tok); break
            if score is not None:
                rows.append({
                    "shooter_name": tds[0],
                    "profile_url": None,
                    "ata_no": None,
                    "discipline": None,
                    "class": None,
                    "yardage": None,
                    "score": score,
                    "targets": 100
                })
    return {"event": event, "rows": rows}
