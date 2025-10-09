from bs4 import BeautifulSoup

def parse_event_results(html: str, event_url: str):
    """
    Return:
    {
      "event": {"url": event_url, "name": "...", "club": "...", "state": "ST", "start_date": "...", "end_date": "..."},
      "rows": [  # list of shooter rows
        {"shooter_name":"...", "profile_url": ".../Shooter/123" or None,
         "ata_no": "...", "discipline":"Singles", "class":"A", "yardage":27, "score":98, "targets":100},
        ...
      ]
    }
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- TODO: adjust selectors for ATA page structure you’ll use ---
    # Example placeholders:
    event = {
        "url": event_url,
        "name": (soup.find("h1") or {}).get_text(strip=True) if soup.find("h1") else None,
        "club": None,
        "state": None,
        "start_date": None,
        "end_date": None,
    }

    rows = []
    for tr in soup.select("table tr"):
        tds = [td.get_text(strip=True) for td in tr.select("td")]
        if not tds or len(tds) < 3:
            continue
        # You’ll map columns properly after inspecting a real page.
        row = {
            "shooter_name": tds[0],
            "profile_url": None,  # if the shooter name is a link, grab a.get("href")
            "ata_no": None,
            "discipline": "Singles",  # placeholder
            "class": None,
            "yardage": None,
            "score": int(tds[1]) if tds[1].isdigit() else None,
            "targets": 100,
        }
        rows.append(row)

    return {"event": event, "rows": rows}
