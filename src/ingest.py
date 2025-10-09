import yaml
from pathlib import Path
from src.db import init_db, get_conn
from src.fetch import fetch_html
from src.parsers.shootscoreboard_event import parse_shootscoreboard_event
from src.parsers.ata_event import parse_ata_event

def load_events_config(path="config/events.yml"):
    p = Path(path)
    if not p.exists():
        print(f"[warn] {path} not found")
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("events", [])

def upsert_event(c, ev):
    c.execute("""
      INSERT INTO events (url, source, name, club, state, start_date, end_date, last_seen)
      VALUES (?, ?, ?, ?, ?, ?, ?, DATE('now'))
      ON CONFLICT(url) DO UPDATE SET
        source=excluded.source, name=excluded.name, club=excluded.club, state=excluded.state,
        start_date=excluded.start_date, end_date=excluded.end_date, last_seen=excluded.last_seen
    """, (ev["url"], ev.get("source"), ev.get("name"), ev.get("club"), ev.get("state"),
          ev.get("start_date"), ev.get("end_date")))

def upsert_shooter(c, name, profile_url, ata_no):
    # Use profile_url when available to dedupe; otherwise just insert by name
    if profile_url:
        c.execute("""
          INSERT INTO shooters (name, profile_url, ata_no, last_seen)
          VALUES (?, ?, ?, DATE('now'))
          ON CONFLICT(profile_url) DO UPDATE SET
            name=excluded.name, ata_no=excluded.ata_no, last_seen=excluded.last_seen
        """, (name, profile_url, ata_no))
    else:
        c.execute("""
          INSERT INTO shooters (name, last_seen)
          VALUES (?, DATE('now'))
        """, (name,))

def insert_result(c, event_url, r):
    c.execute("""
      INSERT INTO results (event_url, shooter_profile, shooter_name, ata_no, discipline, yardage, score, targets, class)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_url, r.get("profile_url"), r.get("shooter_name"), r.get("ata_no"),
          r.get("discipline"), r.get("yardage"), r.get("score"), r.get("targets"), r.get("class")))

def parse_route(source: str, html: str, url: str):
    if source == "shootscoreboard":
        data = parse_shootscoreboard_event(html, url)
    elif source == "shootata":
        data = parse_ata_event(html, url)
    else:
        raise ValueError(f"Unknown source {source}")
    data["event"]["source"] = source
    return data

def main():
    init_db()
    events = load_events_config()
    if not events:
        print("[warn] no events in config/events.yml")
        return

    with get_conn() as c:
        for e in events:
            src = e["source"]
            url = e["url"]
            print(f"[ingest] {src}: {url}")
            html = fetch_html(url)
            parsed = parse_route(src, html, url)
            upsert_event(c, parsed["event"])
            for r in parsed["rows"]:
                upsert_shooter(c, r.get("shooter_name"), r.get("profile_url"), r.get("ata_no"))
                insert_result(c, parsed["event"]["url"], r)
        c.commit()
    print("[ok] ingested events")

if __name__ == "__main__":
    main()
