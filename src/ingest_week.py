from datetime import date
from config.settings import ATA_SEED_EVENT_URLS
from src.storage import init_db, get_conn
from src.fetch import fetch_html
from src.parsers.ata_event_results import parse_event_results

def upsert_event(c, ev):
    c.execute("""
      INSERT INTO events (url, name, club, state, start_date, end_date, last_seen)
      VALUES (?, ?, ?, ?, ?, ?, DATE('now'))
      ON CONFLICT(url) DO UPDATE SET
        name=excluded.name, club=excluded.club, state=excluded.state,
        start_date=excluded.start_date, end_date=excluded.end_date,
        last_seen=excluded.last_seen
    """, (ev["url"], ev["name"], ev["club"], ev["state"], ev["start_date"], ev["end_date"]))

def upsert_shooter(c, name, profile_url, ata_no):
    c.execute("""
      INSERT INTO shooters (name, profile_url, ata_no, last_seen)
      VALUES (?, ?, ?, DATE('now'))
      ON CONFLICT(profile_url) DO UPDATE SET
        name=excluded.name, ata_no=excluded.ata_no, last_seen=excluded.last_seen
    """, (name, profile_url, ata_no))

def insert_result(c, event_url, row):
    c.execute("""
      INSERT INTO results (event_url, shooter_profile, shooter_name, ata_no, discipline, yardage, score, targets, class)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_url, row.get("profile_url"), row.get("shooter_name"), row.get("ata_no"),
        row.get("discipline"), row.get("yardage"), row.get("score"), row.get("targets"), row.get("class")
    ))

def main():
    init_db()
    with get_conn() as c:
        for url in ATA_SEED_EVENT_URLS:
            html = fetch_html(url)
            parsed = parse_event_results(html, url)
            ev = parsed["event"]
            upsert_event(c, ev)
            for r in parsed["rows"]:
                upsert_shooter(c, r["shooter_name"], r.get("profile_url"), r.get("ata_no"))
                insert_result(c, ev["url"], r)
        c.commit()
    print("[ok] ingested weekly seed events into DB")

if __name__ == "__main__":
    main()
