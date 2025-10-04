"""
Central data pipeline.
Swap the placeholder functions with real fetch/parse once ATA approves.
"""

from datetime import date, timedelta

def week_bounds(d: date):
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=6)
    return start, end

# -------- FETCHERS (replace later with real code) --------
def fetch_weekly_results():
    return []  # list of dicts: {event, club, state, dates, url}

def fetch_averages_and_movers():
    return {"singles": [], "handicap": [], "doubles": [], "movers": []}

def fetch_participation_by_club_state():
    return {"clubs": [], "states": []}

def fetch_milestones():
    return []  # list of strings

def fetch_upcoming_shoots():
    return []  # list of dicts

def fetch_ata_updates():
    return []  # list of dicts/strings

# -------- RENDER HELPERS --------
def bullets(items):
    return "".join(f"- {line}\n" for line in items) if items else "_No updates this week._\n"

# -------- BLOCK BUILDERS (used by generate_post.py) --------
def build_top_shoots_block():
    sample = [
        "(sample) Shoot A — Club X, ST (Apr 5–6) → [Full results](#)",
        "(sample) Shoot B — Club Y, FL (Apr 6) → [Full results](#)"
    ]
    return bullets(sample)

def build_leaders_block():
    sample = [
        "(sample) Singles: Jane D. 100/100 (99.2%)",
        "(sample) Handicap: Mike R. 98/100 (27 yd)",
        "(sample) Doubles: Alex P. 98/100 (98.5%)"
    ]
    return bullets(sample)

def build_who_shot_where_block():
    sample = [
        "(sample) Club X: 34 shooters",
        "(sample) Club Y: 18 shooters"
    ]
    return bullets(sample)

def build_milestones_block():
    sample = ["(sample) John S. passed 25,000 lifetime targets"]
    return bullets(sample)
