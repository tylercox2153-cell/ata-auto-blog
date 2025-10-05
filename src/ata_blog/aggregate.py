from datetime import date, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

def week_bounds(d: date):
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)          # Sunday
    return start, end

@dataclass
class ShooterResult:
    name: str
    event: str   # e.g., "Singles", "Handicap", "Doubles"
    score: int
    out_of: int
    yardage: Optional[int] = None
    club: Optional[str] = None
    state: Optional[str] = None

@dataclass
class ShootSummary:
    title: str
    club: str
    state: str
    start_date: date
    end_date: date
    url: str

def compute_leaders(results: List[ShooterResult]) -> Dict[str, Any]:
    # simple best-of for each event
    leaders = {}
    for r in results:
        key = r.event.lower()
        best = leaders.get(key)
        if (best is None) or (r.score * 1.0 / r.out_of) > (best.score * 1.0 / best.out_of):
            leaders[key] = r
    return leaders

def build_weekly_context(shoots: List[ShootSummary], results: List[ShooterResult]):
    leaders = compute_leaders(results)
    return {
        "shoots": shoots,
        "leaders": leaders
    }
