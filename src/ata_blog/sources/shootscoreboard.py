from __future__ import annotations
import requests
from bs4 import BeautifulSoup
from datetime import date
from typing import List
from dataclasses import dataclass
from dateutil import parser as dtp

@dataclass
class ShootRow:
    title: str
    club: str
    state: str
    start_date: date
    end_date: date
    url: str

@dataclass
class ResultRow:
    name: str
    event: str
    score: int
    out_of: int
    yardage: int | None
    club: str | None
    state: str | None

HEADERS = {"User-Agent": "ata-auto-blog/1.0 (education, non-commercial)"}

def get(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

def parse_shoot_page(url: str) -> List[ResultRow]:
    """Parse a specific shoot's results page: https://shootscoreboard.com/scores.cfm?shootid=XXXX"""
    soup = get(url)
    rows: List[ResultRow] = []

    # Example heuristic: find result tables by header keywords
    for table in soup.find_all("table"):
        header_text = (table.find("thead") or table).get_text(" ", strip=True).lower()
        if any(k in header_text for k in ["singles", "handicap", "doubles", "score"]):
            for tr in table.find_all("tr")[1:]:
                tds = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
                if len(tds) < 3:
                    continue
                # very rough parsing; improve as you learn the page shape
                name = tds[0]
                score_bits = [s for s in tds if "/" in s]
                event_guess = "Singles" if "single" in header_text else ("Handicap" if "handicap" in header_text else ("Doubles" if "double" in header_text else "Event"))
                if score_bits:
                    try:
                        score, out_of = score_bits[0].split("/", 1)
                        score_i = int(score)
                        out_of_i = int(out_of)
                    except Exception:
                        continue
                else:
                    continue
                rows.append(ResultRow(
                    name=name, event=event_guess, score=score_i, out_of=out_of_i,
                    yardage=None, club=None, state=None
                ))
    return rows

def parse_listing_page(url: str) -> List[ShootRow]:
    """Parse a listing of shoots (if you have a listing page)."""
    soup = get(url)
    shoots: List[ShootRow] = []
    for a in soup.select("a[href*='scores.cfm?shootid=']"):
        text = a.get_text(" ", strip=True)
        href = a.get("href")
        if not href:
            continue
        shoot_url = href if href.startswith("http") else f"https://shootscoreboard.com/{href.lstrip('/')}"
        # naive date extraction (customize later)
        # fallback: set today's date for both
        sdate = dtp.parse("today").date()
        shoots.append(ShootRow(title=text, club="Unknown Club", state="UNK",
                               start_date=sdate, end_date=sdate, url=shoot_url))
    return shoots
