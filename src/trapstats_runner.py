#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrapStats Runner — Weekly/Monthly Automated Newsletter
Python 3.13 compatible

What it does:
1) Fetch score tables from provided URLs (ShootScoreboard + future ShootATA endpoints).
2) Parse and normalize into a single DataFrame.
3) Compute highlight stats (leaders, averages, streaks, club/event summaries).
4) Generate a clean Markdown newsletter.
5) (Optional) Email the Markdown to your Substack secret address or your list.
6) Persist a tiny cache for week-over-week comparisons.

Setup:
- pip install requests beautifulsoup4 pandas python-dateutil pydantic markdownify
- Set environment variables for emailing (optional)
  * SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
  * EMAIL_FROM, EMAIL_TO  (EMAIL_TO can be your Substack secret address)
- Customize SOURCE_URLS below with any specific event pages.
- Run: python trapstats_runner.py --period weekly
"""

from __future__ import annotations

import os
import re
import csv
import ssl
import time
import json
import smtplib
import hashlib
import logging
import argparse
from email.mime.text import MIMEText
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field, ValidationError

# ----------------------------- CONFIG ---------------------------------

APP_NAME = "TrapStats Runner"
APP_VERSION = "1.0.0"
UA = f"{APP_NAME}/{APP_VERSION} (+https://github.com/yourname/yourrepo)"

OUT_DIR = os.environ.get("OUT_DIR", "out")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")
DATA_SNAPSHOT = os.path.join(CACHE_DIR, "last_snapshot.csv")
META_SNAPSHOT = os.path.join(CACHE_DIR, "meta.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Put your shoot pages here (you can add more anytime).
# Examples from your earlier notes:
SOURCE_URLS = [
    # ShootScoreboard event pages:
    "https://shootscoreboard.com/scores.cfm?shootid=2004",
    "https://shootscoreboard.com/scores.cfm?shootid=1975",
    "https://shootscoreboard.com/scores.cfm?shootid=2005",
    # Directory/landing page (the script can discover event links too):
    "https://shootscoreboard.com/default.cfm",

    # (Future) If ShootATA exposes relevant event pages, add them here:
    # "https://shootata.com/.../some_event_page"
]

# Map typical column name variants we might encounter -> normalized names
COLUMN_ALIASES = {
    "Shooter": "shooter",
    "Shooter Name": "shooter",
    "Name": "shooter",
    "Last Name": "shooter",
    "Score": "score",
    "Event": "event",
    "Event Type": "event",
    "Class": "class",
    "Yardage": "yardage",
    "Club": "club",
    "Location": "location",
    "Date": "date",
    "Event Date": "date",
    "Category": "category",
}

# Minimal “feature switches”
EMAIL_ENABLED = bool(os.environ.get("SMTP_HOST"))

# ----------------------------- MODELS ---------------------------------

class NewsletterConfig(BaseModel):
    title_prefix: str = Field(default="TrapStats")
    period: str = Field(default="weekly")  # "weekly" or "monthly"
    top_n: int = Field(default=10)
    min_events_for_averages: int = Field(default=2)


class SourceResult(BaseModel):
    url: str
    table_count: int
    row_count: int


# ----------------------------- LOGGING --------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("trapstats")

# ----------------------------- UTILS ----------------------------------

def http_get(url: str, retries: int = 3, backoff: float = 1.5) -> requests.Response:
    """Polite GET with backoff and custom UA."""
    for i in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries - 1:
                raise
            sleep_for = backoff ** i
            log.warning(f"GET failed ({e}); retrying in {sleep_for:.1f}s ...")
            time.sleep(sleep_for)
    raise RuntimeError("Unreachable")

def week_bounds(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)          # Sunday
    return start, end

def month_bounds(d: date) -> Tuple[date, date]:
    start = date(d.year, d.month, 1)
    end = start + relativedelta(months=1) - timedelta(days=1)
    return start, end

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        norm = COLUMN_ALIASES.get(c, c)
        mapping[c] = norm
    return df.rename(columns=mapping)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # Scores: keep numeric if possible
    if "score" in df.columns:
        # Some scores might be like "100/100" -> take first part
        df["score"] = (
            df["score"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
    # Yardage numeric if present
    if "yardage" in df.columns:
        df["yardage"] = (
            df["yardage"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
    # Clean shooter/club/event strings
    for col in ["shooter", "club", "event", "location", "class", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def discover_event_links(html: str, base_url: str) -> List[str]:
    """From a directory page, find additional event links (heuristic)."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "scores.cfm?shootid=" in href and href.startswith("http"):
            links.append(href)
        elif "scores.cfm?shootid=" in href:
            # relative path
            if base_url.endswith("/"):
                links.append(base_url + href.lstrip("/"))
            else:
                links.append(base_url.rsplit("/", 1)[0] + "/" + href.lstrip("/"))
    return sorted(set(links))

def read_html_tables(url: str) -> List[pd.DataFrame]:
    """Try bs4 extraction first; fallback to pandas.read_html."""
    resp = http_get(url)
    # First, see if the page contains a main scores table
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    dfs: List[pd.DataFrame] = []
    for tbl in tables:
        try:
            # Convert table to DataFrame
            rows = []
            headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
            # If no th, infer headers from first row
            for tr in tbl.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if not rows:
                continue
            if headers and len(rows) > 1 and len(headers) == len(rows[1]):
                df = pd.DataFrame(rows[1:], columns=headers)
            else:
                df = pd.DataFrame(rows)
                # Set first row as header if it looks like header
                if df.shape[0] > 1:
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
            # Drop empty columns
            df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"\s*")]
            if df.shape[0] and df.shape[1]:
                dfs.append(df.reset_index(drop=True))
        except Exception:
            continue

    # Fallback
    if not dfs:
        try:
            dfs = pd.read_html(resp.text)  # type: ignore
        except ValueError:
            dfs = []
    return dfs

def parse_sources(urls: List[str]) -> Tuple[pd.DataFrame, List[SourceResult], List[str]]:
    """Return consolidated DataFrame + per-source summary + discovered URLs."""
    frames = []
    summaries: List[SourceResult] = []
    discovered: List[str] = []

    for url in urls:
        try:
            resp = http_get(url)
            html = resp.text
            # Discover more event links from landing pages
            more = discover_event_links(html, base_url=url)
            if more:
                discovered.extend(more)
            # Extract tables
            dfs = read_html_tables(url)
            row_count = 0
            for df in dfs:
                df = normalize_columns(df)
                df = coerce_types(df)
                # Heuristic: require at least a shooter+score or event+score
                cols = set(df.columns.str.lower())
                if {"score"} & cols and ({"shooter"} & cols or {"event"} & cols):
                    df["source"] = url
                    frames.append(df)
                    row_count += len(df)
            summaries.append(SourceResult(url=url, table_count=len(dfs), row_count=row_count))
            log.info(f"Parsed {url} -> {len(dfs)} tables, rows kept: {row_count}")
        except Exception as e:
            log.warning(f"Failed to parse {url}: {e}")

    # Deduplicate discovered links and (optionally) parse them too
    new_links = [u for u in sorted(set(discovered)) if u not in urls]
    for url in new_links:
        try:
            dfs = read_html_tables(url)
            row_count = 0
            for df in dfs:
                df = normalize_columns(df)
                df = coerce_types(df)
                cols = set(df.columns.str.lower())
                if {"score"} & cols and ({"shooter"} & cols or {"event"} & cols):
                    df["source"] = url
                    frames.append(df)
                    row_count += len(df)
            summaries.append(SourceResult(url=url, table_count=len(dfs), row_count=row_count))
            log.info(f"Parsed discovered {url} -> {len(dfs)} tables, rows kept: {row_count}")
        except Exception as e:
            log.warning(f"Failed to parse discovered {url}: {e}")

    if not frames:
        return pd.DataFrame(), summaries, new_links
    big = pd.concat(frames, ignore_index=True)
    # Standardize typical core columns presence
    for col in ["shooter", "score", "event", "club", "date", "location", "class", "category", "yardage"]:
        if col not in big.columns:
            big[col] = pd.NA
    # Drop fully empty rows on score
    big = big.dropna(subset=["score"], how="all")
    return big, summaries, new_links

def load_previous_snapshot() -> Optional[pd.DataFrame]:
    if not os.path.exists(DATA_SNAPSHOT):
        return None
    try:
        return pd.read_csv(DATA_SNAPSHOT)
    except Exception:
        return None

def save_snapshot(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    try:
        df_out = df.copy()
        df_out.to_csv(DATA_SNAPSHOT, index=False)
        with open(META_SNAPSHOT, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save snapshot: {e}")

def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    today = date.today()
    if period == "weekly":
        start, end = week_bounds(today)
    else:
        start, end = month_bounds(today)
    if "date" not in df.columns or df["date"].isna().all():
        # If no dates parsed, return full (some sites omit dates in tables)
        return df
    mask = df["date"].between(start, end)
    return df.loc[mask].copy()

def compute_highlights(df: pd.DataFrame, cfg: NewsletterConfig) -> Dict[str, Any]:
    highlights: Dict[str, Any] = {}

    if df.empty:
        return {
            "leaders_overall": [],
            "leaders_by_event": {},
            "clubs_top": [],
            "shooter_consistency": [],
            "volume_summary": {"rows": 0, "unique_shooters": 0, "unique_events": 0, "unique_clubs": 0},
        }

    # Volume
    highlights["volume_summary"] = {
        "rows": len(df),
        "unique_shooters": df["shooter"].nunique(dropna=True),
        "unique_events": df["event"].nunique(dropna=True),
        "unique_clubs": df["club"].nunique(dropna=True),
    }

    # Leaders overall
    leaders_overall = (
        df.dropna(subset=["score"])
        .sort_values("score", ascending=False)
        .head(cfg.top_n)
        .loc[:, ["shooter", "score", "event", "club", "date"]]
        .fillna("")
        .to_dict("records")
    )
    highlights["leaders_overall"] = leaders_overall

    # Leaders by event (Singles, Handicap, Doubles heuristics)
    event_map = {
        "singles": r"singles|single",
        "handicap": r"handicap|hdcp",
        "doubles": r"doubles|dbl",
    }
    leaders_by_event: Dict[str, List[Dict[str, Any]]] = {}
    if "event" in df.columns:
        for key, pat in event_map.items():
            sub = df[df["event"].astype(str).str.lower().str.contains(pat, na=False)]
            if not sub.empty:
                leaders_by_event[key] = (
                    sub.sort_values("score", ascending=False)
                    .head(cfg.top_n)
                    .loc[:, ["shooter", "score", "club", "date"]]
                    .fillna("")
                    .to_dict("records")
                )
    highlights["leaders_by_event"] = leaders_by_event

    # Shooter consistency (avg score for shooters with >= min_events_for_averages)
    if "shooter" in df.columns:
        counts = df.groupby("shooter", dropna=True)["score"].agg(["count", "mean", "max"]).reset_index()
        consistent = counts[counts["count"] >= cfg.min_events_for_averages]
        consistent = consistent.sort_values(["mean", "max"], ascending=[False, False]).head(cfg.top_n)
        highlights["shooter_consistency"] = (
            consistent.rename(columns={"count": "events", "mean": "avg", "max": "best"})
            .to_dict("records")
        )
    else:
        highlights["shooter_consistency"] = []

    # Club highlights
    if "club" in df.columns:
        clubs = (
            df.groupby("club", dropna=True)
            .agg(avg_score=("score", "mean"), events=("score", "count"))
            .reset_index()
            .sort_values(["avg_score", "events"], ascending=[False, False])
            .head(cfg.top_n)
        )
        highlights["clubs_top"] = clubs.to_dict("records")
    else:
        highlights["clubs_top"] = []

    return highlights

def compare_with_previous(current: pd.DataFrame, previous: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Return week-over-week deltas."""
    if previous is None or previous.empty:
        return {"new_rows": len(current), "delta_avg_score": None, "notes": "No previous snapshot."}
    try:
        curr_avg = current["score"].mean() if "score" in current.columns else None
        prev_avg = previous["score"].mean() if "score" in previous.columns else None
        delta_avg = None
        if curr_avg is not None and prev_avg is not None:
            delta_avg = round(curr_avg - prev_avg, 2)
        # New shooter count
        curr_shooters = set(current["shooter"].dropna().astype(str)) if "shooter" in current.columns else set()
        prev_shooters = set(previous["shooter"].dropna().astype(str)) if "shooter" in previous.columns else set()
        new_shooters = len(curr_shooters - prev_shooters)
        return {
            "new_rows": max(0, len(current) - len(previous)),
            "delta_avg_score": delta_avg,
            "new_shooters": new_shooters,
            "notes": "",
        }
    except Exception as e:
        return {"new_rows": len(current), "delta_avg_score": None, "notes": f"Compare error: {e}"}

def render_markdown(
    cfg: NewsletterConfig,
    period_start: date,
    period_end: date,
    highlights: Dict[str, Any],
    deltas: Dict[str, Any],
    source_summaries: List[SourceResult],
    discovered_links: List[str],
) -> str:
    title = f"{cfg.title_prefix} — {cfg.period.capitalize()} Report ({period_start.isoformat()} to {period_end.isoformat()})"
    lines = []
    lines.append(f"# {title}\n")
    lines.append("> Curated stats from public match score pages, summarized for busy trapshooters.\n")

    # Volume
    vol = highlights["volume_summary"]
    lines.append("## Volume at a Glance")
    lines.append(f"- Rows parsed: **{vol['rows']}**")
    lines.append(f"- Unique shooters: **{vol['unique_shooters']}**")
    lines.append(f"- Unique events: **{vol['unique_events']}**")
    lines.append(f"- Clubs seen: **{vol['unique_clubs']}**\n")

    # Deltas
    lines.append("## Week-over-Week (or Month-over-Month) Changes")
    delta_avg = "n/a" if deltas.get("delta_avg_score") is None else f"{deltas['delta_avg_score']:+.2f}"
    notes = deltas.get("notes") or ""
    lines.append(f"- New rows vs. last snapshot: **{deltas.get('new_rows', 'n/a')}**")
    lines.append(f"- Avg score change: **{delta_avg}**")
    if "new_shooters" in deltas:
        lines.append(f"- New shooters since last snapshot: **{deltas['new_shooters']}**")
    if notes:
        lines.append(f"- Note: {notes}")
    lines.append("")

    # Leaders overall
    lines.append("## Overall Leaders")
    if highlights["leaders_overall"]:
        lines.append("| Shooter | Score | Event | Club | Date |")
        lines.append("|---|---:|---|---|---|")
        for row in highlights["leaders_overall"]:
            lines.append(
                f"| {row.get('shooter','')} | {row.get('score','')} | {row.get('event','')} | {row.get('club','')} | {row.get('date','')} |"
            )
        lines.append("")
    else:
        lines.append("_No qualifying results found._\n")

    # Leaders by event
    lines.append("## Leaders by Event Type")
    if highlights["leaders_by_event"]:
        for etype, rows in highlights["leaders_by_event"].items():
            lines.append(f"### {etype.capitalize()}")
            if rows:
                lines.append("| Shooter | Score | Club | Date |")
                lines.append("|---|---:|---|---|")
                for r in rows:
                    lines.append(f"| {r.get('shooter','')} | {r.get('score','')} | {r.get('club','')} | {r.get('date','')} |")
                lines.append("")
    else:
        lines.append("_Event-type breakdown not available._\n")

    # Consistency
    lines.append("## Most Consistent Shooters (min events met)")
    if highlights["shooter_consistency"]:
        lines.append("| Shooter | Events | Avg | Best |")
        lines.append("|---|---:|---:|---:|")
        for r in highlights["shooter_consistency"]:
            lines.append(
                f"| {r.get('shooter','')} | {int(r.get('events',0))} | {float(r.get('avg',0)):.2f} | {float(r.get('best',0)):.0f} |"
            )
        lines.append("")
    else:
        lines.append("_Not enough multi-event shooters to compute consistency._\n")

    # Club highlights
    lines.append("## Club Highlights")
    if highlights["clubs_top"]:
        lines.append("| Club | Events | Avg Score |")
        lines.append("|---|---:|---:|")
        for r in highlights["clubs_top"]:
            lines.append(f"| {r.get('club','')} | {int(r.get('events',0))} | {float(r.get('avg_score',0)):.2f} |")
        lines.append("")
    else:
        lines.append("_No club summary available._\n")

    # Provenance (transparent sourcing without linking to a specific ToS-sensitive pattern)
    lines.append("## Data Coverage (summary of pages parsed)")
    lines.append("| URL | Tables | Rows Kept |")
    lines.append("|---|---:|---:|")
    for s in source_summaries:
        lines.append(f"| {s.url} | {s.table_count} | {s.row_count} |")
    if discovered_links:
        lines.append("\n_Additional event pages discovered during crawl:_")
        for u in discovered_links[:15]:
            lines.append(f"- {u}")
    lines.append("")

    lines.append("---\n*Generated automatically. Always verify critical results with official match postings.*")
    return "\n".join(lines)

def email_markdown(subject: str, body_md: str) -> None:
    """Send Markdown as plaintext email (Substack supports email-to-post)."""
    if not EMAIL_ENABLED:
        log.info("Email not configured; skipping send.")
        return
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    email_from = os.environ.get("EMAIL_FROM")
    email_to = os.environ.get("EMAIL_TO")
    if not all([host, port, user, pw, email_from, email_to]):
        log.warning("Missing SMTP env vars; skipping send.")
        return

    msg = MIMEText(body_md, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(user, pw)
        server.sendmail(email_from, [email_to], msg.as_string())
    log.info(f"Emailed newsletter to {email_to}")

def run(period: str = "weekly", title_prefix: str = "TrapStats") -> str:
    # Determine period window for header text
    today = date.today()
    if period == "weekly":
        period_start, period_end = week_bounds(today)
    else:
        period_start, period_end = month_bounds(today)

    cfg = NewsletterConfig(title_prefix=title_prefix, period=period)
    log.info(f"Starting run for period={period} ({period_start}..{period_end})")

    # 1) Parse data
    df_all, src_summaries, discovered = parse_sources(SOURCE_URLS)

    # 2) Filter to period range (if dates present)
    df_period = filter_by_period(df_all, period)

    # 3) Compute highlights
    highlights = compute_highlights(df_period, cfg)

    # 4) Load previous snapshot & compute deltas
    prev = load_previous_snapshot()
    deltas = compare_with_previous(df_period, prev)

    # 5) Render markdown
    if period == "weekly":
        pstart, pend = week_bounds(today)
    else:
        pstart, pend = month_bounds(today)

    md = render_markdown(
        cfg=cfg,
        period_start=pstart,
        period_end=pend,
        highlights=highlights,
        deltas=deltas,
        source_summaries=src_summaries,
        discovered_links=discovered,
    )

    # 6) Save artifact
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"{period}_{stamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    log.info(f"Wrote newsletter: {out_path}")

    # 7) Save snapshot for next run
    meta = {
        "period": period,
        "generated_at": datetime.now().isoformat(),
        "rows": len(df_period),
    }
    # Save only the period-filtered (so WoW comparisons match the content)
    try:
        save_snapshot(df_period, meta)
    except Exception as e:
        log.warning(f"Snapshot save failed: {e}")

    # 8) Optional: send email
    subject = f"{cfg.title_prefix} — {cfg.period.capitalize()} ({pstart.isoformat()}–{pend.isoformat()})"
    email_markdown(subject, md)

    return out_path

# ----------------------------- CLI ------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and optionally email the TrapStats newsletter.")
    parser.add_argument("--period", choices=["weekly", "monthly"], default="weekly", help="Reporting period window")
    parser.add_argument("--title", default="TrapStats", help="Newsletter title prefix")
    args = parser.parse_args()

    path = run(period=args.period, title_prefix=args.title)
    print(f"OK: {path}")
