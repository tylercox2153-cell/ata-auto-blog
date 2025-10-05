#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrapStats Runner — Weekly/Monthly Automated Newsletter
Python 3.12+ compatible

What it does:
1) Fetch score tables from provided URLs (ShootScoreboard + future ShootATA).
2) Parse and normalize into a single DataFrame.
3) Compute highlight stats (leaders, averages, club/event summaries).
4) Generate a Markdown newsletter in out/.
5) (Optional) Email the Markdown to your Substack secret address.
6) Save a small snapshot in .cache/ for WoW deltas.

Deps:
  pip install requests beautifulsoup4 pandas python-dateutil pydantic markdownify lxml

Env (optional for email):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_FROM, EMAIL_TO
"""

from __future__ import annotations

import os
import re
import ssl
import time
import json
import smtplib
import logging
import argparse
from email.mime.text import MIMEText
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field

# ----------------------------- CONFIG ---------------------------------

APP_NAME = "TrapStats Runner"
APP_VERSION = "1.1.0"
UA = f"{APP_NAME}/{APP_VERSION} (+https://github.com/tylercox2153-cell/ata-auto-blog)"

OUT_DIR = os.environ.get("OUT_DIR", "out")
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")
DATA_SNAPSHOT = os.path.join(CACHE_DIR, "last_snapshot.csv")
META_SNAPSHOT = os.path.join(CACHE_DIR, "meta.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Add your pages here. You can keep default.cfm to allow discovery.
SOURCE_URLS = [
    "https://shootscoreboard.com/scores.cfm?shootid=2004",
    "https://shootscoreboard.com/scores.cfm?shootid=1975",
    "https://shootscoreboard.com/scores.cfm?shootid=2005",
    "https://shootscoreboard.com/default.cfm",
]

# Column normalization map
COLUMN_ALIASES = {
    "Shooter": "shooter",
    "Shooter Name": "shooter",
    "Name": "shooter",
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trapstats")

# ----------------------------- HELPERS --------------------------------

def http_get(url: str, retries: int = 3, backoff: float = 1.7) -> requests.Response:
    for i in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries - 1:
                raise
            sleep_for = backoff ** i
            log.warning(f"GET {url} failed: {e} — retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError("unreachable")

def week_bounds(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=6)
    return start, end

def month_bounds(d: date) -> Tuple[date, date]:
    start = date(d.year, d.month, 1)
    end = start + relativedelta(months=1) - timedelta(days=1)
    return start, end

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # harmonize headers
    ren = {}
    for c in df.columns:
        ren[c] = COLUMN_ALIASES.get(str(c).strip(), str(c).strip().lower())
    df = df.rename(columns=ren)
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # score (handles "100/100", "98", "98x100")
    if "score" in df.columns:
        df["score"] = (
            df["score"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
    # yardage numeric
    if "yardage" in df.columns:
        df["yardage"] = (
            df["yardage"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        )
    # strip text-like cols
    for c in ["shooter", "club", "event", "location", "class", "category"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def discover_event_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "scores.cfm?shootid=" in href:
            if href.startswith("http"):
                links.append(href)
            else:
                if base_url.endswith("/"):
                    links.append(base_url + href.lstrip("/"))
                else:
                    links.append(base_url.rsplit("/", 1)[0] + "/" + href.lstrip("/"))
    return sorted(set(links))

def read_html_tables(url: str) -> List[pd.DataFrame]:
    """Parse all <table> elements via bs4; fallback to pandas.read_html."""
    resp = http_get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    dfs: List[pd.DataFrame] = []

    # bs4 → DataFrame
    for tbl in soup.find_all("table"):
        rows = []
        heads = [th.get_text(strip=True) for th in tbl.find_all("th")]
        for tr in tbl.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        try:
            if heads and len(rows) > 1 and len(heads) == len(rows[1]):
                df = pd.DataFrame(rows[1:], columns=heads)
            else:
                df = pd.DataFrame(rows)
                if df.shape[0] > 1:
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
            # drop fully empty cols
            df = df.loc[:, ~pd.Index(df.columns).map(lambda c: str(c).strip()).str.fullmatch(r"")]
            if df.shape[0] and df.shape[1]:
                dfs.append(df.reset_index(drop=True))
        except Exception:
            continue

    # fallback
    if not dfs:
        try:
            dfs = pd.read_html(resp.text)  # requires lxml
        except Exception:
            dfs = []
    return dfs

def _has_core_columns(df: pd.DataFrame) -> bool:
    cols = set(pd.Index(df.columns).map(lambda c: str(c).strip().lower()))
    return ("score" in cols) and (("shooter" in cols) or ("event" in cols))

def parse_sources(urls: List[str]) -> Tuple[pd.DataFrame, List[SourceResult], List[str]]:
    frames: List[pd.DataFrame] = []
    summaries: List[SourceResult] = []
    discovered: List[str] = []

    # First pass: explicit URLs
    for url in urls:
        try:
            resp = http_get(url)
            html = resp.text
            more = discover_event_links(html, url)
            discovered.extend(more)

            dfs = read_html_tables(url)
            kept = 0
            for df in dfs:
                df = normalize_columns(df)
                df = coerce_types(df)
                if _has_core_columns(df):
                    df["source"] = url
                    frames.append(df)
                    kept += len(df)
            summaries.append(SourceResult(url=url, table_count=len(dfs), row_count=kept))
            log.info(f"Parsed {url} -> {len(dfs)} tables, rows kept: {kept}")
        except Exception as e:
            summaries.append(SourceResult(url=url, table_count=0, row_count=0))
            log.warning(f"Failed to parse {url}: {e}")

    # Second pass: discovered event pages (unique)
    for url in [u for u in sorted(set(discovered)) if u not in urls]:
        try:
            dfs = read_html_tables(url)
            kept = 0
            for df in dfs:
                df = normalize_columns(df)
                df = coerce_types(df)
                if _has_core_columns(df):
                    df["source"] = url
                    frames.append(df)
                    kept += len(df)
            summaries.append(SourceResult(url=url, table_count=len(dfs), row_count=kept))
            log.info(f"Parsed discovered {url} -> {len(dfs)} tables, rows kept: {kept}")
        except Exception as e:
            summaries.append(SourceResult(url=url, table_count=0, row_count=0))
            log.warning(f"Failed to parse discovered {url}: {e}")

    if not frames:
        return pd.DataFrame(), summaries, sorted(set(discovered))
    big = pd.concat(frames, ignore_index=True)

    # Ensure standard columns exist
    for c in ["shooter", "score", "event", "club", "date", "location", "class", "category", "yardage"]:
        if c not in big.columns:
            big[c] = pd.NA

    # Drop rows with no numeric score
    big = big.dropna(subset=["score"], how="all")
    return big, summaries, sorted(set(discovered))

def load_previous_snapshot() -> Optional[pd.DataFrame]:
    if not os.path.exists(DATA_SNAPSHOT):
        return None
    try:
        return pd.read_csv(DATA_SNAPSHOT)
    except Exception:
        return None

def save_snapshot(df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    try:
        df.to_csv(DATA_SNAPSHOT, index=False)
        with open(META_SNAPSHOT, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save snapshot: {e}")

def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty:
        return df
    today = date.today()
    start, end = week_bounds(today) if period == "weekly" else month_bounds(today)
    if "date" not in df.columns or df["date"].isna().all():
        return df
    mask = df["date"].between(start, end)
    out = df.loc[mask].copy()
    return out if not out.empty else df  # if filtering kills everything, keep all

def compute_highlights(df: pd.DataFrame, cfg: NewsletterConfig) -> Dict[str, Any]:
    if df.empty:
        return {
            "volume_summary": {"rows": 0, "unique_shooters": 0, "unique_events": 0, "unique_clubs": 0},
            "leaders_overall": [],
            "leaders_by_event": {},
            "shooter_consistency": [],
            "clubs_top": [],
        }

    vol = {
        "rows": len(df),
        "unique_shooters": df["shooter"].nunique(dropna=True),
        "unique_events": df["event"].nunique(dropna=True),
        "unique_clubs": df["club"].nunique(dropna=True),
    }

    leaders_overall = (
        df.dropna(subset=["score"])
        .sort_values("score", ascending=False)
        .head(cfg.top_n)
        .loc[:, ["shooter", "score", "event", "club", "date"]]
        .fillna("")
        .to_dict("records")
    )

    event_map = {"singles": r"singles|single", "handicap": r"handicap|hdcp", "doubles": r"doubles|dbl"}
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

    shooter_consistency = []
    if "shooter" in df.columns:
        counts = df.groupby("shooter", dropna=True)["score"].agg(["count", "mean", "max"]).reset_index()
        consistent = counts[counts["count"] >= cfg.min_events_for_averages]
        consistent = consistent.sort_values(["mean", "max"], ascending=[False, False]).head(cfg.top_n)
        shooter_consistency = (
            consistent.rename(columns={"count": "events", "mean": "avg", "max": "best"}).to_dict("records")
        )

    clubs_top = []
    if "club" in df.columns:
        clubs = (
            df.groupby("club", dropna=True).agg(avg_score=("score", "mean"), events=("score", "count")).reset_index()
        )
        clubs_top = clubs.sort_values(["avg_score", "events"], ascending=[False, False]).head(cfg.top_n).to_dict("records")

    return {
        "volume_summary": vol,
        "leaders_overall": leaders_overall,
        "leaders_by_event": leaders_by_event,
        "shooter_consistency": shooter_consistency,
        "clubs_top": clubs_top,
    }

def compare_with_previous(current: pd.DataFrame, previous: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if previous is None or previous.empty:
        return {"new_rows": len(current), "delta_avg_score": None, "new_shooters": None, "notes": "First snapshot."}
    try:
        curr_avg = current["score"].mean() if "score" in current.columns else None
        prev_avg = previous["score"].mean() if "score" in previous.columns else None
        delta_avg = None if (curr_avg is None or prev_avg is None) else round(curr_avg - prev_avg, 2)

        curr_shooters = set(current["shooter"].dropna().astype(str)) if "shooter" in current.columns else set()
        prev_shooters = set(previous["shooter"].dropna().astype(str)) if "shooter" in previous.columns else set()
        new_shooters = len(curr_shooters - prev_shooters)

        return {"new_rows": max(0, len(current) - len(previous)), "delta_avg_score": delta_avg, "new_shooters": new_shooters, "notes": ""}
    except Exception as e:
        return {"new_rows": len(current), "delta_avg_score": None, "new_shooters": None, "notes": f"Compare error: {e}"}

def render_markdown(cfg: NewsletterConfig, period_start: date, period_end: date,
                    highlights: Dict[str, Any], deltas: Dict[str, Any],
                    source_summaries: List[SourceResult], discovered_links: List[str]) -> str:
    title = f"{cfg.title_prefix} — {cfg.period.capitalize()} Report ({period_start.isoformat()} to {period_end.isoformat()})"
    lines = [f"# {title}\n",
             "> Curated stats from public match score pages, summarized for busy trapshooters.\n"]

    vol = highlights["volume_summary"]
    lines += [
        "## Volume at a Glance",
        f"- Rows parsed: **{vol['rows']}**",
        f"- Unique shooters: **{vol['unique_shooters']}**",
        f"- Unique events: **{vol['unique_events']}**",
        f"- Clubs seen: **{vol['unique_clubs']}**",
        "",
        "## Week-over-Week (or Month-over-Month) Changes",
        f"- New rows vs. last snapshot: **{deltas.get('new_rows','n/a')}**",
        f"- Avg score change: **{'n/a' if deltas.get('delta_avg_score') is None else f'{deltas['delta_avg_score']:+.2f}'}**",
        f"- New shooters since last snapshot: **{deltas.get('new_shooters','n/a')}**",
        "" if not deltas.get("notes") else f"- Note: {deltas['notes']}",
        ""
    ]

    lines.append("## Overall Leaders")
    if highlights["leaders_overall"]:
        lines += ["| Shooter | Score | Event | Club | Date |", "|---|---:|---|---|---|"]
        for r in highlights["leaders_overall"]:
            lines.append(f"| {r.get('shooter','')} | {r.get('score','')} | {r.get('event','')} | {r.get('club','')} | {r.get('date','')} |")
        lines.append("")
    else:
        lines.append("_No qualifying results found._\n")

    lines.append("## Leaders by Event Type")
    if highlights["leaders_by_event"]:
        for etype, rows in highlights["leaders_by_event"].items():
            lines += [f"### {etype.capitalize()}", "| Shooter | Score | Club | Date |", "|---|---:|---|---|"]
            for r in rows:
                lines.append(f"| {r.get('shooter','')} | {r.get('score','')} | {r.get('club','')} | {r.get('date','')} |")
            lines.append("")
    else:
        lines.append("_Event-type breakdown not available._\n")

    lines.append("## Most Consistent Shooters (min events met)")
    if highlights["shooter_consistency"]:
        lines += ["| Shooter | Events | Avg | Best |", "|---|---:|---:|---:|"]
        for r in highlights["shooter_consistency"]:
            lines.append(f"| {r.get('shooter','')} | {int(r.get('events',0))} | {float(r.get('avg',0)):.2f} | {float(r.get('best',0)):.0f} |")
        lines.append("")
    else:
        lines.append("_Not enough multi-event shooters to compute consistency._\n")

    lines.append("## Club Highlights")
    if highlights["clubs_top"]:
        lines += ["| Club | Events | Avg Score |", "|---|---:|---:|"]
        for r in highlights["clubs_top"]:
            lines.append(f"| {r.get('club','')} | {int(r.get('events',0))} | {float(r.get('avg_score',0)):.2f} |")
        lines.append("")
    else:
        lines.append("_No club summary available._\n")

    lines += ["## Data Coverage (summary of pages parsed)", "| URL | Tables | Rows Kept |", "|---|---:|---:|"]
    for s in source_summaries:
        lines.append(f"| {s.url} | {s.table_count} | {s.row_count} |")
    if discovered_links:
        lines.append("\n_Discovered additional event pages:_")
        for u in discovered_links[:20]:
            lines.append(f"- {u}")
    lines.append("\n---\n*Generated automatically. Verify with official match postings.*")
    return "\n".join(lines)

def email_markdown(subject: str, body_md: str) -> None:
    if not EMAIL_ENABLED:
        log.info("Email not configured; skipping send.")
        return
    host = os.environ["SMTP_HOST"]; port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER"); pw = os.environ.get("SMTP_PASS")
    email_from = os.environ.get("EMAIL_FROM"); email_to = os.environ.get("EMAIL_TO")
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
    today = date.today()
    period_start, period_end = week_bounds(today) if period == "weekly" else month_bounds(today)
    cfg = NewsletterConfig(title_prefix=title_prefix, period=period)

    log.info(f"Starting run for period={period} ({period_start}..{period_end})")

    df_all, src_summaries, discovered = parse_sources(SOURCE_URLS)
    if df_all.empty:
        log.warning("Parsed no rows; generating skeleton post so you can verify pipeline.")

    df_period = filter_by_period(df_all, period)
    highlights = compute_highlights(df_period, cfg)
    prev = load_previous_snapshot()
    deltas = compare_with_previous(df_period, prev)

    md = render_markdown(cfg, period_start, period_end, highlights, deltas, src_summaries, discovered)

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(OUT_DIR, f"{period}_{stamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    log.info(f"Wrote newsletter: {out_path}")

    try:
        save_snapshot(df_period, {"period": period, "generated_at": datetime.now().isoformat(), "rows": len(df_period)})
    except Exception as e:
        log.warning(f"Snapshot save failed: {e}")

    subject = f"{cfg.title_prefix} — {cfg.period.capitalize()} ({period_start.isoformat()}–{period_end.isoformat()})"
    email_markdown(subject, md)
    return out_path

# ----------------------------- CLI ------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate (and optionally email) the TrapStats newsletter.")
    parser.add_argument("--period", choices=["weekly", "monthly"], default="weekly")
    parser.add_argument("--title", default="TrapStats")
    args = parser.parse_args()
    path = run(period=args.period, title_prefix=args.title)
    print(f"OK: {path}")
