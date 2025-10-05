from . import data_pipeline as dp
from pathlib import Path

TEMPLATE_PATH = Path("templates/weekly_post.md")
OUT_DIR = Path("out")

def week_bounds(d: date):
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)          # Sunday
    return start, end

def build_weekly_context("last_weekend_block": dp.build_last_weekend_shoots_block(),
):
    """stub data for now—later we’ll replace with real parsed stats."""
    start, _ = week_bounds(date.today())
    return {
        "week_start": start.isoformat(),
        "top_shoots_block": "- (sample) Shoot A — Club X, ST (Apr 5–6)\n- (sample) Shoot B — Club Y, FL (Apr 6)\n",
        "leaders_block": "- (sample) Singles: Jane D. 100/100 (99.2%)\n- (sample) Handicap: Mike R. 98/100 (27 yd)\n- (sample) Doubles: Alex P. 98/100 (98.5%)\n",
        "who_shot_where_block": "- (sample) Club X: 34 shooters\n- (sample) Club Y: 18 shooters\n",
        "milestones_block": "- (sample) John S. passed 25,000 lifetime targets\n",
    }

def render_template(tpl: str, ctx: dict) -> str:
    """lightweight templating: replace {{keys}} with values"""
    out = tpl
    for k, v in ctx.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    ctx = build_weekly_context()
    content = render_template(tpl, ctx)
    outfile = OUT_DIR / f"weekly-{ctx['week_start']}.md"
    outfile.write_text(content, encoding="utf-8")
    print(f"[ok] wrote {outfile}")

if __name__ == "__main__":
    main()
