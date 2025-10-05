from datetime import date, timedelta
from pathlib import Path
from . import data_pipeline as dp

TEMPLATE_PATH = Path("templates/weekly_post.md")
OUT_DIR = Path("out")


def week_bounds(d: date):
    """Return (Mon_start, Sun_end) for the week containing date d."""
    start = d - timedelta(days=d.weekday())   # Monday
    end = start + timedelta(days=6)           # Sunday
    return start, end


def build_weekly_context():
    """Collect all section blocks for the weekly post."""
    start, _ = week_bounds(date.today())
    return {
        "week_start": start.isoformat(),
        "top_shoots_block": dp.build_top_shoots_block(),
        "leaders_block": dp.build_leaders_block(),
        "who_shot_where_block": dp.build_who_shot_where_block(),
        "milestones_block": dp.build_milestones_block(),
        "last_weekend_block": dp.build_last_weekend_shoots_block(),
    }


def render_template(tpl: str, ctx: dict) -> str:
    """Lightweight templating: replace {{keys}} with values."""
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
