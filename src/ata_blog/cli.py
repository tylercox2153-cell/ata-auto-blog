from pathlib import Path
from datetime import date
from rich import print
from dotenv import load_dotenv

from ata_blog.sources.shootscoreboard import parse_shoot_page, ShootRow, ResultRow
from ata_blog.aggregate import build_weekly_context, ShootSummary, ShooterResult
from ata_blog.render import render_markdown
# optional:
try:
    from ata_blog.publish.emailer import send_email_markdown
except Exception:
    send_email_markdown = None  # noqa

TEMPLATES = Path("templates")
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def map_rows(shoots: list[ShootRow], results: list[ResultRow]):
    shoots_m = [
        ShootSummary(
            title=s.title, club=s.club, state=s.state,
            start_date=s.start_date, end_date=s.end_date, url=s.url
        )
        for s in shoots
    ]
    results_m = [
        ShooterResult(
            name=r.name, event=r.event, score=r.score, out_of=r.out_of,
            yardage=r.yardage, club=r.club, state=r.state
        )
        for r in results
    ]
    return shoots_m, results_m

def weekly():
    load_dotenv()
    print("[bold]ATA Weekly Builder[/bold]")

    # 👉 Add the exact shoot URLs you care about this week:
    shoot_urls = [
        # examples:
        # "https://shootscoreboard.com/scores.cfm?shootid=2004",
        # "https://shootscoreboard.com/scores.cfm?shootid=1975",
        # "https://shootscoreboard.com/scores.cfm?shootid=2005",
    ]
    all_results: list[ResultRow] = []
    shoots_simple: list[ShootRow] = []
    for u in shoot_urls:
        print(f"Scraping: {u}")
        all_results.extend(parse_shoot_page(u))
        shoots_simple.append(ShootRow(
            title=f"Shoot {u.split('=')[-1]}", club="TBD", state="TBD",
            start_date=date.today(), end_date=date.today(), url=u
        ))

    shoots_m, results_m = map_rows(shoots_simple, all_results)
    ctx = build_weekly_context(shoots_m, results_m)
    md = render_markdown(TEMPLATES, "weekly_post.md.j2", ctx)

    outfile = OUT_DIR / f"weekly-{date.today().isoformat()}.md"
    outfile.write_text(md, encoding="utf-8")
    print(f"[green]Wrote[/green] {outfile}")

    if send_email_markdown and (os.getenv("EMAIL_TO")):
        try:
            send_email_markdown(subject=f"ATA Weekly – {date.today().isoformat()}", markdown_body=md)
            print("[blue]Emailed draft successfully[/blue]")
        except Exception as e:
            print(f"[yellow]Email skipped/failed: {e}[/yellow]")

if __name__ == "__main__":
    weekly()

