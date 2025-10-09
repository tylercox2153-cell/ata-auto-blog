from src.storage import get_conn

def build_top_shoots_block():
    q = """
    SELECT name, club, state, start_date, end_date, url
    FROM events
    WHERE start_date IS NOT NULL
    ORDER BY start_date DESC LIMIT 6;
    """
    with get_conn() as c:
        rows = c.execute(q).fetchall()
    lines = []
    for r in rows:
        dates = f"{(r['start_date'] or '')}–{(r['end_date'] or '')}".strip("-")
        lines.append(f"{r['name'] or '(Event)'} — {r['club'] or ''} {r['state'] or ''} ({dates}) → [Full results]({r['url']})")
    return bullets(lines)

def build_who_shot_where_block():
    q = """
    SELECT e.club as club, COUNT(*) as shooters
    FROM results r
    JOIN events e ON e.url = r.event_url
    GROUP BY e.club
    ORDER BY shooters DESC LIMIT 8;
    """
    with get_conn() as c:
        rows = c.execute(q).fetchall()
    return bullets([f"{r['club'] or '(Club)'}: {r['shooters']} shooters" for r in rows])

def build_leaders_block():
    q = """
    SELECT shooter_name, discipline, score, targets
    FROM results
    WHERE score IS NOT NULL AND targets IS NOT NULL
    ORDER BY CAST(score AS INT)*1.0 / NULLIF(targets,0) DESC
    LIMIT 5;
    """
    with get_conn() as c:
        rows = c.execute(q).fetchall()
    lines = [f"{r['discipline']}: {r['shooter_name']} {r['score']}/{r['targets']}" for r in rows]
    return bullets(lines)

def build_milestones_block():
    # You’ll need lifetime targets for real milestones. As a placeholder:
    return bullets([])
