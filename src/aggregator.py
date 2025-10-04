from datetime import date, timedelta

def week_bounds(d: date):
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)          # Sunday
    return start, end

def build_weekly_context():
    """Stub – later this will query the DB."""
    start, _ = week_bounds(date.today())
    return {
        "week_start": start.isoformat(),
        "top_shoots_block": "- (sample) Shoot A — Club X, ST (dates)\n",
        "leaders_block": "- (sample) Singles: Jane D. 100/100 (99.2%)\n",
        "who_shot_where_block": "- (sample) Club X: 34 shooters; Club Y: 18\n",
        "milestones_block": "- (sample) John S. passed 25,000 lifetime targets\n",
    }

if __name__ == "__main__":
    print(build_weekly_context())
