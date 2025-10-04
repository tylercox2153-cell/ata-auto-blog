# ATA Auto Blog

A subscription blog that automatically generates weekly Amateur Trapshooting Association (ATA) results, averages, and “who shot where” summaries from ShootATA data.

## Features (Planned)
- Scrape official ATA results and averages
- Generate weekly digest (stats, summaries, highlights)
- Auto-publish to blog every Monday
- Subscription support with Stripe
- Email notification for subscribers

## Tech Stack
- **Backend**: Python (Requests, BeautifulSoup/Playwright)
- **Database**: PostgreSQL
- **Frontend**: Next.js (or Substack for quick launch)
- **Payments**: Stripe
- **Hosting**: Vercel + Railway/Render
- **Automation**: Scheduled cron jobs

## Project Roadmap
- [ ] Setup repo and environment
- [ ] Build scraper (ATA pages → DB)
- [ ] Create weekly aggregator script
- [ ] Generate draft blog posts
- [ ] Publish to frontend with paywall
- [ ] Add Stripe subscriptions
- [ ] Launch beta with testers

## License
MIT
