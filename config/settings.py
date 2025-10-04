import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
USER_AGENT = os.getenv("USER_AGENT", "ata-auto-blog/0.1 (+contact@example.com)")
BASE_URL = os.getenv("BASE_URL", "https://shootata.com/")
USER_AGENT = "ata-auto-blog/0.1 (+your.email@example.com)"
