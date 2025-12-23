import aiohttp
import asyncio
import time
from urllib.parse import urlparse
import urllib.robotparser
from pathlib import Path
from typing import Optional
from .audit import log_action
class SafeCrawler:
def __init__(self, allowlist=None, max_concurrent=4, per_domain_delay=1.0, sandbox_dir='./data/web_crawl'):
