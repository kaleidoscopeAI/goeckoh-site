import asyncio
from .hw_simulator import HWSimulator
from .web_crawler import SafeCrawler
from .audit import log_action
from .config import settings
class ControlMapper:
def __init__(self, crawler: SafeCrawler, hw: HWSimulator):
