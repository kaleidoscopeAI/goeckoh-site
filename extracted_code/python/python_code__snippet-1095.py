from __future__ import annotations

import logging
from argparse import ArgumentParser
from typing import TYPE_CHECKING

from pip._vendor import requests

from pip._vendor.cachecontrol.adapter import CacheControlAdapter
from pip._vendor.cachecontrol.cache import DictCache
from pip._vendor.cachecontrol.controller import logger

