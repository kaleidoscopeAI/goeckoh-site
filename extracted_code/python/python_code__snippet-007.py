import functools
import sys
import typing as t
from asyncio import sleep

from pip._vendor.tenacity import AttemptManager
from pip._vendor.tenacity import BaseRetrying
from pip._vendor.tenacity import DoAttempt
from pip._vendor.tenacity import DoSleep
from pip._vendor.tenacity import RetryCallState

