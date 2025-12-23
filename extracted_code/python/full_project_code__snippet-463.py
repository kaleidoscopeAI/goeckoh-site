import sys
import typing

from pip._vendor.tenacity import BaseRetrying
from pip._vendor.tenacity import DoAttempt
from pip._vendor.tenacity import DoSleep
from pip._vendor.tenacity import RetryCallState

from tornado import gen

