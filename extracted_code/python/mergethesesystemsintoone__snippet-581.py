from abc import ABC, abstractmethod

from typing import Dict, Any, Optional

import logging

from queue import Queue

import threading

import time

from dataclasses import dataclass




class NodeState:










class AnalysisNode(ABC):





    

    def __init__(self, node_id: str, node_type: str):




        

