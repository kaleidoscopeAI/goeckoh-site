import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import components from previous scripts
from unravel_ai_core_engine import SoftwareAnalyzer, FileAnalyzer, Decompiler, DependencyAnalyzer
from kaleidoscope_controller import KaleidoscopeController, ComponentManager, OptimizedTaskScheduler
from llm_service import get_llm_service, LLMMessage
from text_node import TextNode, NodeConfig
from pattern_recognition import PatternRecognition
from system_upgrade_module import SystemUpgrader, UpgradeConfig, LanguageType, UpgradeStrategy

