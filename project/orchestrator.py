"""
Orchestrator for existing EchoVoice bytecode modules.

This does not modify any source; it loads compiled modules as components,
binds their callable surfaces, and exposes a simple async call router.
Add new components by extending `default_descriptors()`.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


# Ensure the package can be imported as `echovoice.*`
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


@dataclass(slots=True)
class ComponentDescriptor:
    name: str
    module: str
    target: str  # dotted path inside the module, e.g. "SimulatedSpeechLoop.run"
    async_call: bool = False  # force async even if not declared as coroutine
    use_instance: bool = True  # if target starts with a class, bind an instance
    init_kwargs: Dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """Loads components from descriptors and routes calls by name."""

    def __init__(self, descriptors: Iterable[ComponentDescriptor]):
        self._modules: Dict[str, Any] = {}
        self._instances: Dict[str, Any] = {}
        self._functions: Dict[str, Tuple[Callable[..., Any], bool]] = {}
        self._errors: Dict[str, str] = {}
        for desc in descriptors:
            self._bind(desc)

    def _load_module(self, module: str):
        if module in self._modules:
            return self._modules[module]
        try:
            mod = importlib.import_module(module)
        except Exception as e:  # pragma: no cover - defensive logging
            self._errors[module] = f"import failed: {e}"
            raise
        self._modules[module] = mod
        return mod

    def _bind(self, desc: ComponentDescriptor) -> None:
        try:
            mod = self._load_module(desc.module)
            parts = desc.target.split(".")
            obj = getattr(mod, parts[0])
            # If the first part is a class and methods are requested, bind an instance.
            if inspect.isclass(obj) and len(parts) > 1 and desc.use_instance:
                key = f"{desc.module}.{parts[0]}"
                if key not in self._instances:
                    self._instances[key] = obj(**desc.init_kwargs)
                obj = self._instances[key]
                for part in parts[1:]:
                    obj = getattr(obj, part)
            else:
                for part in parts[1:]:
                    obj = getattr(obj, part)
            is_async = desc.async_call or inspect.iscoroutinefunction(obj)
            self._functions[desc.name] = (obj, is_async)
        except Exception as e:  # pragma: no cover - defensive logging
            self._errors[desc.name] = f"bind failed: {e}"

    @property
    def errors(self) -> Dict[str, str]:
        return dict(self._errors)

    def list_components(self) -> List[str]:
        return sorted(self._functions.keys())

    async def call(self, name: str, *args, **kwargs) -> Any:
        if name not in self._functions:
            raise KeyError(f"component '{name}' not registered")
        fn, is_async = self._functions[name]
        if is_async:
            return await fn(*args, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def default_descriptors() -> List[ComponentDescriptor]:
    """Starter component set covering key runtime surfaces."""
    return [
        # Lightweight simulated loop (no microphone/TTS dependency).
        ComponentDescriptor(
            name="sim_loop.run",
            module="echovoice.speech_loop",
            target="SimulatedSpeechLoop.run",
            async_call=True,
        ),
        ComponentDescriptor(
            name="sim_loop.mutate",
            module="echovoice.speech_loop",
            target="SimulatedSpeechLoop._maybe_mutate_phrase",
            use_instance=False,
        ),
        # Core safety/metrics helpers.
        ComponentDescriptor(
            name="heart.enforce_first_person",
            module="echovoice.heart_core",
            target="enforce_first_person",
        ),
        ComponentDescriptor(
            name="events.now_ts",
            module="echovoice.events",
            target="now_ts",
        ),
        # Organic seed snapshot for monitoring.
        ComponentDescriptor(
            name="seed.snapshot",
            module="echovoice.seed",
            target="Seed.snapshot",
        ),
        # Trigger matching API.
        ComponentDescriptor(
            name="triggers.match",
            module="echovoice.trigger_engine",
            target="TriggerEngine.match_for_audio",
        ),
        # Gear fabric snapshot for debugging topology.
        ComponentDescriptor(
            name="gears.snapshot",
            module="echovoice.gears",
            target="GearFabric.snapshot",
            use_instance=False,
        ),
    ]


async def _demo() -> None:
    """Demonstration: list components and run the simulated loop once."""
    registry = ComponentRegistry(default_descriptors())
    if registry.errors:
        print("Descriptor bind errors:")
        for k, v in registry.errors.items():
            print(f"  {k}: {v}")
    print("Components:", ", ".join(registry.list_components()))
    # Run one iteration of the simulated loop (will loop forever; cancel quickly in real use).
    sim_task = asyncio.create_task(registry.call("sim_loop.run"))
    await asyncio.sleep(0.1)
    sim_task.cancel()
    try:
        await sim_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(_demo())
