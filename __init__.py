from __future__ import annotations

from importlib import import_module
from typing import Dict

SUPPORTED_BASELINES = [
    "simtact_force",
    "tool_force",
    "vtf",
    "tegtrack",
    "neuralfeels",
    "tac2pose",
    "tactile_ekf",
    "genpose2",
    "ag_pose",
]

MODULES = {
    "simtact_force": f"{__name__}.simtact_force",
    "tool_force": f"{__name__}.tool_force",
    "vtf": f"{__name__}.vtf",
    "tegtrack": f"{__name__}.tegtrack",
    "neuralfeels": f"{__name__}.neuralfeels",
    "tac2pose": f"{__name__}.tac2pose",
    "tactile_ekf": f"{__name__}.tactile_ekf",
    "genpose2": f"{__name__}.genpose2",
    "ag_pose": f"{__name__}.ag_pose",
}


def load_registry() -> Dict[str, object]:
    return {name: import_module(module_path) for name, module_path in MODULES.items()}




