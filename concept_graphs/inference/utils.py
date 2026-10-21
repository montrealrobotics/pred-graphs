from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class NavigationResult:
    found: bool
    obs: Dict[str, Any]  = field(default_factory=dict)
    stats: Dict[int, Any] = field(default_factory=dict)
    is_absent: bool = False
    nav_target: str = ""
