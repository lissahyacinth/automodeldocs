import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class LLMConfig:
    beam_width: int
    description_iterations: int

    @classmethod
    @lru_cache(maxsize=1)
    def from_env(cls) -> "LLMConfig":
        return cls(
            beam_width=int(os.environ.get("BEAM_WIDTH", 3)),
            description_iterations=int(os.environ.get("DESCRIPTION_ITERATIONS", 2)),
        )
