from __future__ import annotations


class ScienceQAMMEnv:
    def reset(self):  # pragma: no cover
        return "obs"

    def step(self, action):  # pragma: no cover
        return "obs", 0.0, True, {}

