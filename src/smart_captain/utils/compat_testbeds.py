"""Small testbed classes used to verify migration adapters."""

from __future__ import annotations


class DummyLegacyTaskEnv:
    """Tiny stand-in for legacy task env classes used in adapter tests."""

    def __init__(self, env_config, shared_auv, train_mode):
        self.env_config = env_config
        self.shared_auv = shared_auv
        self.train_mode = train_mode
        self.synced_from = None

    def reset(self, seed=None, return_info=True, options=None):
        return {"seed": seed, "env": id(self)}, {"return_info": return_info, "options": options}

    def step(self, action):
        return {"action": action, "env": id(self)}, 0.0, True, False, {"done": True}

    def sync_state_from(self, other):
        self.synced_from = other
