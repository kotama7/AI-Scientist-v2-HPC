import random
from typing import Dict, List, Optional


class BanditModelSelector:
    """Simple Thompson-sampling bandit for picking among multiple code models."""

    def __init__(
        self,
        models: List[Dict],
        *,
        enabled: bool = False,
        exploration: float = 0.1,
    ) -> None:
        self.models = [m.copy() for m in models if m.get("model")]
        self.enabled = enabled and len(self.models) > 1
        self.exploration = exploration
        self._stats: Dict[str, Dict[str, int]] = {
            m["model"]: {"trials": 0, "success": 0} for m in self.models
        }

    @classmethod
    def from_code_config(cls, code_cfg) -> Optional["BanditModelSelector"]:
        """Create a selector from agent.code config."""
        bandit_cfg = getattr(code_cfg, "bandit", None)
        enabled = getattr(bandit_cfg, "enabled", False) if bandit_cfg else False
        exploration = (
            getattr(bandit_cfg, "exploration", 0.1) if bandit_cfg else 0.1
        )

        models_cfg = getattr(code_cfg, "models", None)
        models: List[Dict] = []
        if models_cfg:
            for entry in models_cfg:
                # Support both dicts and OmegaConf-style objects
                model_name = entry.get("model") if hasattr(entry, "get") else getattr(entry, "model", None)
                if not model_name:
                    continue
                models.append(
                    {
                        "model": model_name,
                        "temp": entry.get("temp") if hasattr(entry, "get") else getattr(entry, "temp", None),
                        "max_tokens": entry.get("max_tokens") if hasattr(entry, "get") else getattr(entry, "max_tokens", None),
                    }
                )

        if not models:
            fallback_model = getattr(code_cfg, "model", None) if code_cfg else None
            if fallback_model is None:
                return None
            models = [
                {
                    "model": fallback_model,
                    "temp": getattr(code_cfg, "temp", None) if code_cfg else None,
                    "max_tokens": getattr(code_cfg, "max_tokens", None) if code_cfg else None,
                }
            ]

        # If only one model is provided and bandit is disabled, return None to keep default behaviour.
        selector = cls(models, enabled=enabled, exploration=exploration)
        return selector if selector.enabled else None

    def select(self, default_model: str, default_temp: float) -> Dict:
        """Select a model configuration to use."""
        if not self.enabled:
            return {"model": default_model, "temp": default_temp}

        if random.random() < self.exploration:
            choice = random.choice(self.models)
        else:
            choice = max(self.models, key=lambda m: self._sample_score(m["model"]))

        temp = choice.get("temp", default_temp)
        return {
            "model": choice["model"],
            "temp": temp,
            "max_tokens": choice.get("max_tokens"),
        }

    def record_generation(self, model_name: str) -> None:
        """Track that a model produced a runnable node."""
        if not model_name:
            return
        self._stats.setdefault(model_name, {"trials": 0, "success": 0})
        self._stats[model_name]["trials"] += 1

    def record_selection(self, model_name: str) -> None:
        """Track that a node from the given model was selected as best."""
        if not model_name:
            return
        self._stats.setdefault(model_name, {"trials": 0, "success": 0})
        self._stats[model_name]["success"] += 1

    def selection_rates(self) -> Dict[str, float]:
        """Return observed selection rates per model."""
        rates: Dict[str, float] = {}
        for model_name, stat in self._stats.items():
            trials = stat["trials"]
            rates[model_name] = stat["success"] / trials if trials else 0.0
        return rates

    def _sample_score(self, model_name: str) -> float:
        stats = self._stats.get(model_name, {"trials": 0, "success": 0})
        failures = max(stats["trials"] - stats["success"], 0)
        return random.betavariate(stats["success"] + 1, failures + 1)
