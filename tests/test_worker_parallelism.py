import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_scientist.treesearch.worker_plan import resolve_worker_plan


class TestWorkerParallelism(unittest.TestCase):
    def _make_cfg(self, *, num_workers: int, phase_mode: str = "single"):
        return SimpleNamespace(
            agent=SimpleNamespace(num_workers=num_workers),
            exec=SimpleNamespace(
                singularity_image=None,
                workspace_mount="/workspace",
                phase_mode=phase_mode,
            ),
            workspace_dir=Path("/tmp/ai_scientist_test_workspace"),
        )

    def test_worker_count_respects_config_with_single_gpu(self) -> None:
        cfg = self._make_cfg(num_workers=4)
        with patch(
            "ai_scientist.treesearch.worker_plan.get_gpu_count", return_value=1
        ), patch(
            "ai_scientist.treesearch.worker_plan._torch_device_count", return_value=4
        ):
            plan = resolve_worker_plan(cfg)
        self.assertEqual(plan.requested_workers, 4)
        self.assertEqual(plan.actual_workers, 4)
        self.assertIn("gpu_count_lt_requested_cpu_workers_allowed", plan.reasons)

    def test_worker_count_split_mode_not_rounded_down(self) -> None:
        cfg = self._make_cfg(num_workers=4, phase_mode="split")
        with patch(
            "ai_scientist.treesearch.worker_plan.get_gpu_count", return_value=1
        ):
            plan = resolve_worker_plan(cfg)
        self.assertEqual(plan.actual_workers, 4)


if __name__ == "__main__":
    unittest.main()
