from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .utils.phase_execution import run_in_container

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .utils.config import Config


def get_gpu_count(
    *,
    singularity_image: str | None,
    workspace_dir: Path | str,
    workspace_mount: str,
    enable_gpu: bool = True,
) -> int:
    """Get number of available NVIDIA GPUs without running host commands."""
    if not enable_gpu:
        return 0
    if singularity_image:
        try:
            bind_arg = f"{Path(workspace_dir).resolve()}:{workspace_mount}"
            nvidia_smi = run_in_container(
                worker_id=None,
                image_path=singularity_image,
                cmd="nvidia-smi --query-gpu=gpu_name --format=csv,noheader",
                env={},
                binds=[bind_arg],
                use_nv=True,
                pwd=workspace_mount,
            )
            if nvidia_smi.returncode == 0 and nvidia_smi.stdout.strip():
                gpus = nvidia_smi.stdout.strip().split("\n")
                return len(gpus)
        except Exception:
            pass

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        devices = [d for d in cuda_visible_devices.split(",") if d and d != "-1"]
        return len(devices)
    return 0


def _torch_device_count() -> int | None:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return None


@dataclass
class WorkerPlan:
    requested_workers: int
    actual_workers: int
    reasons: list[str]
    gpu_count: int
    torch_device_count: int | None
    cuda_visible_devices: str | None


def resolve_worker_plan(cfg: Any) -> WorkerPlan:
    requested = int(getattr(cfg.agent, "num_workers", 1) or 1)
    reasons: list[str] = []
    if requested <= 0:
        reasons.append("config_fallback")
        requested = 1
    else:
        reasons.append("config_requested")

    use_gpu = bool(getattr(cfg.exec, "use_gpu", True))
    if not use_gpu:
        reasons.append("gpu_disabled_by_config")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES") if use_gpu else None
    torch_count = _torch_device_count() if use_gpu else None
    gpu_count = get_gpu_count(
        singularity_image=getattr(cfg.exec, "singularity_image", None),
        workspace_dir=cfg.workspace_dir,
        workspace_mount=getattr(cfg.exec, "workspace_mount", "/workspace"),
        enable_gpu=use_gpu,
    )
    if use_gpu:
        if gpu_count == 0:
            reasons.append("no_gpu_detected")
        elif gpu_count < requested:
            reasons.append("gpu_count_lt_requested_cpu_workers_allowed")

    return WorkerPlan(
        requested_workers=requested,
        actual_workers=requested,
        reasons=reasons,
        gpu_count=gpu_count,
        torch_device_count=torch_count,
        cuda_visible_devices=cuda_visible_devices,
    )
