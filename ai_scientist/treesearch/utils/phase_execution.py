import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger("ai-scientist")


class CommandExecutionError(RuntimeError):
    """Raised when a command executed inside the worker environment fails."""

    def __init__(self, message: str, *, returncode: int | None = None, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def detect_container_runtime(preferred: str | None = None) -> str | None:
    """Detect available container runtime. Only Singularity is allowed."""
    if preferred and preferred != "singularity":
        logger.warning("Singularity is required; ignoring runtime preference %s", preferred)
    if shutil.which("singularity"):
        return "singularity"
    return None


def summarize_text(text: str, *, max_lines: int = 20, max_chars: int = 2000) -> str:
    """Summarize command output for LLM consumption (full output is logged separately)."""
    if not text:
        return ""
    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines > max_lines:
        lines = lines[-max_lines:]
        text = f"... ({total_lines - max_lines} lines truncated) ...\n" + "\n".join(lines)
    else:
        text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
        text = f"... (truncated to last {max_chars} chars) ...\n{text}"
    return text


def summarize_command_output(stdout: str, stderr: str, *, max_lines: int = 20, max_chars: int = 2000) -> dict[str, str]:
    return {
        "stdout": summarize_text(stdout, max_lines=max_lines, max_chars=max_chars),
        "stderr": summarize_text(stderr, max_lines=max_lines, max_chars=max_chars),
    }


def run_in_container(
    *,
    worker_id: int | None,
    image_path: Path | str,
    cmd: str | Sequence[str],
    env: Mapping[str, str] | None,
    binds: Sequence[str] | None,
    use_nv: bool,
    pwd: str | None = None,
    extra_args: Sequence[str] | None = None,
    runtime: str = "singularity",
) -> subprocess.CompletedProcess[str]:
    """Run a single command inside a Singularity container."""
    if runtime != "singularity":
        raise CommandExecutionError(f"Unsupported runtime: {runtime}")
    if not shutil.which(runtime):
        raise CommandExecutionError("Singularity runtime not found.")
    if not image_path:
        raise CommandExecutionError("Container image path is required.")

    exec_cmd: list[str] = [runtime, "exec"]
    if use_nv:
        exec_cmd.append("--nv")
    if extra_args:
        exec_cmd.extend(list(extra_args))
    if binds:
        for bind in binds:
            exec_cmd.extend(["--bind", bind])
    if pwd:
        exec_cmd.extend(["--pwd", pwd])
    if env:
        for key, value in env.items():
            exec_cmd.extend(["--env", f"{key}={value}"])

    exec_cmd.append(str(image_path))
    if isinstance(cmd, str):
        exec_cmd.extend(["bash", "-lc", cmd])
    else:
        exec_cmd.extend([str(part) for part in cmd])

    return subprocess.run(exec_cmd, capture_output=True, text=True)


class ExecutionEnvironment:
    """Thin wrapper to run commands inside a Singularity instance (host exec disabled)."""

    def __init__(
        self,
        workspace: Path,
        image: str | None,
        *,
        runtime_preference: str | None = None,
        workspace_mount: str = "/workspace",
        gpu_id: int | None = None,
        instance_name: str | None = None,
        enable_writable_tmpfs: bool = True,
        overlay_path: str | None = None,
        extra_start_args: Sequence[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        if isinstance(image, str) and not image.strip():
            image = None
        self.image = Path(image).resolve() if image else None
        self.workspace_mount = workspace_mount
        self.gpu_id = gpu_id
        self.runtime = detect_container_runtime(runtime_preference) if image else None
        self.instance_name: str | None = instance_name
        self._stopped = False
        self._started = False
        self._use_instance = str(os.environ.get("AI_SCIENTIST_USE_INSTANCE", "")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.enable_writable_tmpfs = enable_writable_tmpfs
        self.overlay_path = Path(overlay_path).resolve() if overlay_path else None
        self.extra_start_args = list(extra_start_args) if extra_start_args else []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    @property
    def using_container(self) -> bool:
        return bool(self.runtime and self.instance_name and self._started and self._use_instance)

    def _build_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(extra) if extra else {}
        env.setdefault("PATH", os.environ.get("SINGULARITYENV_PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"))
        if self.gpu_id is not None:
            env.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_id))
        return env

    def start(self) -> None:
        if self._started or self._stopped:
            return
        if not self._use_instance:
            logger.info("AI_SCIENTIST_USE_INSTANCE not set; skipping instance start and using direct singularity exec.")
            return

        if not self.image:
            logger.info("No container image configured; instance start skipped.")
            return
        if not self.image.exists():
            logger.warning("Container image %s not found; instance start skipped.", self.image)
            return
        if not self.runtime:
            logger.warning("No Singularity runtime found; instance start skipped.")
            return

        if not self.instance_name:
            self.instance_name = f"ai-sci-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        bind_arg = f"{self.workspace}:{self.workspace_mount}"
        env_vars = self._build_env()
        env_args: list[str] = []
        for key, value in env_vars.items():
            env_args.extend(["--env", f"{key}={value}"])

        cmd = [self.runtime, "instance", "start", "--nv", *self.extra_start_args]
        if self.enable_writable_tmpfs:
            cmd.append("--writable-tmpfs")
        if self.overlay_path:
            cmd.extend(["--overlay", f"{self.overlay_path}:rw"])
        cmd.extend(env_args)
        cmd.extend(["--bind", bind_arg, str(self.image), self.instance_name])

        logger.info("Starting %s instance %s with image %s", self.runtime, self.instance_name, self.image)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.instance_name = None
            self._use_instance = False
            logger.warning(
                "Failed to start %s instance; will fallback to direct singularity exec. Details: %s",
                self.runtime,
                result.stderr.strip() or result.stdout.strip(),
            )
            return
        self._started = True

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if not self.runtime or not self.instance_name or not self._started:
            return
        cmd = [self.runtime, "instance", "stop", self.instance_name]
        logger.info("Stopping %s instance %s", self.runtime, self.instance_name)
        subprocess.run(cmd, capture_output=True, text=True)
        self.instance_name = None
        self._started = False

    def _map_cwd(self, cwd: Path | None) -> str | None:
        if cwd is None:
            return self.workspace_mount
        cwd = Path(cwd).resolve()
        try:
            rel = cwd.relative_to(self.workspace)
        except ValueError:
            return self.workspace_mount
        return str(Path(self.workspace_mount) / rel)

    def run(
        self,
        command: str | Iterable[str],
        *,
        cwd: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env_vars = self._build_env(extra_env)
        if not (self.runtime and self.image):
            raise CommandExecutionError("Container runtime not available; host execution is disabled.")

        container_cwd = self._map_cwd(cwd)
        bind_arg = f"{self.workspace}:{self.workspace_mount}"
        image_ref = f"instance://{self.instance_name}" if self.using_container else str(self.image)
        result = run_in_container(
            worker_id=None,
            image_path=image_ref,
            cmd=command if isinstance(command, str) else list(command),
            env=env_vars,
            binds=[bind_arg],
            use_nv=True,
            pwd=container_cwd,
            runtime=self.runtime,
        )

        if check and result.returncode != 0:
            raise CommandExecutionError(
                f"Command failed with code {result.returncode}",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result


class SingularityWorkerContainer:
    """Prepare per-worker Singularity images with Phase 1 persistence."""

    def __init__(
        self,
        *,
        base_image: Path | str | None,
        run_root: Path,
        workspace: Path,
        workspace_mount: str,
        worker_id: int | None = None,
        per_worker_sif: bool = True,
        keep_sandbox: bool = False,
        use_fakeroot: bool = True,
        writable_mode: str = "auto",
        enable_writable_tmpfs: bool = True,
        overlay_path: Path | str | None = None,
    ) -> None:
        self.runtime = detect_container_runtime("singularity")
        if isinstance(base_image, str) and not base_image.strip():
            base_image = None
        self.base_image = Path(base_image).resolve() if base_image else None
        self.workspace = workspace.resolve()
        self.workspace_mount = workspace_mount
        worker_label = f"worker-{worker_id if worker_id is not None else 0}"
        if not per_worker_sif:
            worker_label = "worker-shared"
        self.container_root = run_root / "workers" / worker_label / "container"
        self.container_root.mkdir(parents=True, exist_ok=True)
        self.base_copy = self.container_root / "base.sif"
        self.sandbox_dir = self.container_root / f"{worker_label}.sandbox"
        self.worker_sif = self.container_root / f"{worker_label}.sif"
        self.build_log = self.container_root / "build_container.log"
        self.keep_sandbox = keep_sandbox
        self.use_fakeroot = use_fakeroot
        self.writable_mode = writable_mode if writable_mode in {"auto", "tmpfs", "overlay", "none"} else "auto"
        self.enable_writable_tmpfs = enable_writable_tmpfs
        self.overlay_path = Path(overlay_path).resolve() if overlay_path else None

    def _log(self, message: str, extra_logs: Sequence[Path] | None = None) -> None:
        logs = [self.build_log]
        if extra_logs:
            logs.extend(extra_logs)
        for log_file in logs:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as fh:
                fh.write(message)

    def _write_command_result(
        self,
        *,
        result: subprocess.CompletedProcess[str],
        cmd_repr: str,
        step: str,
        extra_logs: Sequence[Path] | None = None,
    ) -> None:
        self._log(f"[{step}] $ {cmd_repr}\n", extra_logs)
        self._log(f"[{step}] exit_code={result.returncode}\n", extra_logs)
        if result.stdout:
            self._log(result.stdout, extra_logs)
        if result.stderr:
            self._log(result.stderr, extra_logs)

    def _ensure_base_copy(self) -> None:
        if not self.base_image:
            return
        if self.base_copy.exists() or self.base_copy.is_symlink():
            try:
                self.base_copy.unlink()
            except OSError:
                pass
        try:
            self.base_copy.symlink_to(self.base_image)
        except OSError:
            shutil.copy2(self.base_image, self.base_copy)

    def _cleanup_existing(self) -> None:
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        if self.worker_sif.exists():
            try:
                self.worker_sif.unlink()
            except OSError:
                pass

    def _base_exec_flags(self) -> list[str]:
        flags: list[str] = []
        if self.writable_mode == "none":
            return flags
        if self.writable_mode in {"auto", "tmpfs"} and self.enable_writable_tmpfs:
            flags.append("--writable-tmpfs")
        elif self.overlay_path and self.writable_mode in {"auto", "overlay"}:
            flags.extend(["--overlay", f"{self.overlay_path}:rw"])
        return flags

    def _phase1_prelude(self) -> str:
        return (
            "mkdir -p /var/lib/apt/lists/partial || true; "
            "chmod 755 /var/lib/apt/lists /var/lib/apt/lists/partial || true"
        )

    def _exec_commands(
        self,
        image: Path,
        commands: Sequence[str | Sequence[str]],
        *,
        writable: bool = False,
        use_tmpfs: bool = False,
        use_overlay: bool = False,
        step: str,
        extra_env: Mapping[str, str] | None,
        log_files: Sequence[Path],
        stop_on_failure: bool = True,
    ) -> tuple[bool, list[str], dict | None]:
        outputs: list[str] = []
        env_vars = {"DEBIAN_FRONTEND": "noninteractive"}
        if extra_env:
            env_vars.update(extra_env)

        common_flags: list[str] = []
        if self.use_fakeroot:
            common_flags.append("--fakeroot")
        if writable:
            common_flags.append("--writable")
        elif use_tmpfs:
            common_flags.append("--writable-tmpfs")
        if use_overlay and self.overlay_path:
            common_flags.extend(["--overlay", f"{self.overlay_path}:rw"])

        bind_arg = f"{self.workspace}:{self.workspace_mount}"

        # Prepend a small prelude to avoid apt failures on missing partial dir
        full_commands: list[str] = [self._phase1_prelude()]
        for raw_cmd in commands or []:
            full_commands.append(" ".join(raw_cmd) if isinstance(raw_cmd, (list, tuple)) else str(raw_cmd))

        failure: dict | None = None
        for shell_cmd in full_commands:
            result = run_in_container(
                worker_id=None,
                image_path=image,
                cmd=shell_cmd,
                env=env_vars,
                binds=[bind_arg],
                use_nv=bool(env_vars.get("CUDA_VISIBLE_DEVICES")),
                pwd=self.workspace_mount,
                extra_args=common_flags,
                runtime=self.runtime,
            )
            self._write_command_result(result=result, cmd_repr=shell_cmd, step=step, extra_logs=log_files)
            summary = summarize_command_output(result.stdout, result.stderr)
            summary_blob = (
                f"[{step}] $ {shell_cmd}\n"
                f"exit_code={result.returncode}\n"
                f"stdout:\n{summary['stdout']}\n"
                f"stderr:\n{summary['stderr']}\n"
            )
            outputs.append(summary_blob)
            if result.returncode != 0:
                failure = failure or {
                    "step": step,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                }
                self._log(f"[{step}] failed with code {result.returncode}\n", log_files)
                if stop_on_failure:
                    return False, outputs, failure
        return failure is None, outputs, failure

    def _run_iterative_phase1(
        self,
        image: Path,
        *,
        step: str,
        iterative_driver: Callable[[list[dict[str, Any]], int, int], dict[str, Any]],
        max_steps: int,
        extra_env: Mapping[str, str] | None,
        log_files: Sequence[Path],
        use_tmpfs: bool,
        use_overlay: bool,
    ) -> tuple[bool, list[str], dict | None, list[str]]:
        outputs: list[str] = []
        history: list[dict[str, Any]] = []
        commands_executed: list[str] = []
        env_vars = {"DEBIAN_FRONTEND": "noninteractive"}
        if extra_env:
            env_vars.update(extra_env)
        bind_arg = f"{self.workspace}:{self.workspace_mount}"

        common_flags: list[str] = []
        if self.use_fakeroot:
            common_flags.append("--fakeroot")
        if use_tmpfs:
            common_flags.append("--writable-tmpfs")
        if use_overlay and self.overlay_path:
            common_flags.extend(["--overlay", f"{self.overlay_path}:rw"])

        fallback_flags: list[str] = []
        if self.use_fakeroot:
            fallback_flags.append("--fakeroot")

        prelude_cmd = self._phase1_prelude()
        prelude_res = run_in_container(
            worker_id=None,
            image_path=image,
            cmd=prelude_cmd,
            env=env_vars,
            binds=[bind_arg],
            use_nv=bool(env_vars.get("CUDA_VISIBLE_DEVICES")),
            pwd=self.workspace_mount,
            extra_args=common_flags,
            runtime=self.runtime,
        )
        self._write_command_result(result=prelude_res, cmd_repr=prelude_cmd, step=f"{step}-prelude", extra_logs=log_files)

        last_returncode: int | None = None
        for idx in range(1, max_steps + 1):
            try:
                response = iterative_driver(history, idx, max_steps)
            except Exception as exc:
                failure = {"step": step, "message": f"Phase 1 iterative driver failed: {exc}"}
                self._log(f"[{step}] iterative driver error: {exc}\n", log_files)
                return False, outputs, failure, commands_executed
            command = str(response.get("command", "")).strip()
            done = bool(response.get("done", False))
            if done:
                if last_returncode not in (None, 0):
                    failure = {
                        "step": step,
                        "message": "Phase 1 marked done but last command failed.",
                        "returncode": last_returncode,
                    }
                    self._log(f"[{step}] done=true after failure\n", log_files)
                    return False, outputs, failure, commands_executed
                return True, outputs, None, commands_executed
            if not command:
                failure = {"step": step, "message": "Phase 1 command missing from LLM response."}
                self._log(f"[{step}] missing command from iterative driver\n", log_files)
                return False, outputs, failure, commands_executed

            result = run_in_container(
                worker_id=None,
                image_path=image,
                cmd=command,
                env=env_vars,
                binds=[bind_arg],
                use_nv=bool(env_vars.get("CUDA_VISIBLE_DEVICES")),
                pwd=self.workspace_mount,
                extra_args=common_flags,
                runtime=self.runtime,
            )
            stderr_lower = (result.stderr or "").lower()
            if result.returncode != 0 and ("fuse" in stderr_lower or "allow_other" in stderr_lower) and common_flags != fallback_flags:
                self._log(f"[{step}] retrying without tmpfs/overlay due to FUSE/allow_other error\n", log_files)
                result = run_in_container(
                    worker_id=None,
                    image_path=image,
                    cmd=command,
                    env=env_vars,
                    binds=[bind_arg],
                    use_nv=bool(env_vars.get("CUDA_VISIBLE_DEVICES")),
                    pwd=self.workspace_mount,
                    extra_args=fallback_flags,
                    runtime=self.runtime,
                )
                common_flags = fallback_flags

            self._write_command_result(result=result, cmd_repr=command, step=step, extra_logs=log_files)
            summary = summarize_command_output(result.stdout, result.stderr)
            summary_blob = (
                f"[{step}] $ {command}\n"
                f"exit_code={result.returncode}\n"
                f"stdout:\n{summary['stdout']}\n"
                f"stderr:\n{summary['stderr']}\n"
            )
            outputs.append(summary_blob)
            history.append(
                {
                    "step": idx,
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout_summary": summary["stdout"],
                    "stderr_summary": summary["stderr"],
                }
            )
            commands_executed.append(command)
            last_returncode = result.returncode

        failure = {
            "step": step,
            "message": "Phase 1 hit max_steps without done=true.",
            "max_steps": max_steps,
        }
        self._log(f"[{step}] max_steps reached without done\n", log_files)
        return False, outputs, failure, commands_executed

    def prepare_phase1(
        self,
        download_commands: Sequence[str | Sequence[str]],
        *,
        workspace: Path,
        workspace_mount: str,
        download_log: Path,
        extra_env: Mapping[str, str] | None = None,
        iterative_driver: Callable[[list[dict[str, Any]], int, int], dict[str, Any]] | None = None,
        max_steps: int = 12,
    ) -> tuple[bool, list[str], dict | None]:
        """Run Phase 1 using base.sif first, persist to sandbox, and build worker SIF."""
        outputs: list[str] = []
        log_files = [download_log]
        self.workspace = workspace.resolve()
        self.workspace_mount = workspace_mount
        phase1_env = dict(extra_env) if extra_env else None
        if phase1_env and "CUDA_VISIBLE_DEVICES" in phase1_env:
            # Phase 1 doesn't need GPU access; skip --nv to avoid noisy warnings.
            phase1_env.pop("CUDA_VISIBLE_DEVICES", None)
        if not self.runtime:
            msg = "Singularity runtime not available; cannot prepare worker container."
            self._log(msg + "\n", log_files)
            return False, outputs + [msg], {"message": msg}
        if not self.base_image or not self.base_image.exists():
            msg = f"Base image not found: {self.base_image}"
            self._log(msg + "\n", log_files)
            return False, outputs + [msg], {"message": msg}

        self._cleanup_existing()
        self._ensure_base_copy()
        if not self.base_copy.exists():
            msg = f"Base image copy missing: {self.base_copy}"
            self._log(msg + "\n", log_files)
            return False, outputs + [msg], {"message": msg}

        if iterative_driver and max_steps < 1:
            msg = "Phase 1 iterative loop requires max_steps >= 1."
            self._log(msg + "\n", log_files)
            return False, outputs + [msg], {"message": msg}

        # Step 0-1: sandbox build
        build_cmd = [self.runtime, "build", "--force"]
        if self.use_fakeroot:
            build_cmd.append("--fakeroot")
        build_cmd.extend(["--sandbox", str(self.sandbox_dir), str(self.base_copy)])
        build_res = subprocess.run(build_cmd, capture_output=True, text=True)
        self._write_command_result(result=build_res, cmd_repr=" ".join(build_cmd), step="sandbox-build", extra_logs=log_files)
        if build_res.returncode != 0:
            failure = {
                "step": "sandbox-build",
                "returncode": build_res.returncode,
                "stderr": build_res.stderr,
            }
            self._log("[sandbox-build] failed while creating sandbox\n", log_files)
            return False, outputs + [build_res.stderr], failure

        commands_to_apply: Sequence[str | Sequence[str]] = download_commands

        # Step 0-2: run Phase 1 on base.sif (non-persistent)
        tmpfs_flags = self._base_exec_flags()
        use_tmpfs = "--writable-tmpfs" in tmpfs_flags
        use_overlay = any(flag.startswith("--overlay") for flag in tmpfs_flags)
        self._log(f"[phase1-base] image={self.base_copy}\n", log_files)
        if iterative_driver:
            base_success, base_outputs, base_failure, commands_to_apply = self._run_iterative_phase1(
                self.base_copy,
                step="phase1-base",
                iterative_driver=iterative_driver,
                max_steps=max_steps,
                extra_env=phase1_env,
                log_files=log_files,
                use_tmpfs=use_tmpfs,
                use_overlay=use_overlay,
            )
            outputs.extend(base_outputs)
            if not base_success:
                return False, outputs, base_failure
        else:
            base_success, base_outputs, base_failure = self._exec_commands(
                self.base_copy,
                download_commands,
                writable=False,
                use_tmpfs=use_tmpfs,
                use_overlay=use_overlay,
                step="phase1-base",
                extra_env=phase1_env,
                log_files=log_files,
            )
            outputs.extend(base_outputs)
            if not base_success:
                stderr = base_failure.get("stderr", "") if base_failure else ""
                if "fuse" in stderr.lower() or "allow_other" in stderr.lower():
                    self._log("[phase1-base] retrying without tmpfs/overlay due to FUSE/allow_other error\n", log_files)
                    base_success, base_outputs, base_failure = self._exec_commands(
                        self.base_copy,
                        download_commands,
                        writable=False,
                        use_tmpfs=False,
                        use_overlay=False,
                        step="phase1-base-retry",
                        extra_env=phase1_env,
                        log_files=log_files,
                    )
                    outputs.extend(base_outputs)
                if not base_success and ("read-only file system" in stderr.lower() or "read only file system" in stderr.lower()):
                    self._log("[phase1-base] continuing despite read-only filesystem; changes will be applied in sandbox\n", log_files)
                    base_success = True
                if not base_success:
                    return False, outputs, base_failure

        # Step 0-3: apply Phase 1 to sandbox (persistent)
        self._log(f"[phase1-sandbox] image={self.sandbox_dir}\n", log_files)
        if iterative_driver:
            sandbox_success, sandbox_outputs, sandbox_failure = self._exec_commands(
                self.sandbox_dir,
                commands_to_apply,
                writable=True,
                use_tmpfs=False,
                use_overlay=False,
                step="phase1-sandbox",
                extra_env=phase1_env,
                log_files=log_files,
                stop_on_failure=False,
            )
            outputs.extend(sandbox_outputs)
            if not sandbox_success:
                return False, outputs, sandbox_failure
        else:
            sandbox_success, sandbox_outputs, sandbox_failure = self._exec_commands(
                self.sandbox_dir,
                download_commands,
                writable=True,
                use_tmpfs=False,
                use_overlay=False,
                step="phase1-sandbox",
                extra_env=phase1_env,
                log_files=log_files,
            )
            outputs.extend(sandbox_outputs)
            if not sandbox_success:
                stderr = sandbox_failure.get("stderr", "") if sandbox_failure else ""
                if "fuse" in stderr.lower() or "allow_other" in stderr.lower():
                    self._log("[phase1-sandbox] retrying without overlay/tmpfs due to FUSE/allow_other error\n", log_files)
                    sandbox_success, sandbox_outputs, sandbox_failure = self._exec_commands(
                        self.sandbox_dir,
                        download_commands,
                        writable=True,
                        use_tmpfs=False,
                        use_overlay=False,
                        step="phase1-sandbox-retry",
                        extra_env=phase1_env,
                        log_files=log_files,
                    )
                    outputs.extend(sandbox_outputs)
                if not sandbox_success:
                    return False, outputs, sandbox_failure

        # Step 0-4: build worker.sif
        build_worker_cmd = [self.runtime, "build", "--force"]
        if self.use_fakeroot:
            build_worker_cmd.append("--fakeroot")
        build_worker_cmd.extend([str(self.worker_sif), str(self.sandbox_dir)])
        worker_res = subprocess.run(build_worker_cmd, capture_output=True, text=True)
        self._write_command_result(
            result=worker_res,
            cmd_repr=" ".join(build_worker_cmd),
            step="worker-sif",
            extra_logs=log_files,
        )
        if worker_res.returncode != 0:
            failure = {
                "step": "worker-sif",
                "returncode": worker_res.returncode,
                "stderr": worker_res.stderr,
            }
            self._log("[worker-sif] failed while building worker image\n", log_files)
            return False, outputs + [worker_res.stderr], failure

        if not self.keep_sandbox and self.sandbox_dir.exists():
            try:
                shutil.rmtree(self.sandbox_dir)
            except OSError:
                pass
        return True, outputs, None

    def create_execution_env(
        self,
        *,
        gpu_id: int | None,
        enable_writable_tmpfs: bool,
        overlay_path: Path | None,
        extra_start_args: Sequence[str] | None,
    ) -> ExecutionEnvironment | None:
        if not self.worker_sif.exists():
            return None
        env = ExecutionEnvironment(
            workspace=self.workspace,
            image=self.worker_sif,
            runtime_preference="singularity",
            workspace_mount=self.workspace_mount,
            gpu_id=gpu_id,
            instance_name=self.worker_sif.stem,
            enable_writable_tmpfs=enable_writable_tmpfs,
            overlay_path=str(overlay_path) if overlay_path else None,
            extra_start_args=extra_start_args,
        )
        env.start()
        return env


def choose_compiler(env: ExecutionEnvironment, candidates: Iterable[str]) -> str | None:
    """Pick the first available compiler from the candidate list within the execution environment."""
    for compiler in candidates:
        compiler_cmd = str(compiler)
        try:
            result = env.run(["which", compiler_cmd])
        except FileNotFoundError:
            continue

        if result.returncode == 0 and result.stdout.strip():
            return compiler_cmd
    return None


def collect_available_compilers(env: ExecutionEnvironment) -> list[dict[str, str]]:
    """
    Discover available compilers inside the execution environment.

    The caller is responsible for starting the environment if a container image is configured.
    """
    compilers: list[dict[str, str]] = []
    for candidate in ("mpicc", "mpicxx", "gcc", "g++", "clang", "clang++", "cc", "c++", "icx", "icpx"):
        try:
            which_res = env.run(["bash", "-lc", f"command -v {candidate}"], cwd=env.workspace)
        except FileNotFoundError:
            continue
        if which_res.returncode != 0:
            continue
        compiler_path = which_res.stdout.strip().splitlines()[0] if which_res.stdout else candidate
        version_res = env.run(["bash", "-lc", f"{candidate} --version | head -n 1"], cwd=env.workspace)
        version_line = (version_res.stdout or version_res.stderr or "").strip().splitlines()
        compilers.append(
            {
                "name": candidate,
                "path": compiler_path,
                "version": version_line[0] if version_line else "",
            }
        )
    return compilers


def collect_available_libs(env: ExecutionEnvironment, names: Sequence[str] | None = None) -> list[str]:
    """Probe for a short list of shared libraries available inside the environment."""
    probe_names = list(names) if names else ["libcnpy", "libz", "libstdc++", "libc"]
    found: list[str] = []
    for name in probe_names:
        try:
            res = env.run(["bash", "-lc", f"ldconfig -p | grep -m1 {name}"], cwd=env.workspace)
        except FileNotFoundError:
            continue
        if res.returncode == 0 and res.stdout.strip():
            found.append(name)
    return found
