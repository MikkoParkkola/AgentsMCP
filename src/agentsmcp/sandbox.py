"""
agentsmcp.sandbox
=================

Production-ready sandboxing wrapper built on Docker Engine for secure execution
of untrusted code with strict resource limits and audit logging.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import docker
    from docker.errors import DockerException, NotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
    DockerException = Exception
    NotFound = Exception


@dataclass(slots=True)
class SandboxConfig:
    """
    Configuration that defines the sandbox environment.
    
    Attributes
    ----------
    image: Docker image used for the container
    command: Command to run inside the container
    env: Environment variables visible to the container
    cpu_quota: CPU time in microseconds per cpu_period (50000 = 50% of single core)
    cpu_period: Length of a CPU period in microseconds (default 100000)
    mem_limit: Memory limit as Docker-compatible string (e.g. "256m")
    timeout_seconds: Maximum wall-clock time the command may run
    network_mode: Docker network mode ("none" disables all networking)
    read_only_rootfs: Mount the container root-filesystem read-only
    bind_mounts: Host to container read-only bind mounts
    tmpfs: Paths to mount as tmpfs (writable, in-memory) inside container
    working_dir: Working directory inside the container
    user: User under which the command runs (default "nobody")
    """
    image: str
    command: List[str] | str
    env: Mapping[str, str] = field(default_factory=dict)
    cpu_quota: Optional[int] = None
    cpu_period: int = 100_000
    mem_limit: Optional[str] = None
    timeout_seconds: int = 30
    network_mode: str = "none"
    read_only_rootfs: bool = True
    bind_mounts: Mapping[pathlib.Path, pathlib.Path] = field(default_factory=dict)
    tmpfs: Iterable[pathlib.Path] = field(default_factory=list)
    working_dir: Optional[pathlib.Path] = None
    user: Optional[str] = "nobody"

    def _as_host_config(self) -> dict:
        """Return a Docker API host_config dict based on the instance."""
        binds = {
            str(src): {"bind": str(dst), "mode": "ro"}
            for src, dst in self.bind_mounts.items()
        }
        tmpfs_mounts = {str(p): "" for p in self.tmpfs}
        return {
            "binds": binds,
            "tmpfs": tmpfs_mounts,
            "network_mode": self.network_mode,
            "readonly_rootfs": self.read_only_rootfs,
            "auto_remove": False,
            "cpu_period": self.cpu_period,
            "cpu_quota": self.cpu_quota,
            "mem_limit": self.mem_limit,
            "user": self.user,
        }

    def _as_container_kwargs(self) -> dict:
        """Return kwargs suitable for docker_client.containers.create."""
        cmd = self.command if isinstance(self.command, list) else shlex.split(self.command)
        return {
            "image": self.image,
            "command": cmd,
            "environment": dict(self.env),
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "host_config": self._as_host_config(),
        }


class SandboxExecutor:
    """
    Executes untrusted code in an isolated Docker container.
    
    Parameters
    ----------
    config: The sandbox configuration
    logger: Logger used for audit events
    docker_client: Pre-configured Docker client (mainly for testing)
    """
    
    def __init__(
        self,
        config: SandboxConfig,
        logger: Optional[logging.Logger] = None,
        docker_client: Optional[Any] = None,
    ) -> None:
        if not DOCKER_AVAILABLE:
            raise ImportError("Docker SDK not available. Install with: pip install docker")
            
        self.cfg = config
        self.log = logger or logging.getLogger(__name__)
        self.docker = docker_client or docker.from_env()
        self._container_name = f"sandbox-{uuid.uuid4().hex[:12]}"

    async def execute(self, stdin: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Run the configured command inside a sandboxed container.
        
        Parameters
        ----------
        stdin: Data to feed to the container's stdin stream
        
        Returns
        -------
        dict containing:
            - stdout (str): captured standard output
            - stderr (str): captured standard error
            - exit_code (int): process exit status
            - duration (float): wall-clock time in seconds
            - timed_out (bool): True if command exceeded timeout
        """
        start_ts = asyncio.get_event_loop().time()
        self.log.info("Sandbox start: %s", self._container_name)
        container = None
        
        try:
            container = await asyncio.get_event_loop().run_in_executor(
                None, self._create_container
            )
            await self._start_container(container, stdin)
            
            timed_out = await self._wait(container, self.cfg.timeout_seconds)
            
            if timed_out:
                self.log.warning("Sandbox %s timed out - killing", self._container_name)
                await self._kill_container(container)
            
            exit_code = await self._get_exit_code(container)
            stdout, stderr = await self._collect_logs(container)
            
            duration = asyncio.get_event_loop().time() - start_ts
            self.log.info(
                "Sandbox %s finished (exit=%s, time=%.2fs, timeout=%s)",
                self._container_name,
                exit_code,
                duration,
                timed_out,
            )
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "duration": duration,
                "timed_out": timed_out,
            }
            
        except DockerException as exc:
            self.log.exception("Docker error in sandbox %s", self._container_name)
            raise RuntimeError(f"Sandbox execution failed: {exc}") from exc
            
        finally:
            if container:
                await self._cleanup_container(container)

    def _create_container(self) -> Any:
        """Create a Docker container according to the config."""
        kwargs = self.cfg._as_container_kwargs()
        host_cfg = self.docker.api.create_host_config(**kwargs.pop("host_config"))
        container = self.docker.api.create_container(
            name=self._container_name,
            host_config=host_cfg,
            **kwargs,
        )
        return self.docker.containers.get(container["Id"])

    async def _start_container(self, container: Any, stdin: Optional[bytes]) -> None:
        """Start the container, optionally feeding stdin."""
        def _run():
            container.start()
            if stdin:
                sock = container.attach_socket(params={"stdin": 1, "stream": 1})
                sock._sock.sendall(stdin)
                sock._sock.shutdown(1)
        
        await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _wait(self, container: Any, timeout: int) -> bool:
        """Wait for container to finish. Returns True if timed out."""
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, container.wait),
                timeout=timeout,
            )
            return False
        except asyncio.TimeoutError:
            return True

    async def _kill_container(self, container: Any) -> None:
        """Force kill a running container."""
        await asyncio.get_event_loop().run_in_executor(None, container.kill)

    async def _get_exit_code(self, container: Any) -> int:
        """Get the container's exit code."""
        result = await asyncio.get_event_loop().run_in_executor(None, container.wait)
        return result.get("StatusCode", -1)

    async def _collect_logs(self, container: Any) -> Tuple[str, str]:
        """Retrieve stdout and stderr as strings."""
        def _logs():
            out = container.logs(stdout=True, stderr=False, stream=False)
            err = container.logs(stdout=False, stderr=True, stream=False)
            return out.decode(errors="replace"), err.decode(errors="replace")
        
        return await asyncio.get_event_loop().run_in_executor(None, _logs)

    async def _cleanup_container(self, container: Any) -> None:
        """Ensure the container is removed, regardless of its state."""
        def _remove():
            try:
                container.remove(force=True, v=True)
                self.log.debug("Sandbox %s container removed", self._container_name)
            except NotFound:
                self.log.debug("Sandbox %s already absent", self._container_name)
            except DockerException as exc:
                self.log.error("Failed to remove container %s: %s", self._container_name, exc)
        
        await asyncio.get_event_loop().run_in_executor(None, _remove)