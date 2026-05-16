import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class DVCService:
    """Wrapper minimalista de la CLI de DVC. Soporta pull y rollback básico.
    No bloquea inferencia: las operaciones son sobre archivos en model_dir,
    el ModelService sólo cambia de versión cuando se invoca switch.
    """

    def __init__(self, repo_root: Path, model_dir: Path) -> None:
        self._repo_root = repo_root
        self._model_dir = model_dir

    def available(self) -> bool:
        return shutil.which("dvc") is not None

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        logger.info("dvc %s", " ".join(args))
        return subprocess.run(
            ["dvc", *args], cwd=self._repo_root,
            capture_output=True, text=True, check=False,
        )

    def pull(self, target: str | None = None) -> tuple[bool, str]:
        if not self.available():
            return False, "DVC CLI not available in this container"
        args = ["pull"]
        if target:
            args.append(target)
        res = self._run(args)
        ok = res.returncode == 0
        return ok, (res.stdout + res.stderr).strip()

    def rollback(self, previous_model_file: str) -> bool:
        target = self._model_dir / previous_model_file
        return target.exists()
