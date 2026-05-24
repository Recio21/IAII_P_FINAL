"""
github_sync_service.py — Descarga modelos nuevos desde GitHub Releases.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_GH_API  = "https://api.github.com"
_TIMEOUT = 300  # segundos — los .keras pueden ser grandes


class GitHubSyncService:
    def __init__(self, repo: str, model_dir: Path, token: str = "") -> None:
        self._repo      = repo
        self._model_dir = model_dir
        self._token     = token

    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # ── Pública ──────────────────────────────────────────────────────────────

    def sync(self) -> Dict:
        releases = self._fetch_releases()
        if releases is None:
            return {"ok": False, "error": "GitHub API unreachable — check network or token", "downloaded": [], "skipped": [], "errors": []}
        if not isinstance(releases, list):
            return {"ok": False, "error": f"Unexpected GitHub API response: {str(releases)[:200]}", "downloaded": [], "skipped": [], "errors": []}
        if len(releases) == 0:
            return {"ok": True, "downloaded": [], "skipped": [], "errors": [], "message": "No releases found in GitHub repo"}

        downloaded: List[str] = []
        skipped:    List[str] = []
        errors:     List[str] = []

        for release in releases:
            assets = release.get("assets", [])
            for asset in assets:
                name = asset.get("name", "")
                if not self._is_model_asset(name):
                    continue
                dest = self._model_dir / name
                if dest.exists():
                    skipped.append(name)
                    logger.debug("Skipping existing asset: %s", name)
                    continue
                url = asset.get("browser_download_url", "")
                logger.info("Downloading %s from %s", name, url)
                ok, err = self._download(url, dest)
                if ok:
                    downloaded.append(name)
                    logger.info("Downloaded: %s", name)
                else:
                    errors.append(f"{name}: {err}")
                    logger.error("Failed to download %s: %s", name, err)

        return {
            "ok": True,
            "downloaded": downloaded,
            "skipped": skipped,
            "errors": errors,
            "total_releases": len(releases),
            "total_available": len(downloaded) + len(skipped),
        }

    # ── Privadas ─────────────────────────────────────────────────────────────

    def trigger_training(self, epochs: int = 5) -> Dict:
        """Dispara el workflow de GitHub Actions via workflow_dispatch."""
        if not self._token:
            return {"ok": False, "message": "GITHUB_TOKEN not configured — cannot trigger workflow"}
        url = f"{_GH_API}/repos/{self._repo}/actions/workflows/train-and-release.yml/dispatches"
        payload = {"ref": "main", "inputs": {"epochs": str(epochs)}}
        try:
            resp = httpx.post(url, json=payload, headers=self._headers(), timeout=15)
            if resp.status_code == 204:
                return {"ok": True, "message": f"✓ Training workflow triggered ({epochs} epochs). Check GitHub Actions for progress."}
            else:
                return {"ok": False, "message": f"GitHub API error {resp.status_code}: {resp.text[:200]}"}
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def _fetch_releases(self) -> Optional[list]:
        url = f"{_GH_API}/repos/{self._repo}/releases"
        try:
            resp = httpx.get(url, timeout=15, headers=self._headers())
            logger.info("GitHub API %s → %d", url, resp.status_code)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("GitHub API error: %s", exc)
            return None

    @staticmethod
    def _is_model_asset(name: str) -> bool:
        return name.endswith(".keras") or (
            name.startswith("metadata_") and name.endswith(".json")
        )

    def _download(self, url: str, dest: Path):
        tmp = dest.with_suffix(".tmp")
        try:
            headers = self._headers()
            # browser_download_url no necesita Accept especial
            headers.pop("Accept", None)
            with httpx.stream("GET", url, timeout=_TIMEOUT, follow_redirects=True, headers=headers) as resp:
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
            tmp.rename(dest)
            return True, None
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            return False, str(exc)
