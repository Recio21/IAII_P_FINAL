from __future__ import annotations
import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Iterable
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | seeder | %(message)s")
log = logging.getLogger("seeder")


class SeederSettings(BaseSettings):
    api_base_url: str = "http://backend:8000"
    seeder_rps: float = 5.0
    seeder_concurrency: int = 4
    seeder_image_dir: Path = Path("/app/sample_images")
    seeder_timeout: float = 10.0
    seeder_backoff_initial: float = 1.0
    seeder_backoff_max: float = 30.0
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXT]


async def _send_one(client: httpx.AsyncClient, path: Path) -> tuple[bool, int]:
    try:
        with path.open("rb") as f:
            data = f.read()
        files = {"file": (path.name, data, "image/jpeg")}
        r = await client.post("/predict", files=files)
        return r.status_code < 400, r.status_code
    except (httpx.RequestError, httpx.HTTPError) as exc:
        log.warning("request error: %s", exc)
        return False, -1


async def _worker(
    name: int,
    queue: asyncio.Queue[Path],
    client: httpx.AsyncClient,
    cfg: SeederSettings,
) -> None:
    backoff = cfg.seeder_backoff_initial
    while True:
        path = await queue.get()
        try:
            ok, status = await _send_one(client, path)
            if ok:
                backoff = cfg.seeder_backoff_initial
                log.info("[w%d] %s -> %s", name, path.name, status)
            else:
                log.warning("[w%d] failure status=%s; backoff=%.1fs", name, status, backoff)
                await asyncio.sleep(backoff)
                backoff = min(cfg.seeder_backoff_max, backoff * 2)
        finally:
            queue.task_done()


async def _producer(queue: asyncio.Queue[Path], images: Iterable[Path], rps: float) -> None:
    interval = 1.0 / max(rps, 0.01)
    images = list(images)
    if not images:
        log.error("No images found; producer exiting")
        return
    while True:
        await queue.put(random.choice(images))
        await asyncio.sleep(interval)


async def main() -> None:
    cfg = SeederSettings()
    images = _list_images(cfg.seeder_image_dir)
    log.info("Found %d images in %s", len(images), cfg.seeder_image_dir)
    log.info("Target=%s rps=%.2f concurrency=%d", cfg.api_base_url, cfg.seeder_rps, cfg.seeder_concurrency)

    timeout = httpx.Timeout(cfg.seeder_timeout)
    limits = httpx.Limits(max_connections=cfg.seeder_concurrency * 2, max_keepalive_connections=cfg.seeder_concurrency)
    async with httpx.AsyncClient(base_url=cfg.api_base_url, timeout=timeout, limits=limits) as client:
        # Wait until backend is healthy
        for attempt in range(30):
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    break
            except httpx.RequestError:
                pass
            await asyncio.sleep(2)
        else:
            log.error("Backend never became healthy; exiting")
            return

        queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=cfg.seeder_concurrency * 4)
        workers = [asyncio.create_task(_worker(i, queue, client, cfg)) for i in range(cfg.seeder_concurrency)]
        producer = asyncio.create_task(_producer(queue, images, cfg.seeder_rps))
        await asyncio.gather(producer, *workers)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("seeder stopped")
