"""Capture transcript files as stable opaque byte snapshots."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from codex_memory.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class TranscriptCapture:
    status: Literal["captured", "unavailable"]
    source_path: str | None
    blob_sha256: str | None
    byte_size: int | None


class TranscriptCaptureError(RuntimeError):
    """Raised when an existing transcript cannot be captured consistently."""


def capture_transcript(
    path: Path | str | None,
    blobs: BlobStore,
    attempts: int = 3,
    delay_seconds: float = 0.05,
) -> TranscriptCapture:
    """Store a stable transcript snapshot without interpreting its content."""

    _validate_retry_options(attempts, delay_seconds)
    if path is None:
        return TranscriptCapture("unavailable", None, None, None)

    source_path = str(path)
    transcript_path = Path(path)
    observed_existing = False
    for attempt in range(attempts):
        try:
            before = transcript_path.stat()
            observed_existing = True
            data = transcript_path.read_bytes()
            after = transcript_path.stat()
        except FileNotFoundError:
            if attempt == 0 and not observed_existing:
                return TranscriptCapture("unavailable", source_path, None, None)
        except OSError:
            pass
        else:
            if (
                before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
                and len(data) == before.st_size
            ):
                try:
                    ref = blobs.put_bytes(data)
                except Exception as error:
                    raise TranscriptCaptureError("stable transcript blob storage failed") from error
                return TranscriptCapture("captured", source_path, ref.sha256, ref.byte_size)

        if attempt < attempts - 1:
            time.sleep(delay_seconds)

    raise TranscriptCaptureError(f"transcript could not be captured stably after {attempts} attempts")


def _validate_retry_options(attempts: object, delay_seconds: object) -> None:
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be an integer greater than or equal to 1")
    if (
        isinstance(delay_seconds, bool)
        or not isinstance(delay_seconds, (int, float))
        or not math.isfinite(delay_seconds)
        or delay_seconds < 0
    ):
        raise ValueError("delay_seconds must be a finite non-negative number")
