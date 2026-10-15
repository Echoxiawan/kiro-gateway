# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
# Copyright (C) 2026 Echoxiawan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Debug logging module for requests.

Supports three modes (DEBUG_MODE):
- off: logging disabled
- errors: logs are saved only on errors (4xx, 5xx)
- all: every request gets its own directory under <DEBUG_DIR>/requests/

Each request owns its directory and its buffers, tracked in a context variable
so concurrent requests (several clients hitting the gateway at once) never write
into each other's snapshot. Application logs are captured through a loguru sink
filtered by request state for the same reason.

The newest request is also exposed at <DEBUG_DIR>/latest for convenience.

In "errors" mode, data is buffered in memory and flushed to files
only when flush_on_error() is called.
"""

import contextvars
import io
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from kiro.config import DEBUG_DIR, DEBUG_KEEP_REQUESTS, DEBUG_MODE


# Files that make up one request's debug snapshot.
REQUEST_FILES = (
    "request_body.json",
    "kiro_request_body.json",
    "response_stream_raw.txt",
    "response_stream_modified.txt",
    "app_logs.txt",
    "error_info.json",
)

# Relative name of the symlink that points at the newest request directory.
LATEST_LINK_NAME = "latest"


@dataclass
class _RequestLogState:
    """
    Logging state for one request.

    Holding this per request (instead of on the singleton) is what keeps
    concurrent requests from overwriting each other's files and log buffers.
    """

    directory: Path
    request_body: Optional[bytes] = None
    kiro_request_body: Optional[bytes] = None
    raw_chunks: bytearray = field(default_factory=bytearray)
    modified_chunks: bytearray = field(default_factory=bytearray)
    app_logs: io.StringIO = field(default_factory=io.StringIO)
    sink_id: Optional[int] = None
    # False only for the fallback state used when no request is in progress.
    request_scoped: bool = True


_current_state: contextvars.ContextVar[Optional[_RequestLogState]] = contextvars.ContextVar(
    "kiro_debug_request_state", default=None
)


class DebugLogger:
    """
    Singleton for managing debug request logs.

    Operating modes:
    - off: does nothing
    - errors: buffers data, flushes to files only on errors
    - all: writes each request immediately into its own directory
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DebugLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.debug_dir = Path(DEBUG_DIR)
        self._initialized = True

    def _is_enabled(self) -> bool:
        """Checks if logging is enabled."""
        return DEBUG_MODE in ("errors", "all")

    def _is_immediate_write(self) -> bool:
        """Checks if immediate file writing is needed (all mode)."""
        return DEBUG_MODE == "all"

    def _state(self) -> _RequestLogState:
        """
        Returns the logging state for the caller.

        Falls back to a directory-scoped state when no request is in progress,
        so the logger stays usable outside the request lifecycle (tests, tooling).
        """
        state = _current_state.get()
        if state is not None:
            return state
        return _current_state.get() or self._create_fallback_state()

    def _create_fallback_state(self) -> _RequestLogState:
        """Creates and installs a non-request state rooted at DEBUG_DIR."""
        state = _RequestLogState(directory=self.debug_dir, request_scoped=False)
        self._attach_app_logs_sink(state)
        _current_state.set(state)
        return state

    # ==================== Request lifecycle ====================

    def prepare_new_request(self):
        """
        Starts a fresh debug snapshot for the current request.

        In "all" mode each request gets its own directory under
        <DEBUG_DIR>/requests/, so a new request never overwrites or steals the
        files of a request that is still streaming.
        In "errors" mode buffers are reset and files are written only on error.
        """
        if not self._is_enabled():
            return

        # A task must not inherit another request's state.
        self._finalize_state(_current_state.get())

        if self._is_immediate_write():
            state = _RequestLogState(directory=self._new_request_directory())
        else:
            state = _RequestLogState(directory=self.debug_dir)

        self._attach_app_logs_sink(state)
        _current_state.set(state)

        if self._is_immediate_write():
            try:
                self._update_latest_link(state.directory)
                self._prune_archives()
            except Exception as e:
                logger.error(f"[DebugLogger] Error preparing request directory: {e}")

    def _new_request_directory(self) -> Path:
        """Creates and returns a unique directory for one request."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
        requests_dir = self.debug_dir / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)

        directory = requests_dir / stamp
        suffix = 1
        while directory.exists():
            directory = requests_dir / f"{stamp}-{suffix}"
            suffix += 1

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _update_latest_link(self, directory: Path) -> None:
        """Points <DEBUG_DIR>/latest at the newest request directory."""
        link = self.debug_dir / LATEST_LINK_NAME
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(directory.relative_to(self.debug_dir), target_is_directory=True)
        except OSError:
            # Symlinks are unavailable on some platforms/filesystems - the
            # timestamped directories remain the source of truth.
            pass

    def _prune_archives(self):
        """Removes the oldest archives beyond DEBUG_KEEP_REQUESTS (0 keeps all)."""
        if DEBUG_KEEP_REQUESTS <= 0:
            return

        requests_dir = self.debug_dir / "requests"
        if not requests_dir.is_dir():
            return

        archives = sorted((d for d in requests_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
        for old_archive in archives[:-DEBUG_KEEP_REQUESTS]:
            shutil.rmtree(old_archive, ignore_errors=True)

    def _attach_app_logs_sink(self, state: _RequestLogState) -> None:
        """
        Captures application logs for one request into its own buffer.

        The sink is filtered by request state: loguru delivers every record to
        every sink, so without the filter a concurrent request's logs would be
        duplicated into this request's app_logs.txt.
        """
        state.sink_id = logger.add(
            state.app_logs,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            colorize=False,
            filter=lambda record: _current_state.get() is state,
        )

    def _detach_app_logs_sink(self, state: _RequestLogState) -> None:
        """Removes this request's loguru sink."""
        if state.sink_id is None:
            return
        try:
            logger.remove(state.sink_id)
        except ValueError:
            # Sink already removed
            pass
        state.sink_id = None

    def _finalize_state(self, state: Optional[_RequestLogState]) -> None:
        """Detaches hooks and clears the request state so nothing leaks."""
        if state is None:
            return
        self._detach_app_logs_sink(state)
        state.raw_chunks.clear()
        state.modified_chunks.clear()
        state.request_body = None
        state.kiro_request_body = None
        state.app_logs = io.StringIO()
        if _current_state.get() is state:
            _current_state.set(None)

    # ==================== Public logging API ====================

    def log_request_body(self, body: bytes):
        """
        Saves the request body (from client).

        In "all" mode: writes immediately to the request's own file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        state = self._state()
        if self._is_immediate_write():
            self._write_request_body_to_file(body, state.directory)
        else:
            state.request_body = body

    def log_kiro_request_body(self, body: bytes):
        """
        Saves the modified request body (to Kiro API).

        In "all" mode: writes immediately to the request's own file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        state = self._state()
        if self._is_immediate_write():
            self._write_kiro_request_body_to_file(body, state.directory)
        else:
            state.kiro_request_body = body

    def log_raw_chunk(self, chunk: bytes):
        """
        Appends raw response chunk (from provider).

        In "all" mode: writes immediately to the request's own file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        state = self._state()
        if self._is_immediate_write():
            self._append_chunk_to_file(chunk, state.directory, "response_stream_raw.txt")
        else:
            state.raw_chunks.extend(chunk)

    def log_modified_chunk(self, chunk: bytes):
        """
        Appends modified chunk (to client).

        In "all" mode: writes immediately to the request's own file.
        In "errors" mode: buffers.
        """
        if not self._is_enabled():
            return

        state = self._state()
        if self._is_immediate_write():
            self._append_chunk_to_file(chunk, state.directory, "response_stream_modified.txt")
        else:
            state.modified_chunks.extend(chunk)

    def log_error_info(self, status_code: int, error_message: str = ""):
        """
        Writes error information to file.

        Works in both modes (errors and all).

        Args:
            status_code: HTTP error status code
            error_message: Error message (optional)
        """
        if not self._is_enabled():
            return

        state = self._state()
        try:
            state.directory.mkdir(parents=True, exist_ok=True)
            error_file = state.directory / "error_info.json"
            error_info = {
                "status_code": status_code,
                "error_message": error_message,
            }
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            logger.debug(f"[DebugLogger] Error info saved (status={status_code})")
        except Exception as e:
            logger.error(f"[DebugLogger] Error writing error_info: {e}")

    def flush_on_error(self, status_code: int, error_message: str = ""):
        """
        Flushes buffers to files on error.

        In "errors" mode: flushes buffers and saves error_info.
        In "all" mode: only saves error_info (data already written).

        Args:
            status_code: HTTP error status code
            error_message: Error message (optional)
        """
        if not self._is_enabled():
            return

        state = self._state()

        if self._is_immediate_write():
            self.log_error_info(status_code, error_message)
            self._write_app_logs_to_file(state)
            self._finalize_state(state)
            return

        has_data = any([
            state.request_body,
            state.kiro_request_body,
            state.raw_chunks,
            state.modified_chunks,
        ])
        if not has_data:
            self._finalize_state(state)
            return

        try:
            state.directory.mkdir(parents=True, exist_ok=True)
            for name in REQUEST_FILES:
                leftover = state.directory / name
                if leftover.exists():
                    leftover.unlink()

            if state.request_body:
                self._write_request_body_to_file(state.request_body, state.directory)

            if state.kiro_request_body:
                self._write_kiro_request_body_to_file(state.kiro_request_body, state.directory)

            if state.raw_chunks:
                self._append_chunk_to_file(bytes(state.raw_chunks), state.directory, "response_stream_raw.txt")

            if state.modified_chunks:
                self._append_chunk_to_file(
                    bytes(state.modified_chunks), state.directory, "response_stream_modified.txt"
                )

            # Save error information
            self.log_error_info(status_code, error_message)

            # Save application logs
            self._write_app_logs_to_file(state)

            logger.info(f"[DebugLogger] Error logs flushed to {state.directory} (status={status_code})")
        except Exception as e:
            logger.error(f"[DebugLogger] Error flushing buffers: {e}")
        finally:
            self._finalize_state(state)

    def discard_buffers(self):
        """
        Finishes the current request's logging.

        Called when a request completed successfully. In "all" mode the captured
        application logs are written into the request's own directory; in
        "errors" mode the buffers are dropped.
        """
        if not self._is_enabled():
            return

        state = self._state()
        if DEBUG_MODE == "all":
            self._write_app_logs_to_file(state)
        self._finalize_state(state)

    # ==================== Private file writing methods ====================

    def _write_request_body_to_file(self, body: bytes, directory: Path):
        """Writes request body to file."""
        try:
            file_path = directory / "request_body.json"
            self._write_json_or_raw(body, file_path)
        except Exception as e:
            logger.error(f"[DebugLogger] Error writing request_body: {e}")

    def _write_kiro_request_body_to_file(self, body: bytes, directory: Path):
        """Writes Kiro request body to file."""
        try:
            file_path = directory / "kiro_request_body.json"
            self._write_json_or_raw(body, file_path)
        except Exception as e:
            logger.error(f"[DebugLogger] Error writing kiro_request_body: {e}")

    @staticmethod
    def _write_json_or_raw(body: bytes, file_path: Path):
        """Writes pretty-printed JSON, falling back to the raw bytes."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            json_obj = json.loads(body)
        except json.JSONDecodeError:
            with open(file_path, "wb") as f:
                f.write(body)
            return
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _append_chunk_to_file(chunk: bytes, directory: Path, file_name: str):
        """Appends a chunk to a stream file in the given directory."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            file_path = directory / file_name
            with open(file_path, "ab") as f:
                f.write(chunk)
        except Exception:
            pass

    def _write_app_logs_to_file(self, state: _RequestLogState):
        """Writes captured application logs to the request's file."""
        try:
            logs_content = state.app_logs.getvalue()
            if not logs_content.strip():
                return

            state.directory.mkdir(parents=True, exist_ok=True)
            file_path = state.directory / "app_logs.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(logs_content)
        except Exception:
            # Don't log the error through the logger to avoid recursion
            pass


# Global instance
debug_logger = DebugLogger()
