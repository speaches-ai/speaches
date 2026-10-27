from __future__ import annotations

import inspect
import logging
from multiprocessing import connection, get_context
import os
import pickle
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from multiprocessing.connection import Connection

    from speaches.executors.shared.base_model_manager import BaseModelManager

logger = logging.getLogger(__name__)

# Messages are tuples. parent -> child: ("call", rid, method, request, kwargs), ("cancel", rid), ("shutdown",).
# child -> parent: ("ready",), ("load_error", exc), ("item", rid, obj), ("value", rid, obj), ("done", rid),
# ("error", rid, exc).

STOP_GRACE_SECONDS = 5
STOP_TERMINATE_SECONDS = 2


class ModelWorkerDiedError(RuntimeError):
    def __init__(self, exitcode: int | None) -> None:
        super().__init__(f"Model worker process died (exit code {exitcode})")
        self.exitcode = exitcode


def _picklable_exception(exc: BaseException) -> BaseException:
    try:
        pickle.loads(pickle.dumps(exc))  # noqa: S301
    except Exception:  # noqa: BLE001
        return RuntimeError(f"{type(exc).__name__}: {exc}")
    return exc


def _send(conn: Connection, lock: threading.Lock, message: tuple[Any, ...]) -> bool:
    try:
        with lock:
            conn.send(message)
    except (OSError, EOFError):
        return False
    return True


def _run_call(
    manager: BaseModelManager[Any],
    conn: Connection,
    lock: threading.Lock,
    cancels: dict[int, threading.Event],
    rid: int,
    method: str,
    request: Any,
    kwargs: dict[str, Any],
) -> None:
    cancel = cancels[rid]
    try:
        result = getattr(manager, method)(request, **kwargs)
        if inspect.isgenerator(result):
            try:
                for item in result:
                    if cancel.is_set():
                        break
                    if not _send(conn, lock, ("item", rid, item)):
                        return
            finally:
                result.close()
            _send(conn, lock, ("done", rid))
        else:
            _send(conn, lock, ("value", rid, result))
    except Exception as e:
        logger.exception(f"Call {method} failed in model worker")
        _send(conn, lock, ("error", rid, _picklable_exception(e)))
    finally:
        cancels.pop(rid, None)


def _worker_main(
    conn: Connection,
    manager_cls: type[BaseModelManager[Any]],
    ctor_args: tuple[Any, ...],
    model_id: str,
    log_level: str,
) -> None:
    from speaches.logger import setup_logger

    setup_logger(log_level)
    lock = threading.Lock()
    cancels: dict[int, threading.Event] = {}
    try:
        manager = manager_cls(-1, *ctor_args)
        with manager.load_model(model_id):
            pass
    except Exception as e:
        logger.exception(f"Failed to load {model_id} in model worker")
        _send(conn, lock, ("load_error", _picklable_exception(e)))
        logging.shutdown()
        os._exit(1)
    _send(conn, lock, ("ready",))

    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            # Parent is gone (even by SIGKILL): exit so no orphan keeps the memory.
            break
        kind = message[0]
        if kind == "call":
            _, rid, method, request, kwargs = message
            cancels[rid] = threading.Event()
            threading.Thread(
                target=_run_call,
                args=(manager, conn, lock, cancels, rid, method, request, kwargs),
                daemon=True,
            ).start()
        elif kind == "cancel":
            event = cancels.get(message[1])
            if event is not None:
                event.set()
        elif kind == "shutdown":
            break
    logging.shutdown()
    os._exit(0)


class ModelWorker:
    def __init__(
        self,
        model_id: str,
        manager_cls: type[BaseModelManager[Any]],
        ctor_args: tuple[Any, ...],
        log_level: str,
        on_death: Callable[[ModelWorker], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self._on_death = on_death
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[tuple[str, Any]]] = {}
        self._next_rid = 0
        self._dead = False
        self._stopped = False

        ctx = get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_worker_main,
            args=(child_conn, manager_cls, ctor_args, model_id, log_level),
            name=f"speaches-worker-{model_id}",
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._reader = threading.Thread(target=self._read_loop, name=f"{self._proc.name}-reader", daemon=True)

    def start(self) -> None:
        try:
            self._wait_ready()
        except BaseException:
            self.stop()
            raise
        self._reader.start()

    def _wait_ready(self) -> None:
        sentinel = self._proc.sentinel
        while True:
            ready = connection.wait([self._conn, sentinel])
            if self._conn in ready:
                try:
                    message = self._conn.recv()
                except (EOFError, OSError):
                    raise ModelWorkerDiedError(self._exitcode()) from None
                if message[0] == "ready":
                    return
                if message[0] == "load_error":
                    raise message[1]
            else:
                raise ModelWorkerDiedError(self._exitcode())

    def _exitcode(self) -> int | None:
        self._proc.join(1)
        return self._proc.exitcode

    def _read_loop(self) -> None:
        sentinel = self._proc.sentinel
        while True:
            ready = connection.wait([self._conn, sentinel])
            if self._conn not in ready:
                break
            try:
                message = self._conn.recv()
            except (EOFError, OSError):
                break
            with self._state_lock:
                pending = self._pending.get(message[1])
            if pending is not None:
                pending.put((message[0], message[2] if len(message) > 2 else None))
        self._mark_dead()

    def _mark_dead(self) -> None:
        exitcode = self._exitcode()
        with self._state_lock:
            self._dead = True
            pending = list(self._pending.values())
            stopping = self._stopped
        if not stopping:
            logger.error(f"Model worker for {self.model_id} died (exit code {exitcode})")
        for q in pending:
            q.put(("error", ModelWorkerDiedError(exitcode)))
        self._conn.close()
        if not stopping and self._on_death is not None:
            self._on_death(self)

    def _submit(self, method: str, request: Any, kwargs: dict[str, Any]) -> tuple[int, queue.Queue[tuple[str, Any]]]:
        q: queue.Queue[tuple[str, Any]] = queue.Queue()
        with self._state_lock:
            if self._dead or self._stopped:
                raise ModelWorkerDiedError(self._proc.exitcode)
            self._next_rid += 1
            rid = self._next_rid
            self._pending[rid] = q
        try:
            if not _send(self._conn, self._send_lock, ("call", rid, method, request, kwargs)):
                raise ModelWorkerDiedError(self._exitcode())
        except BaseException:
            self._release(rid, cancel=False)
            raise
        return rid, q

    def _release(self, rid: int, *, cancel: bool) -> None:
        with self._state_lock:
            self._pending.pop(rid, None)
            alive = not self._dead
        if cancel and alive:
            _send(self._conn, self._send_lock, ("cancel", rid))

    def value[R](self, method: str, request: Any, kwargs: dict[str, Any]) -> R:
        rid, q = self._submit(method, request, kwargs)
        try:
            kind, payload = q.get()
            if kind == "error":
                raise payload
            return payload
        finally:
            self._release(rid, cancel=False)

    def stream[R](self, method: str, request: Any, kwargs: dict[str, Any]) -> Generator[R]:
        rid, q = self._submit(method, request, kwargs)
        finished = False
        try:
            while True:
                kind, payload = q.get()
                if kind == "item":
                    yield payload
                    continue
                finished = True
                if kind == "error":
                    raise payload
                return
        finally:
            self._release(rid, cancel=not finished)

    def stop(self) -> None:
        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True
        start = time.perf_counter()
        _send(self._conn, self._send_lock, ("shutdown",))
        self._proc.join(STOP_GRACE_SECONDS)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(STOP_TERMINATE_SECONDS)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join()
        if self._reader.is_alive():
            self._reader.join(STOP_TERMINATE_SECONDS)
        else:
            self._conn.close()
        logger.debug(f"Model worker for {self.model_id} stopped in {time.perf_counter() - start:.2f}s")
