"""The board relay decides off the writer and catches up from a durable cursor.

Each test runs the real crew-log writer and the real relay consumer on a temp data home,
with a real binding file, so the binding read, the header read and the board's newest
unit are the ones a gateway uses. A "restart" is :func:`work_relay.forget_for_tests`:
it drops every in-memory relay state and keeps what is on disk, like a new process.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from kiro_crew import crew_log as lg
from kiro_crew import session_ledger
from kiro_crew.crew_log import CrewLog
from kiro_crew.crew_log import eager as crew_log_eager
from kiro_crew.crew_log import emit as crew_log_emit
from kiro_crew.crew_log import projection as crew_log
from kiro_crew.crew_log import work_relay, work_relay_feed
from kiro_crew.crew_log.emit_runtime import _write as emit_write
from kiro_crew.work_ledger_writes import _write_binding

LEAD = "chat-relay-feed-lead"
WORKER = "chat-relay-feed-worker"
LEAD_UNIT = "u-feed-lead"
WORKER_UNIT = "u-feed-worker"


@pytest.fixture(autouse=True)
def _live_crew_log(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
        patch.setenv("KIROCREW_CREW_LOG", "1")
        crew_log_emit.reset_caches()
        crew_log_eager.stop_for_tests()
        crew_log.forget_slot_folds()
        work_relay.forget_for_tests()
        yield patch
        crew_log_emit.flush()
        crew_log_eager.stop_for_tests()
        crew_log.forget_slot_folds()
        work_relay.forget_for_tests()
        crew_log_emit.reset_caches()


def _unit(unit: str, slot: str) -> None:
    CrewLog.create(lg.KIND_SESSION, unit, owner="owner", agent="lead", slot=slot)


def _settle() -> None:
    assert crew_log_emit.flush(timeout=10.0)
    assert crew_log_eager.drain(timeout=10.0)


def _board() -> None:
    _unit(LEAD_UNIT, LEAD)
    data = {"slot": LEAD, "actor": "conductor", "by": LEAD, "action": "goal", "goal": "g"}
    assert crew_log_emit.on_work_recorded(LEAD_UNIT, {**data, "round": 1}, timeout=5.0)
    _settle()


def _bind() -> None:
    _write_binding(WORKER, LEAD, "it_0000000a")


def _turn(turn: int, credits: float) -> None:
    emit_write(
        WORKER_UNIT,
        "turn/completed",
        {
            "turn": turn,
            "stop_reason": "end_turn",
            "depth": 0,
            "duration_ms": 10,
            "model": "opus",
            "provider": "kiro",
            "credits": credits,
            "tokens": {"input": 5, "output": 1, "cache_read": 0, "cache_write": 0},
        },
    )
    _settle()


def _charges() -> "list[float]":
    log = CrewLog.open(lg.KIND_SESSION, LEAD_UNIT)
    return [e.data["credits"] for e in log.iter_from(1) if e.type == "work/charged"]


def _restart() -> None:
    """Forget every in-memory relay state, as a new process would; disk stays."""
    _settle()
    work_relay.forget_for_tests()


def test_the_writer_lane_never_resolves_a_relay(_live_crew_log):
    """The header, binding and newest-unit reads all run on the relay's own thread."""
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    threads: "list[str]" = []
    for name in ("_unit_slot", "_bound_board"):
        real = getattr(work_relay, name)

        def spy(*args: Any, _real: Any = real) -> Any:
            threads.append(threading.current_thread().name)
            return _real(*args)

        _live_crew_log.setattr(work_relay, name, spy)
    real_newest = session_ledger.newest_work_unit

    def newest(board: str) -> str:
        threads.append(threading.current_thread().name)
        return real_newest(board)

    _live_crew_log.setattr(session_ledger, "newest_work_unit", newest)
    _turn(1, 2.5)
    assert _charges() == [2.5]
    assert threads and set(threads) == {"crew-log-relay"}, threads


def test_a_closer_the_crash_cut_off_is_relayed_after_the_restart(_live_crew_log):
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    with pytest.MonkeyPatch.context() as crash:
        # The closer commits, and the process dies before the relay decides it.
        crash.setattr(work_relay_feed, "note", lambda *a, **k: None)
        _turn(2, 2.0)
    assert _charges() == [1.0]
    _restart()
    _turn(3, 3.0)
    assert _charges() == [1.0, 2.0, 3.0]


def test_a_copy_that_landed_before_its_cursor_was_written_is_not_copied_again(
    _live_crew_log,
):
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    work_relay_feed.persist()
    cursor = work_relay_feed._cursor_path(work_relay_feed._state_home)
    assert json.loads(cursor.read_text())["units"][WORKER_UNIT] > 0
    # A crash after the copy landed and before the cursor moved: no cursor on disk.
    cursor.unlink()
    _restart()
    _turn(2, 2.0)
    assert _charges() == [1.0, 2.0]


def test_a_catch_up_does_not_relay_what_was_written_before_the_bind(_live_crew_log):
    _board()
    _unit(WORKER_UNIT, WORKER)
    _turn(1, 1.0)
    time.sleep(0.05)
    _bind()
    _restart()
    _turn(2, 2.0)
    assert _charges() == [2.0]


def test_a_full_queue_is_caught_up_from_the_cursor_in_order(_live_crew_log):
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    _live_crew_log.setattr(work_relay_feed, "QUEUE_LIMIT", 0)
    for turn in (2, 3, 4):
        _turn(turn, float(turn))
    assert work_relay_feed.overflowed() >= 3
    assert _charges() == [1.0, 2.0, 3.0, 4.0]


def test_the_cursor_only_moves_forward_and_survives_a_restart(_live_crew_log):
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    _turn(2, 2.0)
    work_relay_feed.persist()
    path = work_relay_feed._cursor_path(work_relay_feed._state_home)
    first = json.loads(path.read_text())["units"][WORKER_UNIT]
    _restart()
    _turn(3, 3.0)
    work_relay_feed.persist()
    later = json.loads(path.read_text())["units"][WORKER_UNIT]
    assert later > first
    assert _charges() == [1.0, 2.0, 3.0]


def test_a_copy_the_writer_refuses_holds_the_cursor_below_it(_live_crew_log):
    """A refused copy is not counted settled, so the next start tries it again."""
    from kiro_crew.crew_log.errors import CrewLogError

    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    work_relay_feed.persist()
    path = work_relay_feed._cursor_path(work_relay_feed._state_home)
    before = json.loads(path.read_text())["units"][WORKER_UNIT]

    def refuse(*_a: Any, **_k: Any) -> None:
        raise CrewLogError("refused for the test")

    with pytest.MonkeyPatch.context() as refused:
        refused.setattr(work_relay, "_append_run", refuse)
        _turn(2, 2.0)
        work_relay_feed.persist()
    assert json.loads(path.read_text())["units"][WORKER_UNIT] == before
    assert _charges() == [1.0]
    _restart()
    _turn(3, 3.0)
    assert _charges() == [1.0, 2.0, 3.0]


def _wait_for(charges: "list[float]", seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while _charges() != charges and time.monotonic() < deadline:
        time.sleep(0.05)
        crew_log_emit.flush(timeout=2.0)
    assert _charges() == charges


def test_a_failed_binding_read_is_retried_and_its_copies_keep_their_order(_live_crew_log):
    """A read that fails holds the cursor; the unit's later closers wait for the retry."""
    _live_crew_log.setattr(work_relay_feed, "_RETRY_SECONDS", 0.2)
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    real = work_relay._bound_board
    calls = {"n": 0}

    def once(worker: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return real(worker)

    _live_crew_log.setattr(work_relay, "_bound_board", once)
    _turn(2, 2.0)
    _turn(3, 3.0)
    _wait_for([1.0, 2.0, 3.0])
    work_relay_feed.persist()
    path = work_relay_feed._cursor_path(work_relay_feed._state_home)
    assert json.loads(path.read_text())["units"][WORKER_UNIT] > 0


def test_a_copy_dropped_and_requeued_by_the_writer_settles_once_it_lands(_live_crew_log):
    """The requeued follow-on carries the ``after``, so the cursor moves past it."""
    from kiro_crew.crew_log.writer import WriterLimits

    crew_log_emit.reset_caches(limits=WriterLimits(max_write_attempts=1, retry_backoff_seconds=0.0))
    # Long, so only the requeued copy's own ``after`` can release the hold here.
    _live_crew_log.setattr(work_relay_feed, "_RETRY_SECONDS", 600.0)
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    real = work_relay._append_run
    calls = {"n": 0}

    def fail_once(*args: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk hiccup")
        return real(*args)

    _live_crew_log.setattr(work_relay, "_append_run", fail_once)
    _turn(2, 2.0)
    _wait_for([1.0, 2.0])
    _turn(3, 3.0)
    _wait_for([1.0, 2.0, 3.0])
    with work_relay_feed._cond:
        assert WORKER_UNIT not in work_relay_feed._held


# --------------------------------------------------------------------------- #
# The cursor file is advisory, atomic, idempotent to replay, and has one writer
# --------------------------------------------------------------------------- #


def _cursor_file() -> Any:
    return work_relay_feed._cursor_path(work_relay_feed._state_home)


def _crash_cut_closer(turn: int, credits: float) -> None:
    """Commit a closer the relay never hears about, as a crash before its wake would."""
    with pytest.MonkeyPatch.context() as crash:
        crash.setattr(work_relay_feed, "note", lambda *a, **k: None)
        _turn(turn, credits)


@pytest.mark.parametrize("damage", ["garbage", "stale"])
def test_an_unreadable_or_stale_cursor_falls_back_to_a_rescan(_live_crew_log, damage):
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    work_relay_feed.persist()
    path = _cursor_file()
    _crash_cut_closer(2, 2.0)
    _settle()
    if damage == "garbage":
        path.write_text("{not json", encoding="utf-8")
    else:
        path.write_text(json.dumps({"schema": 1, "units": {WORKER_UNIT: 10_000}}))
    _restart()
    _turn(3, 3.0)
    assert _charges() == [1.0, 2.0, 3.0]


def test_the_cursor_is_replaced_whole_and_a_failed_replace_keeps_the_old_one(
    _live_crew_log,
):
    from kiro_crew import atomic_write as aw

    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    work_relay_feed.persist()
    path = _cursor_file()
    before = path.read_text(encoding="utf-8")
    seen: "dict[str, Any]" = {}
    real = aw.atomic_write

    def spy(target: Any, content: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        real(target, content, **kwargs)

    def no_rename(*_a: Any, **_k: Any) -> None:
        raise OSError("crash before the rename")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aw, "atomic_write", spy)
        patch.setattr(aw, "publish_replacement", no_rename)
        _turn(2, 2.0)
        work_relay_feed.persist()
    assert seen.get("fsync") is True, seen
    assert path.read_text(encoding="utf-8") == before, "a torn or partial cursor was left"
    assert not [p for p in path.parent.iterdir() if p.name not in ("cursor.json", "relay.lock")]


def test_replaying_from_an_old_cursor_bills_each_closer_once(_live_crew_log):
    """A cursor behind what landed replays those entries; nothing is copied or billed twice."""
    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    for turn in (1, 2, 3):
        _turn(turn, float(turn))
    _settle()
    _cursor_file().write_text(json.dumps({"schema": 1, "units": {WORKER_UNIT: 1}}))
    _restart()
    _turn(4, 4.0)
    assert _charges() == [1.0, 2.0, 3.0, 4.0]
    from kiro_crew.crew_log.projection_readers import _slot_units_for_fold

    units = _slot_units_for_fold(LEAD, "workstreams")
    state = crew_log.fold_slot_warm("workstreams", units, slot=LEAD).state
    assert state["spend"][WORKER]["credits"] == 10.0
    assert state["spend"][WORKER]["reported"] == 4


def test_a_second_relayer_is_detected_and_never_writes_the_cursor(_live_crew_log, caplog):
    import logging
    import os

    from kiro_crew.platform_lock_compat import release_lock, try_acquire_lock_or_raise

    _board()
    _unit(WORKER_UNIT, WORKER)
    _bind()
    _turn(1, 1.0)
    work_relay_feed.persist()
    path = _cursor_file()
    before = path.read_text(encoding="utf-8")
    _restart()
    # Another process holds the relay lock for this data home.
    fd = os.open(path.with_name("relay.lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert try_acquire_lock_or_raise(fd, exclusive=True)
        with caplog.at_level(logging.WARNING, logger="kiro_crew.crew_log.work_relay_feed"):
            _turn(2, 2.0)
            work_relay_feed.persist()
        assert "another process holds" in caplog.text
        assert path.read_text(encoding="utf-8") == before
    finally:
        release_lock(fd)
        os.close(fd)
