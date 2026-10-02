"""scripts/voice_loop.sh: the supervisor the robot's voice loop runs under.

It is three lines of bash that DELETE a directory, so it gets tests: the
Restart button on the dashboard is the only thing that may erase what the
robot remembers, and a crash, a `voice-stop` or a typo in the path must all
leave the memory exactly where it is.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "voice_loop.sh"

# What demo/run_demo.py exits with when the dashboard asked for a restart.
RESTART_EXIT_CODE = 42


def test_the_restart_exit_code_is_the_same_number_on_both_sides():
    """Bash and Python each hold their own copy of it: the loop exits with
    demo/run_demo.py's, the supervisor restarts on this one. Drift between
    them means the button stops the demo instead of restarting it."""
    from demo.run_demo import RESTART_EXIT_CODE as FROM_THE_LOOP

    assert f"RESTART_EXIT_CODE={FROM_THE_LOOP}\n" in SCRIPT.read_text()
    assert FROM_THE_LOOP == RESTART_EXIT_CODE


def _loop(tmp_path: Path, exits: list[int]) -> Path:
    """A stand-in for the voice loop that exits with `exits` in order."""
    runs = tmp_path / "runs"
    fake = tmp_path / "fake_loop.sh"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f"codes=({' '.join(str(code) for code in exits)})\n"
        f'runs="{runs}"\n'
        'n=$(cat "$runs" 2>/dev/null || echo 0)\n'
        'echo $((n + 1)) > "$runs"\n'
        'exit "${codes[$n]}"\n')
    fake.chmod(0o755)
    return fake


def _run(memory_dir: Path, fake: Path):
    return subprocess.run(["bash", str(SCRIPT), str(memory_dir), "bash", str(fake)],
                          capture_output=True, text=True, timeout=30)


def _memory(tmp_path: Path) -> Path:
    memory = tmp_path / "reachy-demo" / "memory"
    (memory / "memory").mkdir(parents=True)
    (memory / "memory" / "wal").write_text("what the robot heard")
    return memory


def test_a_restart_wipes_the_memory_and_starts_the_loop_again(tmp_path):
    memory = _memory(tmp_path)
    result = _run(memory, _loop(tmp_path, [RESTART_EXIT_CODE, 0]))

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "runs").read_text().strip() == "2", "the loop came back"
    assert not memory.exists(), "the memory was cleared"
    # Moved, not deleted: a wipe in the wrong minute is recoverable.
    previous = memory.parent / "memory-previous"
    assert (previous / "memory" / "wal").read_text() == "what the robot heard"


def test_only_the_last_wipe_is_kept(tmp_path):
    """The robot's card has a few GB free and this button gets pressed between
    takes — a trail of every conversation it ever had would fill it."""
    memory = _memory(tmp_path)
    stale = memory.parent / "memory-previous"
    stale.mkdir()
    (stale / "older.txt").write_text("from two demos ago")

    assert _run(memory, _loop(tmp_path, [RESTART_EXIT_CODE, 0])).returncode == 0
    assert not (stale / "older.txt").exists()
    assert (stale / "memory" / "wal").exists()


def test_any_other_exit_ends_the_run_with_the_memory_untouched(tmp_path):
    """`voice-stop`, Ctrl-C, a crash: a restart button must never be the
    reason a demo comes back after someone deliberately stopped it."""
    memory = _memory(tmp_path)
    result = _run(memory, _loop(tmp_path, [3]))

    assert result.returncode == 3
    assert (tmp_path / "runs").read_text().strip() == "1", "it did not come back"
    assert (memory / "memory" / "wal").exists()
    assert not (memory.parent / "memory-previous").exists()


def test_a_path_that_is_not_a_memory_directory_is_refused(tmp_path):
    """The path comes from a shell variable in scripts/robot_service.sh — an
    unset one must refuse loudly, not take a home directory with it."""
    fake = _loop(tmp_path, [RESTART_EXIT_CODE, 0])
    for path in ("/tmp", "memory", "/home/pollen/../.."):
        result = subprocess.run(["bash", str(SCRIPT), path, "bash", str(fake)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode != 0, path
        assert "refusing to wipe" in result.stderr, path
        (tmp_path / "runs").unlink(missing_ok=True)


def test_it_refuses_to_run_with_no_command(tmp_path):
    result = subprocess.run(["bash", str(SCRIPT), str(tmp_path / "memory")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "no command" in result.stderr
