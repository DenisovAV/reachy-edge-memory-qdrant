#!/usr/bin/env bash
# Keep the voice loop running, and wipe the robot's memory when the dashboard
# asks for it.
#
#   voice_loop.sh <memory-dir> <command...>
#
# The Restart button on the Mac's dashboard (demo/display/static/dashboard.html
# -> demo/display/web.py's /control) is read by the voice loop's own watcher
# thread (demo/run_demo.py's _RestartWatcher), which ends the run with
# RESTART_EXIT_CODE. That, and ONLY that, is what this supervisor restarts on:
# a crash, a SIGTERM from `robot_service.sh voice-stop`, or Ctrl-C ends the run
# and leaves the memory alone — a restart button must never be the reason a
# demo comes back from the dead after someone deliberately stopped it.
#
# The wipe happens HERE, between two runs, rather than inside the loop: by now
# the process is gone, so its Qdrant Edge shards are closed and their last
# writes are on disk (emulator/edge_store.py — a shard locks its directory for
# as long as it is open), and the fresh run opens a directory nothing is
# holding.
#
# The old memory is MOVED, not deleted, so a wipe in the wrong minute is
# recoverable — but only the last one is kept (`-previous`). The robot's card
# has a few GB free and this button gets pressed between takes, once per demo
# run; a trail of every conversation the robot ever had would fill it.
set -u

MEMORY_DIR="${1:?usage: voice_loop.sh <memory-dir> <command...>}"
shift
[ "$#" -gt 0 ] || { echo "voice_loop.sh: no command given" >&2; exit 2; }

RESTART_EXIT_CODE=42
PREVIOUS="${MEMORY_DIR%/}-previous"

wipe_memory() {
  # This function deletes a directory it was handed on the command line, so
  # it checks the shape of that path before doing anything: an absolute path,
  # at least three segments deep (/home/pollen/reachy-demo/memory), no "..".
  # A typo or an unset variable then refuses loudly instead of taking a home
  # directory with it.
  case "${MEMORY_DIR%/}" in
    *..*) echo "voice_loop.sh: refusing to wipe '$MEMORY_DIR' (..)" >&2; return 1 ;;
    /*/*/*) : ;;
    *) echo "voice_loop.sh: refusing to wipe '$MEMORY_DIR' (too shallow)" >&2
       return 1 ;;
  esac
  if [ ! -d "$MEMORY_DIR" ]; then
    echo "  memory: nothing at $MEMORY_DIR to clear"
    return 0
  fi
  rm -rf "$PREVIOUS"
  if mv "$MEMORY_DIR" "$PREVIOUS"; then
    echo "  memory: cleared — the old one is kept as $PREVIOUS"
  else
    echo "voice_loop.sh: could not move $MEMORY_DIR aside" >&2
    return 1
  fi
}

while :; do
  "$@"
  code=$?
  if [ "$code" -ne "$RESTART_EXIT_CODE" ]; then
    exit "$code"
  fi
  echo
  echo "=== restart from the dashboard ==="
  wipe_memory || exit 1
  echo "=== starting the voice loop again ==="
  echo
done
