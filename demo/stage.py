"""Everything the laptop runs for the demo, from one command.

The robot runs the demo itself; this side is only models and a screen — four
services: the embedders (SigLIP, bge, faces), the model service (speech
recognition, the language model, the voice), the detector, and the dashboard.
Four terminals is three too many to watch while a robot is talking to an
audience, so this starts all four, prefixes each line with the service that
printed it, waits until each one says it is ready, prints the address the robot
needs, and stops them together on Ctrl-C.

Two things it refuses to do quietly, both of which cost us real time already:

- START ON A PORT SOMETHING ELSE ANSWERS ON. These services bind 0.0.0.0, and
  another process holding 127.0.0.1 on the same port keeps the loopback — so
  the page you open is someone else's app while the service looks fine in its
  own log (seen: a java process on 8080; dashboard pushes got 404).
  Checked by CONNECTING, not by binding, because binding succeeds.
- KEEP GOING WITH A SERVICE MISSING. If one exits, the rest come down too: a
  demo that is quietly missing its memory or its voice is worse on stage than
  one that failed while there was still time to fix it.

With `--robot` it also brings the robot at `--robot-host` up through
scripts/robot_service.sh — camera and microphone, then the voice loop — and
stops them on the way out, so the camera is never left held.

It is also the actor behind the dashboard's Stream button (StreamWatcher
below). The dashboard cannot reach the robot at all; it holds a flag, this
process polls it and runs the same script commands, because this is the
process that already has the script, the brain's address and every port this
run actually used.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import socket
import subprocess
import shutil
import sys
import threading
import time

# How long a service may take to say it is ready. The LLM's warmup dominates:
# measured ~90s cold on the model cache, a few seconds warm.
READY_TIMEOUT_S = 240.0
# How long a service gets to exit on SIGTERM before it is killed.
STOP_GRACE_S = 5.0

_COLOURS = {"embed": "\033[35m", "serve": "\033[36m", "detect": "\033[33m",
            "dash": "\033[32m", "voice": "\033[34m"}
# How long the voice loop may take to say it listens, once the models are up:
# the camera warms up first (demo/run_demo.py's run_voice, up to ~6 s).
VOICE_READY_TIMEOUT_S = 60.0
# How long it gets to exit: it stops the camera and the scene writer (up to
# ~6 s each) before it saves the rest of the conversation — 20 s, as
# scripts/robot_service.sh gives it on the robot, so it is never killed
# mid-save.
VOICE_STOP_GRACE_S = 20.0
# The Reachy Mini daemon's port — the emulator's too (demo/robot_reachy.py).
EMULATOR_PORT = 8000
# demo/run_demo.py's: the voice loop's exit when the dashboard's Restart was
# pressed (a test keeps the two equal; scripts/voice_loop.sh has it too).
RESTART_EXIT_CODE = 42
_RESET = "\033[0m"


@dataclasses.dataclass(frozen=True)
class Service:
    """One Mac-side process: how to start it, and how to tell it is up."""

    name: str
    argv: list[str]
    port: int
    # Printed by the service when it is ready to answer. Matched as a
    # substring on its own output — every one of these prints such a line
    # (see each module's main()), and that is a more honest readiness signal
    # than an open port: serve.py binds BEFORE it warms the models up.
    ready: str
    # Seconds between SIGTERM and SIGKILL on the way out.
    stop_grace: float = STOP_GRACE_S


def build_services(args) -> list[Service]:
    python = sys.executable
    services = [
        Service("embed", [python, "-u", "-m", "demo.embed_service",
                          "--port", str(args.embed_port)]
                + (["--no-warmup"] if args.no_warmup else []),
                args.embed_port, "embed_service on"),
        Service("serve", [python, "-u", "-m", "demo.serve",
                          "--port", str(args.port), "--asr", args.asr]
                + (["--asr-model", args.asr_model] if args.asr_model else [])
                + (["--llm", args.llm] if args.llm else [])
                + (["--no-warmup"] if args.no_warmup else []),
                args.port,
                # Without the warmup there is no "warm in" line to wait for.
                "model service on" if args.no_warmup else "warm in"),
        Service("detect", [python, "-u", "-m", "demo.detect_service",
                           "--port", str(args.detect_port),
                           "--host", "0.0.0.0"],
                args.detect_port, "detect_service on"),
    ]
    if not args.sim:
        # With --sim the voice loop runs here and serves the dashboard itself,
        # on the laptop's own camera; a second one would want the same port.
        services.append(
            Service("dash", [python, "-u", "-m", "demo.display.web",
                             "--port", str(args.web_port),
                             "--robot-host", args.robot_host],
                    args.web_port, "dashboard on"))
    skip = set(args.skip or ())
    return [s for s in services if s.name not in skip]


# Where the voice loop keeps its memory with --sim (its Qdrant Edge shards).
SIM_MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "memory")


def voice_loop_service(args) -> Service:
    """The voice loop on this machine, for --sim: the laptop's camera,
    microphone and speaker, the models just started here, the emulator's
    daemon at --robot-host — and the dashboard, which it serves itself."""
    return Service("voice", [sys.executable, "-u", "-m", "demo.run_demo",
                             "--brain", "127.0.0.1", "--port", str(args.port),
                             "--detect-port", str(args.detect_port),
                             "--embed-port", str(args.embed_port),
                             "--web-port", str(args.web_port),
                             "--robot-host", args.robot_host,
                             "--video", args.video, "--audio", args.audio,
                             "--memory-dir", SIM_MEMORY_DIR],
                   args.web_port, "Speech threshold", stop_grace=VOICE_STOP_GRACE_S)


def wipe_memory(memory_dir: str) -> bool:
    """The dashboard's Restart with --sim, as scripts/voice_loop.sh does it on
    the robot: between two runs (the shards are closed by now), the memory is
    moved aside, not deleted — and only the last one is kept, as -previous.
    False, and why, when it could not be moved."""
    memory_dir = memory_dir.rstrip(os.sep)
    previous = memory_dir + "-previous"
    if not os.path.isdir(memory_dir):
        print(f"  memory: nothing at {memory_dir} to clear", flush=True)
        return True
    try:
        if os.path.exists(previous):
            shutil.rmtree(previous)
        os.rename(memory_dir, previous)
    except OSError as exc:
        print(f"  memory: could not move {memory_dir} aside ({exc})", flush=True)
        return False
    print(f"  memory: cleared — the old one is kept as {previous}", flush=True)
    return True


def start_voice_loop(supervisor, args) -> bool:
    """With --sim: start the voice loop and wait until it listens. Restart
    pressed at any point — even while it is still starting — moves the memory
    aside and starts it again, as scripts/voice_loop.sh does on the robot.
    False, said, when it did not come up."""
    while True:
        supervisor.add(voice_loop_service(args))
        late = supervisor.wait_ready(VOICE_READY_TIMEOUT_S)
        if not late:
            return True
        if (late == ["voice"]
                and supervisor.returncode("voice") == RESTART_EXIT_CODE):
            if not wipe_memory(SIM_MEMORY_DIR):
                return False
            continue
        print(f"  did not come up: {', '.join(late)} — stopping the rest")
        return False


def port_answers(port: int, host: str = "127.0.0.1",
                 timeout: float = 0.3) -> bool:
    """Whether something already answers on this port — see the module
    docstring for why this connects instead of trying to bind."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def lan_address() -> str:
    """This Mac's address on the LAN, which is what the robot must be given as
    --brain. No traffic is sent: a UDP socket only needs a route to pick the
    interface it would use."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def _spawn(service: Service):
    # start_new_session: each service gets its own process group, so stopping
    # it reaches anything it spawned rather than orphaning a grandchild (a
    # `uv run python` wrapper leaves exactly that behind).
    return subprocess.Popen(service.argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)


def _signal_group(process, sig) -> None:
    os.killpg(os.getpgid(process.pid), sig)


class Supervisor:
    """Starts the services, streams their output, stops them together."""

    def __init__(self, services: list[Service], *, spawn=_spawn,
                 send_signal=_signal_group, out=None, colour: bool | None = None) -> None:
        self._services = services
        self._spawn = spawn
        self._send_signal = send_signal
        self._out = out if out is not None else sys.stdout
        self._colour = (self._out.isatty() if colour is None else colour)
        self._processes: dict[str, object] = {}
        self._ready: dict[str, threading.Event] = {}
        self._exited = threading.Event()
        self._dead: list[str] = []
        self._added: set[str] = set()
        # A service started again (add) and the output thread of its last
        # run finishing: one at a time, or the old run's end would mark the
        # new one dead.
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start every service at once — they spend their first seconds
        loading models, and doing that in parallel is the whole point."""
        for service in self._services:
            self._start(service)

    def add(self, service: Service) -> None:
        """Start one more, once the others are up (the voice loop, which
        needs the models answering), or start it again after it exited.
        Stopped before the others on the way out."""
        with self._lock:
            self._services = [s for s in self._services if s.name != service.name]
            self._services.append(service)
            self._added.add(service.name)
            if service.name in self._dead:
                self._dead.remove(service.name)
                if not self._dead:
                    self._exited.clear()
            self._start(service)

    def returncode(self, name: str) -> int | None:
        """How a service ended. Asked once its output has closed — the
        process may still be a moment from being reaped, so it is waited
        for, briefly, rather than read as "still running"."""
        try:
            return self._processes[name].wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def _start(self, service: Service) -> None:
        process = self._spawn(service)
        ready = threading.Event()
        self._processes[service.name] = process
        self._ready[service.name] = ready
        threading.Thread(target=self._pump, args=(service, process, ready),
                         daemon=True).start()

    def wait_ready(self, timeout: float = READY_TIMEOUT_S) -> list[str]:
        """Block until every service has announced itself; returns the names
        of those that did not within `timeout`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._dead:
                break  # one died: say so now, don't wait out the slow ones
            if all(event.is_set() for event in self._ready.values()):
                break
            time.sleep(0.1)
        # A service that exits also "sets" its event (see _pump, which unblocks
        # on a closed stdout), so the dead are named here explicitly.
        return [service.name for service in self._services
                if not self._ready[service.name].is_set()
                or service.name in self._dead]

    def wait(self, poll: float = 0.5) -> str | None:
        """Block until a service exits (returns its name) or the caller is
        interrupted."""
        while not self._exited.wait(poll):
            pass
        return self._dead[0] if self._dead else None

    def stop(self) -> None:
        # Those added after the others stop first, on their own: the voice
        # loop saves what is left of the conversation on the way out, and
        # that needs the embeddings it was started after still answering.
        names = list(self._processes)
        self._stop([name for name in names if name in self._added])
        self._stop([name for name in names if name not in self._added])

    def _stop(self, names: list[str]) -> None:
        for name in names:
            process = self._processes[name]
            if process.poll() is not None:
                continue
            try:
                self._send_signal(process, signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as exc:
                self._write(name, f"could not stop: {type(exc).__name__}: {exc}")
        started = time.monotonic()
        grace = {service.name: service.stop_grace for service in self._services}
        for name in names:
            process = self._processes[name]
            left = max(0.0, started + grace.get(name, STOP_GRACE_S) - time.monotonic())
            try:
                process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                self._write(name, "did not stop in time — killing it")
                try:
                    self._send_signal(process, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    # — output —

    def _pump(self, service: Service, process, ready: threading.Event) -> None:
        for line in process.stdout:
            text = line.rstrip("\n")
            self._write(service.name, text)
            if service.ready in text:
                ready.set()
        # stdout closed: the service is on its way out. Unblock wait_ready so
        # a service that died during startup is reported instead of waited on.
        ready.set()
        with self._lock:
            if self._processes.get(service.name) is not process:
                return  # an earlier run of a service started again since
            self._dead.append(service.name)
            self._exited.set()

    def _write(self, name: str, text: str) -> None:
        prefix = f"{name:>6} | "
        if self._colour:
            prefix = f"{_COLOURS.get(name, '')}{prefix}{_RESET}"
        print(prefix + text, file=self._out, flush=True)


def robot_env(args, brain: str) -> dict[str, str]:
    """What scripts/robot_service.sh reads: which machine is the brain, which
    robot to talk to, and every port this launcher actually used — including
    the dashboard's, or the robot would push its events at the default while
    the dashboard listens somewhere else."""
    return {"BRAIN": brain, "ROBOT": args.robot_host,
            "BRAIN_PORT": str(args.port),
            "DETECT_PORT": str(args.detect_port),
            "EMBED_PORT": str(args.embed_port),
            "WEB_PORT": str(args.web_port)}


# The longest a robot command may take. voice-start is 15-20 s; the first
# `prepare` of a placement installs packages on the robot and takes minutes.
# Bounded all the same: the shutdown runs these with Ctrl-C ignored, and a
# hung ssh there would leave the launcher unkillable.
ROBOT_COMMAND_TIMEOUT_S = 300.0


def _robot(script: str, command: str, env: dict[str, str], out=sys.stdout,
           timeout: float = ROBOT_COMMAND_TIMEOUT_S) -> bool:
    """One scripts/robot_service.sh command. Returns whether it succeeded."""
    print(f"robot | {command}", file=out, flush=True)
    try:
        result = subprocess.run([script, command], env={**os.environ, **env},
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"robot | {command} did not finish in {timeout:.0f} s",
              file=out, flush=True)
        return False
    if result.returncode != 0:
        print(f"robot | {command} failed ({result.returncode})", file=out, flush=True)
    return result.returncode == 0


def _sleep_robot(host: str, out=sys.stdout) -> bool:
    """Play `goto_sleep` on the robot: head down, motors left disabled.

    Over the daemon's own HTTP API (:8000/api) from here, not over SSH the
    way scripts/robot_service.sh wakes it: that script's wake_up is the last
    step of `voice-start`, while sleeping is what happens AFTER the loop and
    the camera service are gone, and there is no command left to hang it on.
    The Mac drove this same API across the network for the whole demo before
    the loop moved onto the robot (demo/run_demo.py's --robot-host), so this
    is the path that is already known to work.

    Imported inside the function: only a run with the robot needs it."""
    from demo.robot_reachy import HttpReachyRobot

    try:
        HttpReachyRobot(host=host).sleep()
        return True
    except Exception as exc:  # noqa: BLE001 — a robot that will not lie down
        # is not a reason to leave the camera held: the stop already happened.
        print(f"robot | could not play goto_sleep ({type(exc).__name__}: {exc})",
              file=out, flush=True)
        return False


class StreamWatcher:
    """The dashboard's Stream button, acted on here.

    The page cannot reach the robot — it holds a flag and this thread polls
    it, exactly as the voice loop polls the Restart flag. A REQUEST, never a
    counter: a counter resets when the dashboard restarts, and a stage that
    kept a baseline from before would read the difference as a press and put
    a talking robot to sleep because someone reloaded the Mac's screen.

    The work is what scripts/robot_service.sh already does — `start` +
    `voice-start` (camera+mic service, motors enabled, wake_up, the loop:
    15-20 s), `voice-stop` + `stop` + goto_sleep the other way — and it runs
    on this thread, so a second press cannot start it twice: the poll that
    would see it does not happen until the first one is finished and
    acknowledged.

    Only a state it is not already in is acted on, so an acknowledgement that
    failed to land (an unreachable dashboard) costs another ack next poll,
    not a second wake-up in front of the room.
    """

    def __init__(self, host: str, port: int, script: str, env: dict[str, str],
                 robot_host: str, *, streaming: bool = False, poll: float = 1.0,
                 run=_robot, sleep_robot=_sleep_robot, out=None) -> None:
        self._host = host
        self._port = port
        self._script = script
        self._env = env
        self._robot_host = robot_host
        self._streaming = streaming
        self._poll = poll
        self._run = run
        self._sleep_robot = sleep_robot
        self._out = out if out is not None else sys.stdout
        self._stop = threading.Event()
        self._acting = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="stream-watch")

    def start(self) -> "StreamWatcher":
        # Say what the robot is doing before anyone touches the button: the
        # dashboard comes up assuming nothing is streaming, and with --robot
        # the robot is already awake by now — the button would offer to start
        # a robot that is running.
        self._ack()
        self._thread.start()
        return self

    def stop(self) -> None:
        """Let a press that is being served right now finish first.

        Ctrl-C two seconds after Stream on would otherwise have this thread's
        `voice-start` and the shutdown's `voice-stop` in two ssh sessions at
        once, and the loop that came up last would be left running — holding
        the camera and the Qdrant shard on a robot that may be in another
        room, the one thing scripts/robot_service.sh is written to prevent."""
        self._stop.set()
        if self._acting.is_set():
            print("  the stream button is still working — one moment",
                  file=self._out, flush=True)
        self._thread.join(timeout=60.0 if self._acting.is_set() else 2.0)

    def streaming(self) -> bool:
        return self._streaming

    def _ack(self) -> None:
        from demo.display.remote import ack_stream

        ack_stream(self._host, self._port, self._streaming)

    def _loop(self) -> None:
        from demo.display.remote import stream_request

        while not self._stop.wait(self._poll):
            asked = stream_request(self._host, self._port)
            if asked is None:
                continue
            if asked != self._streaming:
                self.act(asked)
            self._ack()

    def act(self, on: bool) -> bool:
        """Bring the robot up or put it to sleep; returns the state it is in
        afterwards. Public so the button's work can be tested without a
        thread, and without ever reaching a real robot."""
        self._acting.set()
        try:
            return self._act(on)
        finally:
            self._acting.clear()

    def _act(self, on: bool) -> bool:
        if on:
            started = (self._run(self._script, "start", self._env, out=self._out)
                       and self._run(self._script, "voice-start", self._env,
                                     out=self._out))
            # A `start` that failed leaves the camera+mic service down and
            # voice-start refuses anyway; saying so is what keeps the button
            # from going green over a robot that is not there.
            self._streaming = started
        else:
            # The loop first, then the service it reads from, then the move:
            # the same order scripts/robot_service.sh's own `stop` uses, and
            # goto_sleep last because the loop would otherwise keep moving the
            # head after the robot had lain down.
            self._run(self._script, "voice-stop", self._env, out=self._out)
            self._run(self._script, "stop", self._env, out=self._out)
            self._sleep_robot(self._robot_host, out=self._out)
            self._streaming = False
        return self._streaming


def _stop_on_sigterm() -> None:
    """`kill` on the launcher must take the services with it. Ctrl-C already
    arrives as KeyboardInterrupt and runs the shutdown below; SIGTERM's
    default would end this process on the spot and leave four services
    holding their ports, so make it arrive the same way."""
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, interrupt)
    except ValueError:
        pass  # not the main thread (a test driving main) — nothing to do


def _finish_undisturbed() -> None:
    """From here on, another Ctrl-C or SIGTERM must not cut the shutdown short.
    Seen live: `pkill -f demo.stage` hit both `uv run` and this process, `uv`
    passed its SIGTERM on too, and the second one landed inside voice-stop —
    the robot kept its camera and all four services kept their ports."""
    def already_stopping(signum, frame):
        print("  already stopping — one moment", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, already_stopping)
        except ValueError:
            pass  # not the main thread


def _ensure_face_model(args, services: list[Service]) -> bool:
    """The face embedder has no LiteRT build to download: the first start
    builds it from its PyTorch weights (emulator/models.py), before the
    services, so the first face is not a request that waits two minutes —
    and only when something will use it: the laptop's embed service, or the
    robot when ON_ROBOT places faces there (started with --robot, or later
    from the dashboard's Stream button; never with --sim, which deploys
    nothing). Built or not, the laptop goes on and
    says faces are off; the robot cannot do without it (every deploy would
    stop), so neither does the run. False then."""
    on_robot = (not args.sim
                and "faces" in os.environ.get("ON_ROBOT", "").split(","))
    if not on_robot and not any(service.name == "embed" for service in services):
        return True
    from emulator import models

    try:
        models.fetch("hsface", build=True)
    except Exception as exc:  # noqa: BLE001 — said, and the demo still talks
        if on_robot:
            print(f"faces on the robot need the face embedder: {exc}", flush=True)
            return False
        print(f"  faces off: {exc}", flush=True)
    return True


def _refresh_knowledge() -> None:
    """Rebuild the robot's knowledge snapshot when its facts were edited since
    (demo/knowledge.py). In a child process: the embedding model it loads has
    no business staying in this one for the whole show."""
    from demo import knowledge

    if not knowledge.is_stale():
        return
    print("  the knowledge facts changed — rebuilding the snapshot")
    result = subprocess.run([sys.executable, "-m", "demo.knowledge", "build"],
                            check=False)
    if result.returncode != 0:
        # Said, not stopped on: the robot still starts, with the snapshot it
        # had — which is now older than the facts file.
        print("  WARNING: the snapshot could not be rebuilt (see above); the "
              "robot answers from the old facts until "
              "`uv run python -m demo.knowledge build` succeeds")


def parse_args(argv=None):
    from demo.display import DEFAULT_DASHBOARD_PORT
    from demo.embed_service import DEFAULT_PORT as EMBED_PORT

    p = argparse.ArgumentParser(
        description="Start every laptop-side service for the demo in one place")
    p.add_argument("--port", type=int, default=9500,
                   help="demo/serve.py — speech recognition, the LLM, the voice")
    p.add_argument("--embed-port", type=int, default=EMBED_PORT,
                   help="demo/embed_service.py — SigLIP, bge, faces")
    p.add_argument("--detect-port", type=int, default=9600,
                   help="demo/detect_service.py — the object detector")
    p.add_argument("--web-port", type=int, default=DEFAULT_DASHBOARD_PORT,
                   help="the dashboard, and where the robot pushes its events")
    p.add_argument("--robot-host", default=None,
                   help="the robot: reachy-mini.local by default, the "
                        "emulator on this machine (127.0.0.1) with --sim")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--robot", action="store_true",
                       help="also bring the robot up (scripts/robot_service.sh: "
                            "camera and microphone, then the voice loop) and "
                            "stop it again on the way out")
    where.add_argument("--sim", action="store_true",
                       help="no robot: run the voice loop here too, with this "
                            "machine's camera, microphone and speaker, against "
                            "the Reachy Mini emulator — it serves the dashboard")
    p.add_argument("--video", default="default",
                   help="with --sim: the camera, \"default\" or an ffmpeg "
                        "avfoundation index (passed to demo/run_demo.py)")
    p.add_argument("--audio", default="default",
                   help="with --sim: the microphone, the same way")
    p.add_argument("--llm", default=None, help="passed to demo/serve.py")
    p.add_argument("--asr", choices=("whisper", "moonshine"), default="whisper",
                   help="passed to demo/serve.py")
    p.add_argument("--asr-model", default=None, metavar="NAME",
                   help="passed to demo/serve.py (small.en by default)")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the model warmups; the first turn pays for it")
    p.add_argument("--skip", action="append", metavar="NAME",
                   choices=("embed", "serve", "detect", "dash"),
                   help="do not start this service (repeatable)")
    args = p.parse_args(argv)
    if args.robot_host is None:
        args.robot_host = "127.0.0.1" if args.sim else "reachy-mini.local"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    services = build_services(args)

    if args.sim and not port_answers(EMULATOR_PORT, args.robot_host):
        print(f"no robot answers at {args.robot_host}:{EMULATOR_PORT} — start the "
              "emulator first:\n  mjpython -m reachy_mini.daemon.app.main --sim --no-media")
        return 1
    for service in services + ([voice_loop_service(args)] if args.sim else []):
        if port_answers(service.port):
            why = ("the voice loop's dashboard could not bind it"
                   if service.name == "voice" else
                   f"{service.name} would bind 0.0.0.0 and lose the loopback to it")
            print(f"port {service.port} already answers — something else is "
                  f"running there ({why}). Stop it, or pass a different port.")
            return 1

    _refresh_knowledge()
    if not _ensure_face_model(args, services):
        return 1
    _stop_on_sigterm()
    brain = lan_address()
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "robot_service.sh")
    robot = robot_env(args, brain)
    supervisor = Supervisor(services)
    stream = None
    robot_up = False
    # One try from the first start: a Ctrl-C while the models load, or while
    # the robot is being brought up, stops what was started too — the
    # services run in sessions of their own, so the terminal's Ctrl-C never
    # reaches them, and left running they hold their ports.
    try:
        supervisor.start()
        late = supervisor.wait_ready()
        if late:
            print(f"  did not come up: {', '.join(late)} — stopping the rest")
            return 1
        # The models answer: now the voice loop, which needs them.
        if args.sim and not start_voice_loop(supervisor, args):
            return 1

        dashboard_host = "127.0.0.1" if args.sim else brain
        print(f"\n  ready. dashboard http://{dashboard_host}:{args.web_port}"
              f"   ·   brain {brain}:{args.port}")
        streaming = False
        if args.sim:
            print("  the voice loop runs here, against the emulator — just talk")
        elif args.robot:
            robot_up = True
            streaming = (_robot(script, "start", robot)
                         and _robot(script, "voice-start", robot))
        else:
            print("  the dashboard's ▶ Stream on button brings the robot up, or:")
            print(f"  on the robot:  BRAIN={brain} scripts/robot_service.sh start")
            print(f"                 BRAIN={brain} scripts/robot_service.sh voice-start")
        # Over loopback, whatever --host the dashboard was bound wide on: the
        # dashboard is one of the services started right here.
        stream = (StreamWatcher("127.0.0.1", args.web_port, script, robot,
                                args.robot_host, streaming=streaming).start()
                  if any(service.name == "dash" for service in services) else None)
        print("  Ctrl-C stops everything.\n")

        while True:
            dead = supervisor.wait()
            if args.sim and dead == "voice":
                code = supervisor.returncode("voice")
                if code == RESTART_EXIT_CODE:
                    # Restart from the dashboard: on the robot
                    # scripts/voice_loop.sh does this; here it is the
                    # launcher's job.
                    if not (wipe_memory(SIM_MEMORY_DIR)
                            and start_voice_loop(supervisor, args)):
                        return 1
                    print("  started over — just talk", flush=True)
                    continue
                print(f"\n  the voice loop ended (exit {code}) — stopping the rest")
                return 0 if code == 0 else 1
            if dead:
                print(f"\n  {dead} exited — stopping the rest")
            break
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        _finish_undisturbed()
        if stream is not None:
            stream.stop()
        # `or`: the robot may be up because someone pressed Stream on rather
        # than because of --robot, and leaving here with the camera still
        # held is what strands it until someone SSHes in.
        if robot_up or (stream is not None and stream.streaming()):
            # First, so the robot stops talking to services that are about to
            # go away — and so the camera is released even if this run failed.
            # Then to sleep, as the Stream button does: a robot left with its
            # motors on and head up after the show is not "stopped".
            _robot(script, "voice-stop", robot)
            _robot(script, "stop", robot)
            _sleep_robot(args.robot_host)
        supervisor.stop()
    print("  stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
