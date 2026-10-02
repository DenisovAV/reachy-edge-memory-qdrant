"""demo/launch.sh — the Raspberry Pi 5 mode, named in the README and the
article before it lands. Until it does, the command must stop loudly: never
look like a run that worked."""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "launch.sh"


def test_the_pi5_command_says_it_is_coming_rather_than_half_start():
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                            timeout=30, env={"PI": "raspberrypi.local", "PATH": "/usr/bin:/bin"})
    assert result.returncode != 0
    assert "coming soon" in result.stderr
