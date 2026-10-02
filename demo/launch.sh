#!/usr/bin/env bash
# The models on a Raspberry Pi 5 instead of the laptop, with the robot
# simulated on the Mac:
#
#   PI=raspberrypi.local ./demo/launch.sh
#
# Coming soon. This mode is being moved onto the current services
# (demo/stage.py) and tested on the board; until then this command says so
# and stops, rather than start something half-working.
echo "The Raspberry Pi 5 mode is coming soon." >&2
echo "For now the models run on the laptop — see README.md, \"Run it without a robot\"." >&2
exit 2
