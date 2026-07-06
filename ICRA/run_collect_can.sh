#!/usr/bin/env bash
# Convenience launcher for the PickPlaceCan keyboard demo collector.
#
# Runs the collector with the visual target markers for milk/bread/cereal
# hidden (the can is kept). Any extra args you pass are forwarded to the
# Python script, e.g.:  ./run_collect_can.sh --pos-sensitivity 1.0
#
# NOTE: activate the conda env that has robosuite first, e.g.:
#           conda activate robosuite
#       so that `python` resolves to the interpreter with robosuite 1.5.
set -euo pipefail

# Run from the directory this script lives in, so relative paths (./demos/...)
# are stable regardless of where you invoke it from.
cd "$(dirname "$0")"

python collect_pickplace_can.py --hide-other-targets "$@"
