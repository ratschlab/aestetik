#!/usr/bin/env bash
set -euo pipefail
pip uninstall -y aestetik
rm -rf dist/ build/ src/aestetik.egg-info/ aestetik.egg-info/
pip install -e .
