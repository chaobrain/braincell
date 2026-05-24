#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CPP=/usr/bin/cpp CC=/usr/bin/cc CXX=/usr/bin/c++ nrnivmodl ion/CdpStC_NoCAM_MA20_GoC.mod
