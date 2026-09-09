#!/usr/bin/env bash
set -eu
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
python - <<'PYTHON'
from validation.neuron._mechanisms import compile_mechanisms
from validation.neuron._paths import model_dir
print(compile_mechanisms("goc_ma2020", files=[
    model_dir("goc_ma2020") / "mechanisms/ion/CdpStC_CAMOnly_MA20_GoC.mod",
]))
PYTHON
