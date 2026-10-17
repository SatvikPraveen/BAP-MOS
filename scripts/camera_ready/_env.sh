#!/usr/bin/env bash
# Shared environment and preflight for camera-ready jobs (sourced by the .slurm files).
#
# CR_ROOT holds the clone (BAP-MOS/), the pinned env (env/), caches and DATA_SHA256.txt:
#   ${CR_ROOT}/BAP-MOS            this repository, branch camera-ready-experiments
#   ${CR_ROOT}/env                python env with requirements.txt pins
#   ${CR_ROOT}/DATA_SHA256.txt    sha256 of every file under data/prostate (written by prep.sh)
#
# Every job verifies the data manifest before starting and refuses to run on a mismatch;
# nothing here rebuilds shared data. All caches and temp files stay under CR_ROOT.
set -euo pipefail

CR_ROOT="${CR_ROOT:-/data/satvikpraveen/camera_ready}"
REPO="${CR_ROOT}/BAP-MOS"
PY="${CR_ROOT}/env/bin/python"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${CR_ROOT}/cache/xdg"
export TORCH_HOME="${CR_ROOT}/cache/torch"
export MPLCONFIGDIR="${CR_ROOT}/cache/mpl"
export PIP_CACHE_DIR="${CR_ROOT}/cache/pip"
export WANDB_DIR="${CR_ROOT}/cache/wandb"
export WANDB_MODE=disabled
export TMPDIR="${CR_ROOT}/cache/tmp"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "${WANDB_DIR}" "${TMPDIR}" \
  "${CR_ROOT}/logs"
cd "${REPO}"

cr_preflight() {
  # Refuse to start unless the pinned data snapshot is unchanged.
  if [[ ! -f "${CR_ROOT}/DATA_SHA256.txt" ]]; then
    echo "preflight: ${CR_ROOT}/DATA_SHA256.txt missing; run scripts/camera_ready/prep.sh" >&2
    exit 3
  fi
  if ! sha256sum --quiet -c "${CR_ROOT}/DATA_SHA256.txt"; then
    echo "preflight: data under ${REPO}/data differs from DATA_SHA256.txt; refusing to run" >&2
    exit 3
  fi
  echo "preflight: data manifest ok ($(wc -l < "${CR_ROOT}/DATA_SHA256.txt") files)"
  echo "preflight: commit $(git -C "${REPO}" rev-parse --short HEAD)" \
    "$(git -C "${REPO}" status --porcelain --untracked-files=no | wc -l) modified tracked files"
  "${PY}" -c "import torch; print('preflight: torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
}
