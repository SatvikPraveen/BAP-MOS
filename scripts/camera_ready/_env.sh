#!/usr/bin/env bash
# Shared environment and preflight for camera-ready jobs (sourced by the .slurm files).
#
# CR_ROOT holds the clone (BAP-MOS/), the pinned envs (env/, env_nnunet/), caches and manifests:
#   ${CR_ROOT}/BAP-MOS            this repository, branch camera-ready-experiments
#   ${CR_ROOT}/env                python env with requirements.txt pins
#   ${CR_ROOT}/DATA_SHA256_{prostate,pfus1}.txt   sha256 of every data file (prep.sh pin)
#
# Every job verifies the data manifest before starting and refuses to run on a mismatch;
# nothing here rebuilds shared data. All caches and temp files stay under CR_ROOT.
set -euo pipefail

# Default: the folder that holds this clone (works on every cluster; submit.sh sets it too).
CR_ROOT="${CR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
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

# One pinned manifest per corpus, so a job only hashes the data it reads.
cr_manifest() {
  if [[ "${DATASET:-pooled}" == pfus1 ]]; then
    echo "${CR_ROOT}/DATA_SHA256_pfus1.txt"
  else
    echo "${CR_ROOT}/DATA_SHA256_prostate.txt"
  fi
}

cr_preflight() {
  # Refuse to start unless the pinned data snapshot is unchanged.
  local manifest
  manifest="$(cr_manifest)"
  if [[ ! -f "${manifest}" ]]; then
    echo "preflight: ${manifest} missing; run scripts/camera_ready/prep.sh pin" >&2
    exit 3
  fi
  # Full hashing takes ~30 min on network storage, so it runs once: a passing check records
  # the manifest hash plus a name/size/mtime fingerprint of every listed file, and later jobs
  # rehash only if either differs.
  local stamp="${manifest}.verified" want
  want="$(sha256sum < "${manifest}" | cut -c1-64) $(cut -c67- "${manifest}" | tr '\n' '\0' \
    | xargs -0 stat -c '%n %s %Y' | sha256sum | cut -c1-64)"
  if [[ "$(cat "${stamp}" 2>/dev/null)" == "${want}" ]]; then
    echo "preflight: data manifest ok ($(wc -l < "${manifest}") files, ${manifest##*/}, fingerprint matches verified hash)"
  else
    if ! sha256sum --quiet -c "${manifest}"; then
      echo "preflight: data under ${REPO}/data differs from ${manifest}; refusing to run" >&2
      exit 3
    fi
    echo "${want}" > "${stamp}.$$" && mv -f "${stamp}.$$" "${stamp}"
    echo "preflight: data manifest ok ($(wc -l < "${manifest}") files, ${manifest##*/}, hashed)"
  fi
  echo "preflight: commit $(git -C "${REPO}" rev-parse --short HEAD)" \
    "$(git -C "${REPO}" status --porcelain --untracked-files=no | wc -l) modified tracked files"
  # Run real kernels: some nodes report CUDA available but their driver fails cu129 kernels.
  "${PY}" -c "import torch; print('preflight: torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); x=torch.randn(2,3,64,64,device='cuda'); torch.nn.functional.conv2d(x, torch.randn(4,3,3,3,device='cuda')).sum().item(); print('preflight: cuda kernels ok')" \
    || { echo "preflight: CUDA kernels fail on $(hostname -s); exclude this node" >&2; exit 4; }
}

# PFUS1 validation scores ~2.2k frames x 8 organs per epoch; score them in parallel worker
# processes (identical, order-preserving results; see metrics/evaluator.py).
if [[ "${DATASET:-}" == pfus1 ]]; then
  export BAPMOS_METRIC_WORKERS="${BAPMOS_METRIC_WORKERS:-12}"
fi
