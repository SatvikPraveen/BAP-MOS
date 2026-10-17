#!/usr/bin/env bash
# Keep a cluster's planned camera-ready jobs moving as chained short jobs.
#   bash scripts/camera_ready/keeper.sh <ssh-host> <remote-cr-root> <plan-file>
# Each pass submits every plan line that is not done, not queued and whose prerequisite
# exists, as a short job (KEEPER_TIME, default 1 h) that resumes from its checkpoint. Run it
# repeatedly (e.g. every 10 min) from any machine that can ssh to the cluster.
#
# Plan lines:  job-name | script.slurm | KEY=VALUE ... | done-path | [requires-path]
# Paths are relative to the clone (<remote-cr-root>/BAP-MOS); '#' starts a comment.
set -euo pipefail
host="$1"; root="$2"; plan="$3"
ssh -o BatchMode=yes "${host}" "bash -s -- '${root}' '${KEEPER_TIME:-01:00:00}'" < <(
  cat <<'REMOTE'
set -uo pipefail
root="$1"; tlimit="$2"; cd "${root}/BAP-MOS" || exit 1
git pull -q --ff-only || true
queued="$(squeue -u "$USER" -h -o %j)"
n_sub=0; n_done=0; n_wait=0; n_queued=0
while IFS='|' read -r name script kv done_path req; do
  name="$(echo "${name}" | xargs)"; [[ -z "${name}" || "${name}" == \#* ]] && continue
  script="$(echo "${script}" | xargs)"; done_path="$(echo "${done_path}" | xargs)"
  req="$(echo "${req:-}" | xargs)"
  if [[ -e "${done_path}" ]]; then n_done=$((n_done+1)); continue; fi
  if grep -qx "${name}" <<< "${queued}"; then n_queued=$((n_queued+1)); continue; fi
  if [[ -n "${req}" && ! -e "${req}" ]]; then n_wait=$((n_wait+1)); continue; fi
  read -r -a kvs <<< "${kv}"
  SBATCH_ARGS="--job-name=${name} --time=${tlimit}" \
    bash scripts/camera_ready/submit.sh "${script}" "${kvs[@]}" > /dev/null && n_sub=$((n_sub+1))
done
echo "$(hostname -s) $(date +%H:%M): done=${n_done} queued=${n_queued} waiting=${n_wait} submitted=${n_sub}"
REMOTE
  cat "${plan}"
)
