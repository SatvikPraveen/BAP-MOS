#!/usr/bin/env bash
# Remote half of keeper.sh: reads the plan on stdin and submits what is due.
#   bash keeper_remote.sh <cr-root> <time-limit> < plan.txt
set -uo pipefail
root="$1"; tlimit="$2"; cd "${root}/BAP-MOS" || exit 1
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
    bash scripts/camera_ready/submit.sh "${script}" "${kvs[@]}" < /dev/null > /dev/null && n_sub=$((n_sub+1))
done
echo "$(hostname -s) $(date +%H:%M): done=${n_done} queued=${n_queued} waiting=${n_wait} submitted=${n_sub}"
