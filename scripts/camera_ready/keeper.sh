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
# The remote half is a file in the clone, so stdin carries only the plan lines.
ssh -o BatchMode=yes "${host}" \
  "cd '${root}/BAP-MOS' && git pull -q --ff-only; bash scripts/camera_ready/keeper_remote.sh '${root}' '${KEEPER_TIME:-01:00:00}'" \
  < "${plan}"
