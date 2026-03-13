#!/bin/bash
# Wait for the current training (PID 3805124) to finish, then start multi-epoch training.

WAIT_PID=3805124
LOG_FILE="train_log_continued.txt"

echo "[$(date)] Waiting for PID $WAIT_PID to finish..."

# Poll until the process exits
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $WAIT_PID finished. Sleeping 30s to let GPU memory release..."
sleep 30

echo "[$(date)] Starting multi-epoch training (epochs 2-10)..."

cd /Data0/swangek_data/991/CPRD/examples/modelling/SurvivEHR

/usr/bin/env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH=/Data0/swangek_data/991/FastEHR:/Data0/swangek_data/991/CPRD \
  /Data0/swangek_data/conda_envs/survivehr/bin/python \
  run_experiment.py --config-name=config_CompetingRisk11M \
  data.batch_size=8 optim.accumulate_grad_batches=2 optim.num_epochs=10 \
  2>&1 | tee "$LOG_FILE"

echo "[$(date)] Training completed with exit code $?"
