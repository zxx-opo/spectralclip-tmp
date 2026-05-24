#!/usr/bin/env bash
# Watchdog: every 10 minutes append a status report to logs/watchdog.log.
LOG=/root/freqtta-hsi/logs/watchdog.log
while true; do
  echo "========== $(date) ==========" >> $LOG
  N=$(ls /root/freqtta-hsi/results/*.json 2>/dev/null | wc -l)
  FAM=$(ls /root/freqtta-hsi/results/indomain_*famssm*.json 2>/dev/null | wc -l)
  CROSS=$(ls /root/freqtta-hsi/results/cross_*.json 2>/dev/null | wc -l)
  ABL=$(ls /root/freqtta-hsi/results/ablation_*.json 2>/dev/null | wc -l)
  MASTER=$(ps -p $(cat /root/freqtta-hsi/logs/master.pid 2>/dev/null) -o stat= 2>/dev/null || echo "DEAD")
  echo "total=$N famssm=$FAM cross=$CROSS abl=$ABL master=$MASTER" >> $LOG
  echo "newest results:" >> $LOG
  ls -t /root/freqtta-hsi/results/*.json 2>/dev/null | head -3 >> $LOG
  echo "master log tail:" >> $LOG
  tail -3 /root/freqtta-hsi/logs/master.log >> $LOG
  echo "" >> $LOG
  sleep 600
done
