#!/bin/bash
# Training progress monitoring script

echo "=== Training Progress Monitor ==="
echo "PID: $(cat train_full.pid 2>/dev/null || echo 'Not found')"
echo ""

# Check if process is running
if ps -p $(cat train_full.pid 2>/dev/null) > /dev/null 2>&1; then
    echo "Status: RUNNING"
    ps aux | grep $(cat train_full.pid) | grep -v grep | awk '{print "CPU: " $3 "%, Memory: " $4 "%"}'
else
    echo "Status: NOT RUNNING"
fi
echo ""

# Check log file size
if [ -f "train_full.log" ]; then
    echo "Log file size: $(ls -lh train_full.log | awk '{print $5}')"
fi
echo ""

# Check TensorBoard logs
echo "TensorBoard logs:"
ls -lh runs/streamvc_cpu_full/logs/*.0 2>/dev/null | tail -1 | awk '{print "Latest: " $9 " (" $5 ")"}'
echo ""

# Check checkpoints
echo "Checkpoints:"
ckpt_count=$(ls runs/streamvc_cpu_full/checkpoints/*.pt 2>/dev/null | wc -l)
echo "Total: $ckpt_count"
if [ $ckpt_count -gt 0 ]; then
    ls -lht runs/streamvc_cpu_full/checkpoints/*.pt 2>/dev/null | head -3
fi
echo ""

# Estimate progress (rough estimate based on log file growth)
if [ -f "train_full.log" ]; then
    echo "Recent log entries:"
    tail -5 train_full.log
fi
