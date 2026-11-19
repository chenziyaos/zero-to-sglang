#!/usr/bin/env bash
# SGL-22 3.1 节：消费级机器跑 KT 混合路径前的三项主机设置。
# 需要 root。只改当前运行态，要长期生效请写进开机脚本。
set -euo pipefail

# 让 shmem 按需用透明大页，专家池放在 2 MB 页上
echo advise | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled
# 给内核留 8 GB 空闲页，避免大页分配时卡在直接回收
sudo sysctl -w vm.min_free_kbytes=8388608
# 关 swap，专家页被换出去 decode 会掉到个位数
sudo swapoff -a

echo "shmem_enabled: $(cat /sys/kernel/mm/transparent_hugepage/shmem_enabled)"
echo "min_free_kbytes: $(cat /proc/sys/vm/min_free_kbytes)"
echo "swap: $(swapon --show --noheadings | wc -l) 个设备在用"
