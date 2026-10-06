#!/bin/bash
set -e
SP=/usr/local/python3.12.13/lib/python3.12/site-packages
cp /root/kvtq_integration/kvtq_vllm_plugin.py $SP/
mkdir -p $SP/kvtq_vllm_plugin-0.1.dist-info
cp /root/kvtq_integration/entry_points.txt $SP/kvtq_vllm_plugin-0.1.dist-info/
cp /root/kvtq_integration/METADATA $SP/kvtq_vllm_plugin-0.1.dist-info/
python3 -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.general_plugins')
print([f'{e.name} -> {e.value}' for e in eps])
"