#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/tmp/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
