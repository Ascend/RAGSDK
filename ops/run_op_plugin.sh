#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 导出环境变量
PTA_DIR=op-plugin

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $ASCEND_HOME_DIR/bin/setenv.bash

PYTHON_VERSION=3.10
PYTORCH_VESION=`pip3.10 show torch | grep "Version:" | awk '{print $2}' | awk -F '.' '{print $1"."$2"."$3}' | awk -F '+' '{print $1}'`
export HI_PYTHON=python${PYTHON_VERSION}
export PYTHONPATH=$ASCEND_HOME_DIR/python/site-packages:$PYTHONPATH
export PATH=$ASCEND_HOME_DIR/python/site-packages/bin:$PATH

function main() {
    # 1. PTA自定义算子注册
    cd ${CURRENT_DIR}
    FUNCTION_REGISTE_FIELD="op_plugin_patch/op_plugin_functions.yaml"
    FUNCTION_REGISTE_FILE="${PTA_DIR}/op_plugin/config/v2r1/op_plugin_functions.yaml"
    line="  - func: npu_bert_self_attention_custom(Tensor input, Tensor queryW, Tensor queryBias, Tensor keyW, Tensor keyBias, Tensor valueW, Tensor valueBias, Tensor attentionMask, Tensor dropOutMask, Tensor headMask, int numAttentionHeads, int attentionHeadSize, float dropOutKeepProb) -> Tensor"
    if ! grep -q "\  $line" $FUNCTION_REGISTE_FILE; then
        sed -i "/custom:/r   $FUNCTION_REGISTE_FIELD" $FUNCTION_REGISTE_FILE
    fi

    # 2. 编译PTA插件并安装
    cp -rf op_plugin_patch/*.cpp ${PTA_DIR}/op_plugin/ops/v2r1/opapi
    cd ${PTA_DIR};
    (bash ci/build.sh --python=${PYTHON_VERSION} --pytorch=v$PYTORCH_VESION ; pip3.10 uninstall torch-npu -y ; pip3.10 install dist/*.whl)

}
main
