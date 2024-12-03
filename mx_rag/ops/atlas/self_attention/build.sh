#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

SOC_VERSION="$1"
PY_VERSION="$2"

TORCH_PATH=$(${PY_VERSION} -c "import os; import torch; print(os.path.dirname(torch.__file__))")
TORCH_NPU_PATH=$(echo ${TORCH_PATH} | sed "s/"torch"/"torch_npu"/g")

echo "SOC_VERSION ${SOC_VERSION}"
echo "PY_VERSION ${PY_VERSION}"
echo "TORCH_PATH ${TORCH_PATH}"
echo "TORCH_NPU_PATH ${TORCH_NPU_PATH}"

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash

set -e
${PY_VERSION} -m pip install pybind11
rm -rf build
mkdir -p build
cmake -B build \
    -DPY_VERSION=${PY_VERSION} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH} \
    -DTORCH_PATH=${TORCH_PATH} \
    -DTORCH_NPU_PATH=${TORCH_NPU_PATH}
cmake --build build -j