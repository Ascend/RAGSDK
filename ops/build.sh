#!/bin/bash
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -e

CUR_DIR=$(dirname "$(readlink -f "$0")")

TARGET_PLATFORM="$1"
PY_VERSION="$2"

if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi
mkdir "build"

if [ -d "$CUR_DIR/$TARGET_PLATFORM" ]; then
    echo "Removing existing build directory..."
    rm -rf $CUR_DIR/$TARGET_PLATFORM
fi
mkdir -p $CUR_DIR/$TARGET_PLATFORM

cd build

export PYTHON_INCLUDE_PATH="$($PY_VERSION -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$($PY_VERSION  -c 'from sysconfig import get_paths; print(get_paths()["platlib"])')"
export PYTORCH_INSTALL_PATH="$($PY_VERSION -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export PYTORCH_NPU_INSTALL_PATH="$PYTORCH_INSTALL_PATH/../torch_npu"

cmake -DTARGET_PLATFORM:string=$TARGET_PLATFORM  -DCMAKE_INSTALL_PREFIX=$CUR_DIR/$TARGET_PLATFORM ..

make -j"$(nproc)"
make install
cd ..