#!/bin/bash
# CI一键构建脚本.
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -e

warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH=$(uname -m)
PY_VER=py$(python3.10 -c "import platform; print(platform.python_version().replace('.','')[0:1])")
CUR_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=$(readlink -f "${CUR_PATH}"/..)
SO_OUTPUT_DIR="${ROOT_PATH}"/mx_rag/lib

export CFLAGS="-fstack-protector-strong -fPIC -D_FORTIFY_SOURCE=2 -O2 -ftrapv"
export LDFLAGS="-Wl,-z,relro,-z,now,-z,noexecstack -s"

function clean()
{
    [ -n "$SO_OUTPUT_DIR" ] && rm -rf "$SO_OUTPUT_DIR"
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/dist
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/mx_rag/build
    find "${ROOT_PATH}/mx_rag" -name "*.so" -exec rm {} \;
    echo "clean .so output dir"
    find "${ROOT_PATH}/mx_rag" -name "*.c" -exec rm {} \;
    echo "clean .c output dir"
}

function build_so_package()
{
    echo "prepare .so resource"
    cd "${ROOT_PATH}/mx_rag"
    python3.10 ./setup.py build_ext -j"$(nproc)"
    mkdir -p "${SO_OUTPUT_DIR}"
    cp -arfv build/lib.linux-*/mx_rag/* .
    rm -rf build
}

function build_wheel_package()
{
    cd "${ROOT_PATH}"
    python3.10 ./setup.py bdist_wheel --plat-name linux_"${ARCH}" --python-tag "${PY_VER}"
    echo "prepare resource"
}

function build_ops_package()
{
  cd "${ROOT_PATH}/ops"
  bash build.sh
  cd "${ROOT_PATH}"
}

function package()
{
  bash  "${CUR_PATH}"/package.sh
}

function main()
{
    clean
    build_so_package
    build_wheel_package
    build_ops_package
    package
}

main