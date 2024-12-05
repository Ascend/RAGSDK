#!/bin/bash
# CI一键构建脚本.
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -e

warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH=$(uname -m)
PY310_VER=py$(python3.10 -c "import platform; print(platform.python_version().replace('.','')[0:3])")
PY311_VER=py$(python3.11 -c "import platform; print(platform.python_version().replace('.','')[0:3])")
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
    py=$1
    find "${ROOT_PATH}/mx_rag"  \( -name "*.so" -o -name "*.c" \) -exec  rm -f {} \;

    cd "${ROOT_PATH}/mx_rag"
    ${py} ./setup.py build_ext -j"$(nproc)"
    mkdir -p "${SO_OUTPUT_DIR}"
    cp -arfv build/lib.linux-*/mx_rag/* .
    rm ./setup*.so
    rm ./version*.so
    rm -rf build
}

function build_wheel_package()
{
    py=$1
    tag=$2
    cd "${ROOT_PATH}"
    ${py} ./setup.py bdist_wheel --plat-name linux_"${ARCH}" --python-tag "${tag}"
    echo "prepare resource"
}

function package()
{
  bash "${CUR_PATH}"/package.sh "$1"
}

function main()
{
    clean
    build_so_package python3.10
    build_wheel_package python3.10 "${PY310_VER}"
    package "${PY310_VER}"
    build_so_package python3.11
    build_wheel_package python3.11 "${PY311_VER}"
    package "${PY311_VER}"
}

main