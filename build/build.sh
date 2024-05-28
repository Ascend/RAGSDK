#!/bin/bash
# CI一键构建脚本.
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -e

ARCH=$(uname -m)
PY_VER=py$(python3 -c "import platform; print(platform.python_version().replace('.','')[0:2])")
CUR_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=$(readlink -f "${CUR_PATH}"/..)
CI_PACKAGE_DIR="${ROOT_PATH}"/output/${PY_VER}
OUTPUT_DIR="${ROOT_PATH}"/_package_output_${PY_VER}
SO_OUTPUT_DIR="${ROOT_PATH}"/mx_rag/lib

export CFLAGS="-fstack-protector-strong -fPIC -D_FORTIFY_SOURCE=2 -O2 -ftrapv"
export LDFLAGS="-Wl,-z,relro,-z,now,-z,noexecstack -s"

function clean()
{
    rm -rf "$OUTPUT_DIR"
    rm -rf "$SO_OUTPUT_DIR"
    rm -rf "$CI_PACKAGE_DIR"
    rm -rf "${ROOT_PATH}"/dist
    rm -rf "${ROOT_PATH}"/mx_rag/build
    find "${ROOT_PATH}/mx_rag" -name "*.so" -exec rm {} \;
    echo "clean .so output dir"
    find "${ROOT_PATH}/mx_rag" -name "*.c" -exec rm {} \;
    echo "clean .c output dir"
}

function package()
{
    # copy package
    mkdir -p "${OUTPUT_DIR}"
    cp -r "${ROOT_PATH}"/dist/* "${OUTPUT_DIR}"/
    echo "copy mx_rag package to output dir."

    # clean
    if [ -d "${CI_PACKAGE_DIR}" ]; then
        rm -rf "${CI_PACKAGE_DIR}"
    fi
    echo "clean ci package output dir"

    # package
    mkdir -p -m 700 "${CI_PACKAGE_DIR}"
    echo "start build output package"
    cd "${OUTPUT_DIR}"
    cp -r ./* "${CI_PACKAGE_DIR}"
    chmod 400 "${CI_PACKAGE_DIR}"/*
    python3 -m twine check "${CI_PACKAGE_DIR}"/mx_rag*.whl
}

function build_so_package()
{
    echo "prepare .so resource"
    cd "${ROOT_PATH}/mx_rag"
    python3 ./setup.py build_ext
    mkdir -p "${SO_OUTPUT_DIR}"
    cp -r build/lib.linux-*/mx_rag/document/loader/* document/loader/
}

function build_wheel_package()
{
    cd "${ROOT_PATH}"
    python3 ./setup.py bdist_wheel --plat-name linux_"${ARCH}" --python-tag "${PY_VER}"
    echo "prepare resource"
}

function main()
{
    clean
    build_so_package
    build_wheel_package
    package
}

main