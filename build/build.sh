#!/bin/bash
# CI一键构建脚本.
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -e

warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH=$(uname -m)
PY_VER=py$(python3 -c "import platform; print(platform.python_version().replace('.','')[0:1])")
CUR_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=$(readlink -f "${CUR_PATH}"/..)
PKG_DIR=mindxsdk-mxrag
CI_PACKAGE_DIR="${ROOT_PATH}"/output/"${PKG_DIR}"
OUTPUT_DIR="${ROOT_PATH}"/_package_output_${PY_VER}
SO_OUTPUT_DIR="${ROOT_PATH}"/mx_rag/lib

VERSION_FILE="${ROOT_PATH}"/../mindxsdk/build/conf/config.yaml
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="6.0.RC2"
  fi
}

get_version
echo "MindX SDK mxrag: ${VERSION}" >> "$ROOT_PATH"/version.info
RELEASE_TAR=Ascend-"${PKG_DIR}"_"${VERSION}"_linux-"${ARCH}".tar.gz

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
    cp *.whl "${CI_PACKAGE_DIR}"

    cp -r "${ROOT_PATH}"/patches "${CI_PACKAGE_DIR}"

    mv "${ROOT_PATH}"/version.info "${CI_PACKAGE_DIR}"

    cd "${CI_PACKAGE_DIR}"
    chmod 440 version.info
    chmod 550 *.whl
    find patches \( -name "*.md" -o -name "*.patch" \) -exec chmod 440 {} \;
    chmod 550 `find patches -type d`
    find patches -name "*.sh" -exec chmod 550 {} \;

    cd ../
    tar -zcvf "${RELEASE_TAR}" "${PKG_DIR}" || {
      warn "compression failed, packages might be broken"
    }
}

function build_so_package()
{
    echo "prepare .so resource"
    cd "${ROOT_PATH}/mx_rag"
    python3 ./setup.py build_ext -j"$(nproc)"
    mkdir -p "${SO_OUTPUT_DIR}"
    cp -arfv build/lib.linux-*/mx_rag/* .
    rm -rf build
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