#!/bin/bash
# CI一键构建脚本.
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -e

warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH=$(uname -m)
PY_VER=py$(python3.10 -c "import platform; print(platform.python_version().replace('.','')[0:1])")
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
    [ -n "$OUTPUT_DIR" ] && rm -rf "$OUTPUT_DIR"
    [ -n "$SO_OUTPUT_DIR" ] && rm -rf "$SO_OUTPUT_DIR"
    [ -n "$CI_PACKAGE_DIR" ] && rm -rf "$CI_PACKAGE_DIR"
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/dist
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/mx_rag/build
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

    if [ ! -d "${CI_PACKAGE_DIR}/ops" ]; then
      mkdir -p "${CI_PACKAGE_DIR}/ops"
    fi

    OPP_RUN_NAME=$(basename "${ROOT_PATH}"/ops/BertSelfAttention/build_out/custom_opp_*.run)
    if [[ "$OPP_RUN_NAME" == *"aarch64"* ]]; then
      OPP_RUN_NAME=$( echo "${OPP_RUN_NAME}" | awk -F'_' '{print $1 "_" $2 "_" $4 }')
    else
      OPP_RUN_NAME=$( echo "${OPP_RUN_NAME}" | awk -F'_' '{print $1 "_" $2 "_" $4 ".run"}')
    fi

    cp "${ROOT_PATH}"/ops/BertSelfAttention/build_out/custom_opp_*.run "${CI_PACKAGE_DIR}"/ops/"${OPP_RUN_NAME}"
    cp "${ROOT_PATH}"/ops/run_op_plugin.sh "${CI_PACKAGE_DIR}"/ops
    cp -r "${ROOT_PATH}"/ops/op_plugin_patch "${CI_PACKAGE_DIR}"/ops
    chmod -R 550 "${CI_PACKAGE_DIR}"/ops/"${OPP_RUN_NAME}"
    chmod -R 550 "${CI_PACKAGE_DIR}"/ops/run_op_plugin.sh

    mv "${ROOT_PATH}"/version.info "${CI_PACKAGE_DIR}"
    cp "${ROOT_PATH}"/requirements.txt "${CI_PACKAGE_DIR}"

    cd "${CI_PACKAGE_DIR}"
    #将所有目录设置为750，特殊目录单独处理
    find ./ -type d -exec chmod 750 {} \;
    #将所有文件设置为440，特殊文件单独处理
    find ./ -type f -exec chmod 440 {} \;

    find ./  \( -name "*.sh" -o -name "*.run" \)  -exec  chmod 550 {} \;

    find ops -type d -exec chmod 550 {} \;

    cd ../
    tar -zcvf "${RELEASE_TAR}" "${PKG_DIR}" || {
      warn "compression failed, packages might be broken"
    }
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

function main()
{
    clean
    build_so_package
    build_wheel_package
    build_ops_package
    package
}

main