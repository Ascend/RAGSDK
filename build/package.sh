#!/bin/bash
# CI一键构建脚本.
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

set -e

warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH=$(uname -m)
CUR_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=$(readlink -f "${CUR_PATH}"/..)
PKG_DIR=mindxsdk-mxrag

VERSION_FILE="${ROOT_PATH}"/../mindxsdk/build/conf/config.yaml
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="7.0.RC1"
  fi

  if [ -f "$VERSION_FILE" ]; then
    B_VERSION=$(sed '/.*version_info:/!d;s/.*: //' "$VERSION_FILE")
  else
    B_VERSION="7.0.RC1.B010"
  fi

}

get_version

{
  echo "MindX SDK mxrag:${VERSION}"
  echo "mxrag version:${B_VERSION}"
  echo "Plat: linux $(uname -m)"
} >> "$ROOT_PATH"/version.info


function package()
{
    py_version=$1

    if [ -z "$py_version" ]; then
        echo "python version invalid"
        exit 1
    fi

    cd "${ROOT_PATH}"/output/
    # package
    cp -rf "${ROOT_PATH}"/dist/mx_rag*"${py_version}"*.whl .

    mv "${ROOT_PATH}"/version.info .
    cp -rf "${ROOT_PATH}"/requirements.txt .
    cp -rf "${ROOT_PATH}"/script .

    mkdir -p ./ops/310P
    mkdir -p ./ops/910B
    mkdir -p ./ops/A3
    cp -rf "${ROOT_PATH}"/ops/Ascend910B/lib ./ops/910B
    cp -rf "${ROOT_PATH}"/ops/Ascend910B/lib ./ops/A3
    cp -rf "${ROOT_PATH}"/ops/Ascend310P/lib ./ops/310P

    mkdir -p ./ops/transformer_adapter
    cp -rf "${ROOT_PATH}"/ops/transformer_adapter/* ./ops/transformer_adapter


    cp "${ROOT_PATH}"/build/install.sh .
    cp "${ROOT_PATH}"/build/help.info .

    pkg_version=$(sed -n '1p' version.info |awk -F ':' '{print $2}')

    if [ -z "$pkg_version" ]; then
       echo "get pkg_version failed"
       exit 1
    fi
    sed -i "s/%{PACKAGE_VERSION}%/$pkg_version/g" install.sh
    pkg_arch=$(uname -m)
    sed -i "s/%{PACKAGE_ARCH}%/$pkg_arch/g" install.sh

    #将所有目录设置为750，特殊目录单独处理
    find ./ -type d -exec chmod 750 {} \;
    #将所有文件设置640，特殊文件单独处理
    find ./ -type f -exec chmod 640 {} \;

    find ./  \( -name "*.sh" -o -name "*.run"  -o -name "*.so" \)  -exec  chmod 550 {} \;

    rm -f .gitkeep
}

function main()
{
    package "$1"
}

main "$@"