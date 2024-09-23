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
    VERSION="6.0.RC3"
  fi
}

get_version
echo "MindX SDK mxrag: ${VERSION}" >> "$ROOT_PATH"/version.info


function package()
{
    py_version=$1

    if [ -z "$py_version" ]; then
        echo "python version invalid"
        exit 1
    fi

    ci_package_dir="${ROOT_PATH}"/output/"${PKG_DIR}"/${py_version}/${PKG_DIR}
    # clean
    [ -n "$ci_package_dir" ] && rm -rf "${ci_package_dir}"

    # package
    mkdir -p -m 700 "${ci_package_dir}"
    cp -rf "${ROOT_PATH}"/dist/mx_rag*"${py_version}"*.whl "${ci_package_dir}"

    mv "${ROOT_PATH}"/version.info "${ci_package_dir}"
    cp -rf "${ROOT_PATH}"/requirements.txt "${ci_package_dir}"

    cd "${ci_package_dir}"
    #将所有目录设置为750，特殊目录单独处理
    find ./ -type d -exec chmod 750 {} \;
    #将所有文件设置640，特殊文件单独处理
    find ./ -type f -exec chmod 640 {} \;

    find ./  \( -name "*.sh" -o -name "*.run" \)  -exec  chmod 550 {} \;

    cd ../
    tar -zcvf "${ROOT_PATH}"/output/Ascend-"${PKG_DIR}"_"${VERSION}"_"${py_version}"_linux-"${ARCH}".tar.gz "${PKG_DIR}"
}

function main()
{
    package "$1"
}

main "$@"