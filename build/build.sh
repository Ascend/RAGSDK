#!/bin/bash
# CI一键构建脚本.
# -------------------------------------------------------------------------
# This file is part of the RAGSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# RAGSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

set -e

warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }
ARCH=$(uname -m)
PY311_VER=py$(python3.11 -c "import platform; print(platform.python_version().replace('.','')[0:3])")
CUR_PATH=$(dirname "$(readlink -f "$0")")
ROOT_PATH=$(readlink -f "${CUR_PATH}"/..)
SO_OUTPUT_DIR="${ROOT_PATH}"/mx_rag/lib
TRANSFOMER_ADAPTER_OUTPUT_DIR="${ROOT_PATH}"/ops/transformer_adapter/output

export CFLAGS="-fstack-protector-strong -fPIC -fPIE -O2 -std=c11 -ftrapv -Wall -Wextra -Werror -fno-common"
export LDFLAGS="-s -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack"

function clean()
{
    [ -n "$SO_OUTPUT_DIR" ] && rm -rf "$SO_OUTPUT_DIR"
    [ -n "$TRANSFOMER_ADAPTER_OUTPUT_DIR" ] && rm -rf "$TRANSFOMER_ADAPTER_OUTPUT_DIR"
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/dist
    [ -n "${ROOT_PATH}" ] && rm -rf "${ROOT_PATH}"/mx_rag/build
    find "${ROOT_PATH}/mx_rag" -name "*.so" -exec rm {} \;
    echo "clean .so output dir"
    find "${ROOT_PATH}/mx_rag" -name "*.c" -exec rm {} \;
    echo "clean .c output dir"
}


function update_install_requires()
{
    local py=$1
    ${py} -c '
import re
with open("requirements.txt", "r") as f:
    reqs = []
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
with open("setup.py", "r") as f:
    content = f.read()
new_lines = ["        \"" + r + "\"," for r in reqs]
new_block = "install_requires=[\n" + "\n".join(new_lines) + "\n    ]"
content = re.sub(r"install_requires=\[.*?\]", new_block, content, flags=re.DOTALL)
with open("setup.py", "w") as f:
    f.write(content)
print("setup.py install_requires updated from requirements.txt")
'
}
function build_wheel_package()
{
    local py=$1
    tag=$2
    cd "${ROOT_PATH}"
    update_install_requires "${py}"
    ${py} ./setup.py bdist_wheel --python-tag "${tag}"
    echo "prepare resource"
}

function package()
{
  bash "${CUR_PATH}"/package.sh "$1"
}

function main()
{
    clean
    build_wheel_package python3.11 "${PY311_VER}"
    package "${PY311_VER}"
}

main
