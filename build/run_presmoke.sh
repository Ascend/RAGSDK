#!/bin/bash
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

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")
readonly RUN_PKG_PATH="${CUR_DIR}/../.."
readonly CODE_PATH="${CUR_DIR}/.."
readonly PRESMOKE_DIR="/home/ragSDK/preSmokeTestFiles"
echo "CUR_DIR: $CUR_DIR"
echo "RUN_PKG_PATH: $RUN_PKG_PATH"
echo "PRESMOKE_DIR: $PRESMOKE_DIR"
echo "CODE_PATH: $CODE_PATH"

cd $CODE_PATH
changed=$(git diff ${servicebranch_1} --no-commit-id --name-only)
echo "$changed" > changed_files.txt

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 安装依赖
apt-get update -y
apt-get install -y  libpq-dev 
pip3 install uvicorn

# 安装rag
cp ${RUN_PKG_PATH}/Ascend-mindxsdk-mxrag_*_linux-aarch64.run ${PRESMOKE_DIR}/pkg/
cd ${PRESMOKE_DIR}/pkg/
chmod +x *.run
./Ascend-mindxsdk-mxrag_*_linux-aarch64.run --install --install-path=/usr/local/Ascend --platform=910B
pip3 install -r  /usr/local/Ascend/mxRag/requirements.txt
pip3 install pytest pytest-cov pytest-html

cd $CODE_PATH
# 清理milvus数据库
python3 tests/presmoke/clean_milvus_collections.py
# 起模拟模型和embed服务
python3 tests/presmoke/emb_model_service.py > /dev/null 2>&1 &
API_PID=$!
sleep 1
# 执行demo
export MX_INDEX_FINALIZE=0
python3 tests/presmoke/map_presmoke_list.py
cat map_presmoke_list.txt | xargs python3 -m pytest -s
kill $API_PID 2>/dev/null
