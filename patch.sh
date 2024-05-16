#!/bin/bash
set -e

PATCH_DIR=$(cd $(dirname $0); pwd)

# 1. TEI打patch
cd ../text-embeddings-inference
patch -p1 < $PATCH_DIR/patches/TEI/001_npu_adapter.patch

# 2. transformers打patch
pip install --trusted-host cmc.centralrepo.rnd.huawei.com -i https://cmc.centralrepo.rnd.huawei.com/pypi/simple/ transformers==4.40.2
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
patch -p1 $TRANSFORMER_PACKAGE_PATH/models/bert/modeling_bert.py < $PATCH_DIR/patches/transformers/002_bert.patch
patch -p1 $TRANSFORMER_PACKAGE_PATH/models/xlm_roberta/modeling_xlm_roberta.py < $PATCH_DIR/patches/transformers/003_xlm_roberta.patch
