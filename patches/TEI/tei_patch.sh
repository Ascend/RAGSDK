#!/bin/bash
set -e

PATCH_DIR=$(cd $(dirname $0); pwd)

# 1. TEI打patch
cd ../../../text-embeddings-inference
patch -p1 < $PATCH_DIR/tei/tei.patch

# 2. transformers打patch
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
patch -p1 $TRANSFORMER_PACKAGE_PATH/models/bert/modeling_bert.py < $PATCH_DIR/ops/transformers_bert.patch
patch -p1 $TRANSFORMER_PACKAGE_PATH/models/xlm_roberta/modeling_xlm_roberta.py < $PATCH_DIR/ops/transformers_xlm_roberta.patch