#!/bin/bash

set -euo pipefail

STEP=${1:-all}
DEVICE_ID=${2:-0}
GRAPH_PATH=${3:-/home/data/graph.json}
MODEL_DIR=${4:-/home/data/bert-base-uncased}

ROOT_DIR="graph_completion_data"

TRAIN_DATA_PATH=${ROOT_DIR}/train.tsv
VALID_DATA_PATH=${ROOT_DIR}/valid.tsv
TEST_DATA_PATH=${ROOT_DIR}/test.tsv
LOAD_CKPT_NAME=${MODEL_DIR}/GraphFormers-best.pt
RESULT_PATH=${ROOT_DIR}/result.json

export MASTER_ADDR="127.0.0.1"   # 主节点IP
export MASTER_PORT="2950"          # 通信端口(空闲即可)



# step 0 创建土木补全的训练集，验证集，和待补全的候选实体对
run_step_0() {
    python3 graph_completion_demo.py --step 0 --graph_path $GRAPH_PATH --tsp_dir $ROOT_DIR --model_dir=$MODEL_DIR
}

# step 1 训练图谱补全模型
run_step_1() {
    ASCEND_RT_VISIBLE_DEVICES=$DEVICE_ID python3 graph_completion_demo.py --step 1 --graph_path $GRAPH_PATH --model_name_or_path=$MODEL_DIR --model_dir=$MODEL_DIR --train_data_path=$TRAIN_DATA_PATH --valid_data_path=$VALID_DATA_PATH --tsp_dir $ROOT_DIR
}

# step 2 预测候选实体缺失的边
run_step_2() {
    ASCEND_RT_VISIBLE_DEVICES=$DEVICE_ID python3 graph_completion_demo.py --step 2 --graph_path $GRAPH_PATH --model_name_or_path=$MODEL_DIR --model_dir=$MODEL_DIR --test_data_path=$TEST_DATA_PATH --load_ckpt_name=$LOAD_CKPT_NAME --result_path=$RESULT_PATH --tsp_dir $ROOT_DIR
}
# step 3 补全边保存到图谱中
run_step_3() {
    ASCEND_RT_VISIBLE_DEVICES=$DEVICE_ID python3 graph_completion_demo.py --step 3 --graph_path=$GRAPH_PATH --result_path=$RESULT_PATH --tsp_dir $ROOT_DIR
}

case $STEP in
    0) run_step_0 ;;
    1) run_step_1 ;;
    2) run_step_2 ;;
    3) run_step_3 ;;
    all)
        run_step_0
        run_step_1
        run_step_2
        run_step_3
        ;;
    *)
        echo "Usage: $0 [0|1|2|3|all] [DEVICE_ID] [GRAPH_PATH] [MODEL_DIR]"
        echo "  DEVICE_ID  : 设备ID (默认: 0)"
        echo "  GRAPH_PATH : 图谱文件路径 (默认: /home/data/graph.json)"
        echo "  MODEL_DIR  : 模型目录 (默认: /home/data/bert-base-uncased)"
        echo "Example: $0 all 1 /path/to/graph.json /path/to/model"
        exit 1
        ;;
esac
