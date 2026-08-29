# 知识图谱预测补全 Demo 部署指导

## 功能描述

本文档指导开发者基于RAG SDK 实现知识图谱预测补全 Demo 部署，提升知识图谱检索召回精度。

## 原理

基于已抽取的知识图谱实体三元组数据，拆分成训练集、验证集，拆分子图作为测试集。 基于[GraphFormers](https://github.com/microsoft/GraphFormers)训练bert模型用于预测补全缺失的实体边关系。

## 部署流程

1. 参考[快速入门](../../docs/zh/03-quickstart.md)部署RAG SDK容器。
2. 在graph_completion目录下克隆GraphFormers 项目。

    ```bash
    git clone https://github.com/microsoft/GraphFormers.git
    ```

3. 拷贝当前样例目录下的代码到GraphFormers目录下

    ```bash
    cp -r *.py *.sh GraphFormers/
    ```

4. 进入GraphFormers 项目目录，执行patch操作。

    ```bash
    cd GraphFormers
    patch -p1 < ./GraphFormers.patch
    ```

## 执行模型训练和补全操作

### 前置工作

1. 根据任务类型选择合适的基础模型权重

   | 模型名                                                                                                                       | 说明            |
   |:--------------------------------------------------------------------------------------------------------------------------|:--------------|
   | [google-bert/bert-base-cased](https://www.modelscope.cn/models/google-bert/bert-base-cased)                               | 只支持英语，区分大小写   |
   | [google-bert/bert-base-uncased](https://www.modelscope.cn/models/google-bert/bert-base-uncased)                           | 只支持英语，不区分大小写  |
   | [google-bert/bert-base-multilingual-cased](https://www.modelscope.cn/models/google-bert/bert-base-multilingual-cased)     | 只支持中英文，区分大小写  |
   | [google-bert/bert-base-multilingual-uncased](https://www.modelscope.cn/models/google-bert/bert-base-multilingual-uncased) | 只支持中英文，不区分大小写 |

2. 准备知识图谱三元组数据，拆分成训练集、验证集，拆分子图作为测试集

    ```bash
    DEVICE_ID=0
    GRAPH_PATH=/home/data/graph.json
    MODEL_DIR=/home/data/bert-base-uncased
    STEP=0
    bash run.sh $STEP $DEVICE_ID $GRAPH_PATH $MODEL_DIR
    ```

3. 训练模型

    ```bash
    DEVICE_ID=0
    GRAPH_PATH=/home/data/graph.json
    MODEL_DIR=/home/data/bert-base-uncased
    STEP=1
    bash run.sh $STEP $DEVICE_ID $GRAPH_PATH $MODEL_DIR
    ```

4. 执行图谱预测补全

    ```bash
    DEVICE_ID=0
    GRAPH_PATH=/home/data/graph.json
    MODEL_DIR=/home/data/bert-base-uncased
    STEP=2
    bash run.sh $STEP $DEVICE_ID $GRAPH_PATH $MODEL_DIR
    ```

5. 对知识图谱实体补全边关系并保存到文件

    ```bash
    DEVICE_ID=0
    GRAPH_PATH=/home/data/graph.json
    MODEL_DIR=/home/data/bert-base-uncased
    STEP=3
    bash run.sh $STEP $DEVICE_ID $GRAPH_PATH $MODEL_DIR
    ```

   补全后知识图谱文件存放在GRAPH_PATH目录下的graph_completion.json文件中

> [!NOTE]
>
>- 如需了解参数配置，执行`python3 graph_completion_demo.py --help`命令查看
