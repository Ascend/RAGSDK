# 快速入门

RAG SDK 提供基于昇腾平台的知识库问答能力，支持文档解析、向量检索、Rerank 等功能。本文通过一个端到端的文生文样例，帮助用户快速完成知识库构建与问答验证。

## 前置条件

开始之前，请确认：

- **硬件**：Atlas 300I Duo 推理卡或Atlas 800I A2/A3 推理服务器，并安装对应的驱动、依赖和固件，详见 [安装部署](./installation_guide.md#容器内部署rag-sdk)
- **Docker**：已安装 Docker，且当前用户可运行容器
- **模型**：参考[链接](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)部署embedding模型bge-large-zh-v1.5
- **LLM 服务**：参考[链接](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Dense.html)部署好LLM模型Qwen3-4B

## 步骤 1：拉取镜像

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11
docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11 ragsdk:26.0.0-910b-ubuntu22.04-py3.11
```

## 步骤 2：启动容器

> [!NOTE] 说明
>
> - `--device /dev/davinci0` 中的设备编号需按宿主机实际 NPU 编号调整
> - `-v /path/to/model:/home/data` 将宿主机模型目录挂载到容器内
> - 容器内示例代码位于 `/workspace/RAGSDK_Samples`

```bash
docker run \
    --name ragsdk_demo \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /path/to/model:/home/data \
    -itd ragsdk:26.0.0-910b-ubuntu22.04-py3.11 bash
```

## 步骤 3：进入容器

```bash
docker exec -it ragsdk_demo bash
```

## 步骤 4：创建测试文档

在工作目录下创建测试文档：

```bash
mkdir -p /workspace/testdata
cat > /workspace/testdata/gaokao.txt << 'EOF'
2024年高考语文作文试题
新课标I卷
阅读下面的材料，根据要求写作。（60分）
随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？
以上材料引发了你怎样的联想和思考？请写一篇文章。
要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。
EOF
```

## 步骤 5：构建知识库

进入示例目录，运行知识库构建脚本：

```bash
cd /workspace/RAGSDK_Samples/rag_with_api
python3 rag_demo_knowledge.py \
    --embedding_url http://127.0.0.1:8080/embed \
    --white_path /workspace \
    --file_path /workspace/testdata/gaokao.txt
```

> [!NOTE]
> http://127.0.0.1:8080为示例url参数，具体url配置以用户本地部署使用参数为准。

### 验证知识库构建成功

若输出以下结果，表示知识库构建成功：

```text
['gaokao.txt']
```

## 步骤 6：执行问答

```bash
python3 rag_demo_query.py \
    --embedding_url http://127.0.0.1:8080/embed \
    --llm_url http://127.0.0.1:1025/v1/chat/completions \
    --model_name Qwen3-4B \
    --query "请描述2024年高考作文题目"
```

> [!NOTE]
> 注意http://127.0.0.1:8080和http://127.0.0.1:1025为示例url参数，具体url配置以用户本地部署使用参数配置为准。

### 验证问答成功

若输出包含检索到的文档内容和生成的回答，表示问答流程运行正常：

```text
{'query': '请描述2024年高考作文题目', 'result': '...", 'source_documents': [...]}
```
