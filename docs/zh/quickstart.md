# 快速入门

RAG SDK 提供基于昇腾平台的知识库问答能力，支持文档解析、向量检索、Rerank 等功能。本文通过一个端到端的文生文样例，帮助用户快速完成知识库构建与问答验证。

## 前置条件

开始之前，请确认：

- **硬件**：Atlas 300I Duo 推理卡或Atlas 800I A2/A3 推理服务器，并安装对应的驱动、依赖和固件
- **Docker**：已安装 Docker，且当前用户可运行容器
- **向量模型服务**：参考[mis-tei文档](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)部署好embedding模型bge-large-zh-v1.5
- **大模型服务**：参考[Qwen3-Dense文档](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Dense.html)部署好LLM模型Qwen3-4B

## 步骤 1：拉取镜像

1. **确定待下载镜像版本**
   - 访问昇腾社区[镜像仓](https://www.hiascend.com/developer/ascendhub/detail/b875f781df984480b0385a96fa1b03c9)，查看RAG SDK镜像配套表，获取镜像最新版本以及与之配套的CANN版本
   - 根据当前硬件型号（如 Atlas 800I A2 推理服务器）选择对应版本

    > [!NOTE]
    > 镜像中已安装CANN，无需重复安装<br>
    > 注意区分 CPU 架构（x86_64 / aarch64）

2. **环境预检查**
   - 执行 `npu-smi info` 命令查看当前环境安装的 NPU 驱动版本
   - 通过RAG SDK镜像配套表中获取到的配套CANN版本去[固件与驱动文档](https://www.hiascend.com/hardware/firmware-drivers/community)中查看对应的NPU驱动版本，如果和当前环境安装的驱动版本不配套，请更新NPU驱动至对应版本，NPU驱动更新指导详见[驱动和固件安装指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100568434/36e8d875?idPath=23710424|251366513|254884019|261408772|252764743)。

3. **镜像拉取示例**

   镜像 Tag 格式为 `{version}-{chip}-{os}-{python}`，各变量含义如下：

   | 变量 | 含义         | 示例值 |
   |------|------------|--------|
   | `{version}` | RAG SDK 版本 | `26.0.0` |
   | `{chip}` | 昇腾芯片系列     | `910b` |
   | `{os}` | 基础操作系统     | `ubuntu22.04` / `openeuler24.03` |
   | `{python}` | Python 版本  | `py3.11` |

   ```bash
   TAG={version}-{chip}-{os}-{python}
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:${TAG}
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:${TAG} \
       ragsdk:${TAG}
   ```

   以 26.0.0 版本、910b 芯片、Ubuntu 22.04、Python 3.11为例：

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11 ragsdk:26.0.0-910b-ubuntu22.04-py3.11
    ```

## 步骤 2：启动容器

> [!NOTE] 说明
>
> - `--device /dev/davinci0` 中的设备编号需按宿主机实际 NPU 编号调整
> - `-v /path/to/model:/home/data` 挂载宿主机目录到容器（可选）
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
    --embedding_url http://127.0.0.1:8080/v1/embeddings \
    --white_path /workspace \
    --file_path /workspace/testdata/gaokao.txt
```

> [!NOTE]
> <http://127.0.0.1:8080>为示例url参数，具体url配置以用户本地部署使用参数为准。

### 验证知识库构建成功

若输出以下结果，表示知识库构建成功：

```text
['gaokao.txt']
```

## 步骤 6：执行问答

```bash
python3 rag_demo_query.py \
    --embedding_url http://127.0.0.1:8080/v1/embeddings \
    --llm_url http://127.0.0.1:1025/v1/chat/completions \
    --model_name Qwen3-4B \
    --query "请描述2024年高考作文题目"
```

> [!NOTE]
> 注意<http://127.0.0.1:8080>和<http://127.0.0.1:1025>为示例url参数，具体url配置以用户本地部署使用参数配置为准。

### 验证问答成功

若输出包含检索到的文档内容和生成的回答，表示问答流程运行正常：

```text
{'query': '请描述2024年高考作文题目', 'result': '...", 'source_documents': [...]}
```
