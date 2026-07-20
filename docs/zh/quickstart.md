# 快速入门

RAG SDK 提供基于昇腾平台的知识库问答能力，支持文档解析、向量检索、Rerank 等功能。本文帮助用户快速掌握RAG SDK的基本使用流程。

## 前置条件

开始之前，请确认：

- **硬件**：Atlas 300I Duo 推理卡或Atlas 800I A2/A3 推理服务器，并安装对应的驱动、依赖和固件
- **Docker**：已安装 Docker，且当前用户可运行容器

## 步骤 1：拉取镜像

1. **确定待下载镜像版本**
   - 访问昇腾社区[镜像仓](https://www.hiascend.com/developer/ascendhub/detail/b875f781df984480b0385a96fa1b03c9)，查看RAG SDK镜像配套表，获取镜像最新版本以及与之配套的CANN版本
   - 根据当前硬件型号（如 Atlas 800I A2 推理服务器）选择对应版本

    > [!NOTE]
    > 镜像中已安装CANN，无需重复安装<br>
    > 注意区分 CPU 架构（x86_64 / aarch64）

2. **环境预检查**
   - 执行 `npu-smi info` 命令查看当前环境安装的 NPU 驱动版本
   - 通过RAG SDK镜像配套表中获取配套CANN版本，并参见《[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)》页面查看对应的NPU驱动版本，如果和当前环境安装的驱动版本不配套，需更新NPU驱动至对应版本，更新指导详见《[驱动和固件安装指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100568434/36e8d875?idPath=23710424|251366513|254884019|261408772|252764743)》。

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
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:${TAG} ragsdk:${TAG}
   ```

   以 26.0.0 版本、910b 芯片、Ubuntu 22.04、Python 3.11为例：

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11 ragsdk:26.0.0-910b-ubuntu22.04-py3.11
    ```

## 步骤 2：启动容器

> [!NOTE]
>
> - `--device /dev/davinci0` 中的设备编号需按宿主机实际 NPU编号调整（可选，当前样例未使用NPU卡）
> - `-v /path/to/model:/home/data` 挂载宿主机目录到容器（可选，当前样例未使用模型文件）

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

## 步骤 4： 准备测试样例代码

通常容器镜像中已包含/workspace/example/parse_document示例代码，无需额外拷贝；如果该示例代码不存在，请拷贝[样例代码目录example](../../example)到容器内/workspace目录下。

## 步骤 5：执行测试样例

进入示例目录，运行测试样例：

```bash
cd /workspace/example/parse_document
python3 parse_document.py --file_path ./agent.md
```

若输出以下结果，表示样例执行成功：

```text
total docs:xxxx
```
