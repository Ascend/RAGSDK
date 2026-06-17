# MinerU 加速镜像

> [English](https://gitcode.com/Ascend/RAGSDK/blob/master/example/mineru-accelerate/OVERVIEW.md) | 中文

## 快速参考

- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/RAGSDK/issues)
  - [RAG SDK 代码](https://gitcode.com/Ascend/RAGSDK)
  - [镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/mineru)

## MinerU 加速镜像

基于 vllm-ascend 镜像，集成 MinerU 加速优化，用于文档解析场景的推理加速。目前支持 MinerU2.5模型。

## 支持的硬件

| 芯片系列 | 产品示例 | 架构 |
| -------- | -------- | ---- |
| Atlas 910 | Atlas 800T A2、Atlas 900 A2 PoD | ARM64/ X86_64 |

## 支持的模型

- [OpenDataLab/MinerU2.5-2509-1.2B](https://www.modelscope.cn/models/opendatalab/MinerU2.5-2509-1.2B)
- [OpenDataLab/MinerU2.5-Pro-2605-1.2B](https://www.modelscope.cn/models/opendatalab/MinerU2.5-Pro-2605-1.2B)
- [OpenDataLab/MinerU2.5-Pro-2604-1.2B](https://www.modelscope.cn/models/opendatalab/MinerU2.5-Pro-2604-1.2B)

## 支持的 Tags 及 Dockerfile 链接

### Tag 规范

Tag 遵循以下格式:
`<镜像版本>-<操作系统>-<python版本>-<架构类型>`

| 字段                | 示例值   | 说明                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| mineru镜像版本                | 0.1.22     | 支持的mineru_vl_utils版本为0.1.22     |
| 操作系统                | ubuntu22.04   | 目标操作系统                         |
| python版本                | py3.11      | 目标python版本                         |
| 架构类型                | aarch64      | 目标架构类型                         |

### Dockerfile 链接

| Tag                | Dockerfile  |
| ------------------- | -------- |
| 0.1.22-ubuntu22.04-py3.11-aarch64     | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/example/mineru-accelerate/Dockerfile)  |

## 运行容器

```bash
docker run -u root -itd --name=mineru-accelerate --network=host \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path/to/model:/path/to/model \
    mineru:0.1.22-ubuntu22.04-py3.11-aarch64 bash
```

### 参数说明

- `/path/to/model`：模型存放目录，例如将MinerU2.5-2509-1.2B模型存放于该目录下。
- `--device=/dev/davinci0`：NPU 卡号，请根据实际环境修改（如 davinci0、davinci4 等）。

### 模型配置修改

运行容器后，需在模型目录下的 `config.json` 中添加以下两个字段以启用加速：

```json
"prune_encoder": true,
"process_single_image": true
```

以 MinerU2.5-2509-1.2B 为例，修改 `{model_path}/MinerU2.5-2509-1.2B/config.json`，在文件末尾、`}` 之前添加上述字段即可。

## 进入容器

```bash
docker exec -it mineru-accelerate bash
```

## 快速开始

进入容器后，可参考[MinerU 快速开始使用](https://opendatalab.github.io/MinerU/zh/quick_start/)。

## 如何本地构建

```bash
# 在宿主机上任意目录下 clone 代码, 并进入 mineru-accelerate 目录
git clone https://gitcode.com/Ascend/RAGSDK.git && cd RAGSDK/example/mineru-accelerate

docker build -t mineru:0.1.22-ubuntu22.04-py3.11-aarch64 --network host -f Dockerfile .
```

## 文件说明

| 文件 | 说明 |
| ---- | ---- |
| Dockerfile | 镜像构建文件 |
| patch/vllm_adapt.patch | vllm 适配 MinerU 的补丁，应用于 /vllm-workspace/vllm/ |
| patch/mineru_adapt.patch | mineru_vl_utils 加速优化补丁，应用于 mineru_vl_utils 安装目录 |

## 许可证

查看这些镜像中包含的 RAGSDK 和 Mind 系列软件的[许可证信息](https://gitcode.com/Ascend/RAGSDK/blob/master/LICENSE.md)。
与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
