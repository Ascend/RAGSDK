# RAGSDK

> [English](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/OVERVIEW.md) | 中文

## 快速参考

- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/RAGSDK/issues)
  - [RAGSDK 代码](https://gitcode.com/Ascend/RAGSDK)
  - [RAGSDK API 参考](https://gitcode.com/Ascend/RAGSDK/blob/master/docs/zh/api/README.md)
  - [RAGSDK 文档](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)
  - [镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/ragsdk)

## RAGSDK

RAGSDK是昇腾面向大语言模型的知识增强开发套件，为解决大模型知识更新缓慢以及垂直领域知识回答弱的问题，面向大模型知识库提供垂域调优、生成增强、知识管理等特性，帮助用户搭建专属的高性能、准确度高的大模型问答系统。

## 支持的 Tags 及 Dockerfile 链接

### Tag 规范

Tag 遵循以下格式:
`<ragsdk版本>-<芯片系列>-<操作系统>-<python版本>`

| 字段                | 示例值   | 说明                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| ragsdk版本                | 26.0.0  | ragsdk版本号                                                     |
| 芯片系列                | 910、A3、atlas 300I Pro   | 目标昇腾芯片系列                         |
| 操作系统                | ubuntu22.04、openeuler24.03    | 目标操作系统                         |
| python版本                | py3.11      | 目标python版本                         |

### RAGSDK 26.0.0

| Tag                | Dockerfile  |
| ------------------- | -------- |
| 26.0.0-310p-ubuntu22.04-py3.11               | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/Dockerfile.310p.ubuntu)  |
| 26.0.0-910b-ubuntu22.04-py3.11                | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/Dockerfile.910b.ubuntu)  |
| 26.0.0-a3-ubuntu22.04-py3.11                | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/Dockerfile.a3.ubuntu)  |
| 26.0.0-310p-openeuler24.03-py3.11               | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/Dockerfile.310p.openeuler)  |
| 26.0.0-910b-openeuler24.03-py3.11                | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/Dockerfile.910b.openeuler)  |
| 26.0.0-a3-openeuler24.03-py3.11                | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/Dockerfile.a3.openeuler)  |

## 快速开始

## 如何本地构建

```bash
# 在宿主机上任意目录下 clone 代码, 并进入 docker 目录, 获取dockerfile替换 {cann version} 为实际版本, 替换 {your_repo} 为实际镜像仓库
git clone https://gitcode.com/Ascend/RAGSDK.git && cd RAGSDK/docker

docker build --network host --build-arg CANN_VERSION={cann version} -t {your_repo}/ragsdk:latest -f Dockerfile.<芯片系列>.<操作系统> .

```

## 运行RAGSDK容器

```bash
 docker run -itd --name=rag_sdk_demo --network=host \
     --device=/dev/davinci_manager \
     --device=/dev/hisi_hdc \
     --device=/dev/devmm_svm \
     --device=/dev/davinci0 \
     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
     -v /usr/local/sbin:/usr/local/sbin:ro \
     -v /path/to/model:/path/to/model:ro \
     {镜像名称}:{镜像tag} bash
```

### 参数说明

- `/path/to/model`：模型存放目录，如需加载模型，请将模型文件放入该目录。
- `{镜像名称}`:`{镜像tag}`：指定要运行的 RAGSDK 镜像和标签。

## 进入容器

```bash
docker exec -it rag_sdk_demo bash
```

## RAGSDK使用说明

RAGSDK 提供丰富的示例代码，帮助开发者快速上手，容器内示例代码在`/workspace/RAGSDK_Samples`路径下。您也可以通过以下链接获取最新的 Demo 示例：

- [RAGSDK Demo 示例代码](https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/RAGSDK/MainRepo/Samples)

## 如何二次开发

```dockerfile
# 以RAGSDK 镜像为基础镜像，叠加用户软件
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11
RUN apt update -y &&
    apt install gcc ...
...
```

## 支持的硬件

| 芯片系列                | 产品示例   | 架构                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| Atlas 910                | Atlas 800T A2、Atlas 900 A2 PoD  | ARM64/ X86_64                                                     |
| Atlas A3                |  Atlas 800T A3    | ARM64/ X86_64                         |
| Atlas 300I Pro                |  Atlas 300I Pro、 Atlas 300V Pro  | ARM64/ X86_64                         |

## 许可证

查看这些镜像中包含的 RAGSDK 和 Mind 系列软件的[许可证信息](https://gitcode.com/Ascend/RAGSDK/blob/master/LICENSE.md)。
与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
