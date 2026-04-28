# RAGSDK

> [English](./OVERVIEW.md) | 中文

## 快速参考

- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/RAGSDK/issues)
  - [RAGSDK 代码](https://gitcode.com/Ascend/RAGSDK)
  - [RAGSDK API 参考](../docs/zh/api/README.md)
  - [RAGSDK 文档](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)
  - [AscendHub镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/ragsdk)
  - [Ascend社区](https://www.hiascend.com/)

## RAGSDK

RAGSDK是昇腾面向大语言模型的知识增强开发套件，为解决大模型知识更新缓慢以及垂直领域知识回答弱的问题，面向大模型知识库提供垂域调优、生成增强、知识管理等特性，帮助用户搭建专属的高性能、准确度高的大模型问答系统。

## 支持的 Tags 及 Dockerfile 链接

### Tag 规范 

Tag 遵循以下格式: 
`<ragsdk版本>-<芯片系列>-<操作系统>-<python版本>` 

| 字段                | 示例值   | 说明                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| ragsdk版本                | 26.0.0  | ragsdk版本号                                                     |
| 芯片系列                | 310p、910b、A3    | 目标昇腾芯片系列                         |
| 操作系统                | ubuntu22.04、openeuler24.03    | 目标操作系统                         |
| python版本                | py3.11      | 目标python版本                         |

### RAGSDK 26.0.0

| Tag                | Dockerfile  | 
| ------------------- | -------- | 
| 26.0.0-310p-ubuntu22.04-py3.11               | [Dockerfile](./Dockerfile.310p.ubuntu)  | 
| 26.0.0-910b-ubuntu22.04-py3.11                | [Dockerfile](./Dockerfile.910b.ubuntu)  | 
| 26.0.0-a3-ubuntu22.04-py3.11                | [Dockerfile](./Dockerfile.a3.ubuntu)  | 
| 26.0.0-310p-openeuler24.03-py3.11               | [Dockerfile](./Dockerfile.310p.openeuler)  | 
| 26.0.0-910b-openeuler24.03-py3.11                | [Dockerfile](./Dockerfile.910b.openeuler)  | 
| 26.0.0-a3-openeuler24.03-py3.11                | [Dockerfile](./Dockerfile.a3.openeuler)  | 

## 快速开始

## 如何本地构建

```bash
docker build -t 镜像tag --network host --build-arg -f Dockerfile .
```

## 运行RAGSDK容器

```bash
 docker run -u <user> -itd --name=rag_sdk_demo --network=host \
     --device=/dev/davinci_manager \
     --device=/dev/hisi_hdc \
     --device=/dev/devmm_svm \
     --device=/dev/davinci0 \
     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
     -v /usr/local/sbin:/usr/local/sbin:ro \
     -v /path/to/model:/path/to/model:ro \
     <镜像名称>:<镜像tag>
```

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
| 昇腾910B                | Atlas 800T A2、Atlas 900 A2 PoD  | ARM64/ X86_64                                                     |
| 昇腾A3                |  Atlas 800T A3    | ARM64/ X86_64                         |
| 昇腾310P                |  Atlas 300I Pro、 Atlas 300V Pro  | ARM64/ X86_64                         |

## 许可证

查看这些镜像中包含的 RAGSDK 和 Mind 系列软件的[许可证信息](../LICENSE.md)。
与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
