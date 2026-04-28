# RAGSDK

> [中文](./OVERVIEW.zh.md) | English

## Quick Reference

- Where to get help

  - [Issue Feedback](https://gitcode.com/Ascend/RAGSDK/issues)
  - [RAGSDK Code](https://gitcode.com/Ascend/RAGSDK)
  - [RAGSDK API Reference](../docs/zh/api/README.md)
  - [RAGSDK Documentation](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)
  - [AscendHub Image Repository](https://www.hiascend.com/developer/ascendhub/detail/ragsdk)
  - [Ascend Community](https://www.hiascend.com/)

## RAGSDK

RAGSDK is Ascend's knowledge enhancement development kit for large language models. It addresses the issues of slow knowledge updates and weak domain-specific knowledge answering in large models. It provides features such as domain-specific tuning, generation enhancement, and knowledge management for large model knowledge bases, helping users build exclusive, high-performance, and accurate large model question-answering systems.

## Supported Tags and Dockerfile Links

### Tag Naming Convention 

Tags follow this pattern:

`<ragsdk-version>-<chip-series>-<os>-<python-version>` 

| Field                | Example Values  | Description                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| RAGSDK Version                | 26.0.0  | RAGSDK version                                                     |
| Chip Series                | 310p, 910b, A3    | Target Ascend chip family                         |
| Operating System                | ubuntu22.04, openeuler24.03    | Base operating system                         |
| Python Version                | py3.11      |  Python version                         |

### RAGSDK 26.0.0

| Tag                | Dockerfile  | 
| ------------------- | -------- | 
| 26.0.0-310p-ubuntu22.04-py3.11               | [Dockerfile](./Dockerfile.310p.ubuntu)  | 
| 26.0.0-910b-ubuntu22.04-py3.11                | [Dockerfile](./Dockerfile.910b.ubuntu)  | 
| 26.0.0-a3-ubuntu22.04-py3.11                | [Dockerfile](./Dockerfile.a3.ubuntu)  | 
| 26.0.0-310p-openeuler24.03-py3.11               | [Dockerfile](./Dockerfile.310p.openeuler)  | 
| 26.0.0-910b-openeuler24.03-py3.11                | [Dockerfile](./Dockerfile.910b.openeuler)  | 
| 26.0.0-a3-openeuler24.03-py3.11                | [Dockerfile](./Dockerfile.a3.openeuler)  | 

## Quick Start

## How to Build

```bash
docker build -t image-tag --network host --build-arg -f Dockerfile .
```

## Run RAGSDK Container

```bash
docker run -u <user> -itd --name=rag_sdk_demo --network=host \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path/to/model:/path/to/model:ro \
    <image-name>:<image-tag>
```

## Development

```dockerfile
# Add required software by developer
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11
RUN apt update -y &&
    apt install gcc ...
...

```

## Supported Hardware

| Chip Series                | Product Examples   | Architecture                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| Ascend 910B                | Atlas 800T A2, Atlas 900 A2 PoD  | ARM64/ X86_64                                                     |
| Ascend A3                | Atlas 800T A3    | ARM64/ X86_64                         |
| Ascend 310P                | Atlas 300I Pro、 Atlas 300V Pro    | ARM64/ X86_64                         |

## License

View the [license information](../LICENSE.md) for RAGSDK and Mind series software included in these images.
As with all container images, pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
