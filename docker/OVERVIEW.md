# RAG SDK

> [中文](https://gitcode.com/Ascend/RAGSDK/blob/master/docker/OVERVIEW.zh.md) | English

## Quick Reference

- Where to get help
  - [Issue Feedback](https://gitcode.com/Ascend/RAGSDK/issues)
  - [RAG SDK Code](https://gitcode.com/Ascend/RAGSDK)
  - [RAG SDK API Reference](https://gitcode.com/Ascend/RAGSDK/blob/master/docs/zh/api/README.md)
  - [RAG SDK Documentation](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)
  - [Image Repository](https://www.hiascend.com/developer/ascendhub/detail/ragsdk)

## RAG SDK

RAG SDK is knowledge enhancement development kit for large language models. It addresses the issues of slow knowledge updates and weak domain-specific knowledge answering in large models. It provides features such as domain-specific tuning, generation enhancement, and knowledge management for large model knowledge bases, helping users build exclusive, high-performance, and accurate large model question-answering systems.

## Supported Tags and Dockerfile Links

### Tag Naming Convention

Tags follow this pattern:

`<ragsdk-version>-<chip-series>-<os>-<python-version>`

| Field            | Example Values  | Description              |
|------------------| -------- |--------------------------|
| RAG SDK Version  | 26.0.0  | RAG SDK version          |
| Chip Series      | 910, A3, atlas 300I Pro    | Target Atlas chip family |
| Operating System | ubuntu22.04, openeuler24.03    | Base operating system    |
| Python Version   | py3.11      | Python version           |

## Version Notes

### RAG SDK Image Matching Table

| IMAGE version | RAG SDK version | CANN version |
|---------------|-----------------|--------------|
| 26.0.0        | 26.0.0          | 9.0.0        |

## Quick Start

## How to Build

```bash
# Clone the repository on the host, enter the docker directory, replace {cann version} with the actual version, and replace {your_repo} with the actual image repository
git clone https://gitcode.com/Ascend/RAGSDK.git && cd RAGSDK/docker

docker build --network host --build-arg CANN_VERSION={cann version} -t {your_repo}/ragsdk:latest -f Dockerfile.<chip-series>.<os> .

```

## Run RAG SDK Container

```bash
 docker run -itd --name=rag_sdk_demo --network=host \
     --device=/dev/davinci_manager \
     --device=/dev/hisi_hdc \
     --device=/dev/devmm_svm \
     --device=/dev/davinci0 \
     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
     -v /usr/local/sbin:/usr/local/sbin:ro \
     -v /path/to/model:/path/to/model:ro \
     {image-name}:{image-tag} bash
```

### Parameter Description

- `/path/to/model`: Model storage directory. Place model files in this directory if you need to load models.
- `{image-name}`:`{image-tag}`: Specify the RAG SDK image and tag to run.

## Enter the Container

```bash
docker exec -it rag_sdk_demo bash
```

## RAG SDK Usage

RAG SDK provides comprehensive sample code to help developers get started quickly. The sample code inside the container is located at `/workspace/RAGSDK_Samples`. You can also access the latest demo examples through the following link:

- [RAG SDK Demo Samples](https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/RAGSDK/MainRepo/Samples)

## Development

```dockerfile
# Use the RAG SDK image as the base image and add user software
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11
RUN apt update -y &&
    apt install gcc ...
...
```

## Supported Hardware

| Chip Series                | Product Examples   | Architecture                                                         |
| ------------------- | -------- | ------------------------------------------------------------ |
| Atlas 910                | Atlas 800T A2, Atlas 900 A2 PoD  | ARM64/ X86_64                                                     |
| Atlas A3                | Atlas 800T A3    | ARM64/ X86_64                         |
| Atlas 300I Pro                | Atlas 300I Pro、 Atlas 300V Pro    | ARM64/ X86_64                         |

## License

View the [license information](https://gitcode.com/Ascend/RAGSDK/blob/master/LICENSE.md) for RAG SDK and Mind series software included in these images.
As with all container images, pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
