# Quick Start

RAG SDK provides knowledge base question answering capabilities on the Ascend platform, supporting document parsing, vector retrieval, reranking, and other functions. This document helps you quickly learn the basic usage process of RAG SDK.

## Prerequisites

Before you start, make sure that the following requirements are met:

- **Hardware**: An Atlas 300I Duo inference card or Atlas 800I A2/A3 inference servers, with the corresponding driver, dependencies, and firmware installed.
- **Docker**: Docker is installed, and the current user can run containers.

## Step 1: Pull the Image

1. **Determine the Image Version to Download**
    - Visit the Ascend community [image repository](https://www.hiascend.com/developer/ascendhub/detail/b875f781df984480b0385a96fa1b03c9) to view the RAG SDK image compatibility table and obtain the latest image version and the corresponding CANN version.
    - Select the corresponding version based on the current hardware model, such as the Atlas 800I A2 inference server.

     > [!NOTE]
     > CANN is already installed in the image. You do not need to install it again.<br>
     > Make sure that the CPU architecture (x86_64/aarch64) is correct.

2. **Perform an Environment Precheck**
    - Run the `npu-smi info` command to check the NPU driver version installed in the current environment.
    - Obtain the corresponding CANN version from the RAG SDK image compatibility table, and refer to the [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers/community) page to check the corresponding NPU driver version. If the driver version installed in the current environment is incompatible, update the NPU driver to the corresponding version. For details, see the [Driver and Firmware Installation Guide](https://support.huawei.com/enterprise/en/doc/EDOC1100568434/36e8d875?idPath=23710424|251366513|254884019|261408772|252764743).

3. **Image Pull Example**

   The image tag is in the following format: `{version}-{chip}-{os}-{python}`. The following table describes the variables.

   | Variable    | Description           | Example Value                   |
   | ----------- | --------------------- | -------------------------------- |
   | `{version}` | RAG SDK version       | `26.0.0`                         |
   | `{chip}`    | Ascend chip series    | `910b`                           |
   | `{os}`      | Base operating system | `ubuntu22.04`/`openeuler24.03` |
   | `{python}`  | Python version        | `py3.11`                         |

   ```bash
   TAG={version}-{chip}-{os}-{python}
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:${TAG}
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:${TAG} ragsdk:${TAG}
   ```

   For example, for version 26.0.0, the 910b chip, Ubuntu 22.04, and Python 3.11:

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/ragsdk:26.0.0-910b-ubuntu22.04-py3.11 ragsdk:26.0.0-910b-ubuntu22.04-py3.11
   ```

## Step 2: Start the Container

> [!NOTE]
>
> - Adjust the device number in `--device /dev/davinci0` according to the actual NPU device number on the host (optional; the current example does not use an NPU card).
> - Mount the host directory to the container using `-v /path/to/model:/home/data` (optional; the current example does not use model files).

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

## Step 3: Enter the Container

```bash
docker exec -it ragsdk_demo bash
```

## Step 4: Prepare the Test Sample Code

The `/workspace/example/parse_document` sample code is usually included in the container image, so no additional copying is required. If the sample code does not exist, copy the [example sample code directory](../../example) to the `/workspace` directory in the container.

## Step 5: Run the Test Sample

Enter the sample directory and run the test sample:

```bash
cd /workspace/example/parse_document
python3 parse_document.py --file_path ./agent.md
```

If the following result is displayed, the sample has run successfully:

```text
total docs:xxxx
```
