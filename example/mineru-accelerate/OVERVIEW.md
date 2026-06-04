# MinerU Acceleration Image

> [中文](https://gitcode.com/Ascend/RAGSDK/blob/master/example/mineru-accelerate/OVERVIEW.zh.md) | English

## Quick Reference

- Where to get help
  - [Issue Feedback](https://gitcode.com/Ascend/RAGSDK/issues)
  - [RAGSDK Code](https://gitcode.com/Ascend/RAGSDK)
  - [Image Repository](https://www.hiascend.com/developer/ascendhub/detail/mineru)

## MinerU Acceleration Image

Based on the vllm-ascend image, MinerU acceleration optimization is integrated for inference acceleration in document parsing scenarios. Currently, the MinerU2.5 model is supported.

## Supported Hardware

| Chip Series | Product Examples | Architecture |
| ----------- | ---------------- | ------------ |
| Atlas 910 | Atlas 800T A2, Atlas 900 A2 PoD | ARM64/ X86_64 |

## Supported Models

- [OpenDataLab/MinerU2.5-2509-1.2B](https://www.modelscope.cn/models/opendatalab/MinerU2.5-2509-1.2B)
- [OpenDataLab/MinerU2.5-Pro-2605-1.2B](https://www.modelscope.cn/models/opendatalab/MinerU2.5-Pro-2605-1.2B)
- [OpenDataLab/MinerU2.5-Pro-2604-1.2B](https://www.modelscope.cn/models/opendatalab/MinerU2.5-Pro-2604-1.2B)

## Supported Tags and Dockerfile Links

### Tag Specification

Tags follow the format:
`<mineru-image-version>`

| Field | Example | Description |
| ----- | ------- | ----------- |
| mineru-image-version | 0.1.22 | The supported mineru_vl_utils version is 0.1.22 |

### Dockerfile Links

| Tag | Dockerfile |
| --- | ---------- |
| 0.1.22 | [Dockerfile](https://gitcode.com/Ascend/RAGSDK/blob/master/example/mineru-accelerate/Dockerfile) |

## Run Container

```bash
docker run -u root -itd --name=mineru-accelerate --network=host \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path/to/model:/path/to/model \
    mineru:0.1.22 bash
```

### Parameter Description

- `/path/to/model`: Model storage directory. For example, place the MinerU2.5-2509-1.2B model in this directory.
- `--device=/dev/davinci0`: NPU card ID. Modify according to your actual environment (e.g., davinci0, davinci4, etc.).

### Model Configuration

After running the container, add the following two fields to the `config.json` in the model directory to enable acceleration:

```json
"prune_encoder": true,
"process_single_image": true
```

For MinerU2.5-2509-1.2B, modify `{model_path}/MinerU2.5-2509-1.2B/config.json` by adding the above fields before the closing `}` at the end of the file.

## Enter the Container

```bash
docker exec -it mineru-accelerate bash
```

## Quick Start

After entering the container, you can refer to [MinerU Quick Start](https://opendatalab.github.io/MinerU/zh/quick_start/).

## How to Build

```bash
# Clone the repository on the host, enter the mineru-accelerate directory
git clone https://gitcode.com/Ascend/RAGSDK.git && cd RAGSDK/example/mineru-accelerate

docker build -t mineru:0.1.22 --network host -f Dockerfile .
```

## File Description

| File | Description |
| ---- | ----------- |
| Dockerfile | Image build file |
| patch/vllm_adapt.patch | vllm adaptation patch for MinerU, applied to /vllm-workspace/vllm/ |
| patch/mineru_adapt.patch | mineru_vl_utils acceleration patch, applied to mineru_vl_utils installation directory |

## License

View the [license information](https://gitcode.com/Ascend/RAGSDK/blob/master/LICENSE.md) for RAGSDK and Mind series software included in these images.
As with all container images, pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
