# RAG SDK

- [RAG SDK](#rag-sdk)
- [Latest Updates](#latest-updates)
- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Version Description](#version-description)
- [Environment Deployment](#environment-deployment)
- [Build Process](#build-process)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Reference](#api-reference)
- [FAQ](#faq)
- [Security Statement](#security-statement)
- [Branch Maintenance Strategy](#branch-maintenance-strategy)
- [Version Maintenance Strategy](#version-maintenance-strategy)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Suggestions and Communication](#suggestions-and-communication)

# Latest Updates

- Dec. 30, 2025: RAG SDK is released as open source.

# Introduction

RAG SDK is an Ascend knowledge-enhancement development kit for large language models. It addresses slow knowledge refresh and weak question answering in vertical domains. It provides capabilities such as domain-specific tuning, generation enhancement, and knowledge management for large language model knowledge bases. This helps users build exclusive, high-performance, and highly accurate question-answering systems for large language models.

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/RAGSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/RAGSDK)

</div>

# Directory Structure

```bash
├─build
├─mx_rag
│  ├─cache
│  ├─chain
│  ├─compress
│  ├─corag
│  ├─document
│  ├─embedding
│  ├─evaluate
│  ├─graphrag
│  ├─knowledge
│  ├─llm
│  ├─reranker
│  ├─retrievers
│  ├─storage
│  ├─summary
│  ├─tools
│  ├─utils
├─ops
├─output
├─script
├─tests
│  ├─data
│  └─python
```

# Version Description

For details about RAG SDK version compatibility and feature changes, see [Release Notes](./docs/en/release_notes.md).

# Environment Deployment

RAG SDK supports two installation methods: deployment inside a container and deployment on a physical machine.

# Build Process

This section describes the workflow for building the `.run` package and running `ut` in RAG SDK container.

1. Download RAG SDK image from [AscendHub](https://www.hiascend.com/developer/ascendhub/detail/ragsdk) and start the container.

2. Clone the repository in the container.

    ```bash
    git clone https://gitcode.com/Ascend/RAGSDK.git
    cd RAGSDK
    ```

3. Go to the `build` subdirectory in `RAGSDK` and run the build script.

    ```bash
    cd build
    bash build.sh
    ```

4. After the build completes, the `.run` package is stored in the `output` subdirectory of `RAGSDK`. Go to the `output` directory to install the `.run` package.

    ```bash
    cd ./output/
    ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install --install-path=<installation path> --platform=<chip_type>
    ```

   > [!NOTE]
   >
   > `<chip_type>` indicates the chip type. You can query it by running the `npu-smi info` command on a server with an Ascend AI processor installed. Remove the last digit from the `Name` value you obtain. That result is the value of `--platform`. If the server is an Atlas 800I A3 SuperNode server, use `A3`.

5. After installing the `.run` package, go to the `tests` subdirectory in `RAGSDK` and run the test cases.

    ```bash
    cd ./tests/
    bash run_py_test.sh
    ```

# Quick Start

RAG SDK provides a quick way to build question-answering systems on the Ascend platform. It provides multimodal document parsing, knowledge base management, and other capabilities, lowers the barrier to developing large language model applications, and supports integration with the open-source ecosystem.

- Quick setup: It provides modular feature APIs and supports on-demand calls. With prebuilt end-to-end workflow templates, users can quickly build a question-answering service with only a small amount of code.
- Multimodal parsing: It supports parsing documents, tables, PDFs, images, and other file types to provide diverse corpora for large language models.
- High-performance inference: It provides Ascend-friendly model optimization and acceleration for higher throughput and shorter response times.

For detailed instructions, see [User Guide](./docs/en/user_guide.md).

# Features

RAG SDK components provide multimodal document parsing, knowledge base management, and other capabilities. They lower the barrier to developing large language model applications and support integration with the open-source ecosystem. For detailed features and usage guidance, see the corresponding sections in the [User Guide](./docs/en/user_guide.md). The released features are as follows:

- ✅Text generation.
- ✅Image retrieval from text.
- ✅Multi-turn conversation.
- ✅Agentic RAG example.
- ✅Chat with RAG SDK.

# API Reference

See [API Reference](./docs/en/api/README.md) for details.

# FAQ

See the [FAQ](./docs/en/faq.md) for related questions.

# Security Statement

- This component is currently deployed in container mode.
- The container has certain permission risks. You are advised to apply additional security hardening.
- For other security hardening measures, see [Security Hardening](./docs/en/security_hardening.md).
- For the public network address list, see [Public Network Addresses](./docs/en/resource/RAG_SDK_public_network_addresses.xlsx).

# Branch Maintenance Strategy

The maintenance phases of version branches are as follows:

| Status | Time | Description |
| -------- | -------- | ------------------------------------------------------------ |
| Planning | 1 to 3 months | Planned features. |
| Development | 3 months | Develop new features, fix issues, and release new versions regularly. |
| Maintenance | 3 to 12 months | Regular branches are maintained for 3 months, and long-term support branches are maintained for 12 months. Major bugs are fixed. No new features are merged. Patch versions are released based on the impact of the bug. |
| End of Life (EOL) | N/A | The branch no longer accepts any changes. |

# Version Maintenance Strategy

| Version | Maintenance Strategy | Current Status | Release Date | Next Status | EOL Date |
| -------- | -------- | -------- | ---------------- | ----------------------------- | ---------- |
| master | Long-term support | Development | 2025-12-30 |  | - |

# Disclaimer

- The code in this repository contains multiple development branches. These branches may include unfinished, experimental, or untested features. Before formal release, do not apply these branches to any production environment or to projects that depend on critical business operations. Be sure to use the official release version to ensure code stability and security.
- Any issues, losses, or data corruption caused by using development branches are not the responsibility of this project or its contributors.
- For the official version, see the release version.

# License

RAG SDK is licensed under Mulan PSL v2. The corresponding license text is available in [LICENSE](./LICENSE.md).

The documents in the `docs` directory of RAG SDK are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

# Suggestions and Communication

Everyone is welcome to contribute to the community. If you have any questions or suggestions, submit them through [GitCode Issues](https://gitcode.com/Ascend/RAGSDK/issues), and we will reply as soon as possible. Thank you for your support.
