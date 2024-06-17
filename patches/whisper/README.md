# 安装openai-whisper补丁说明

## 安装环境准备
1. 安装Python
2. 安装torch
3. 官网获取torch_npu whl包安装torch_npu

## 安装补丁步骤
在patches/whisper目录下执行一键式补丁脚本
```bash
cd mxRAG/patches/whisper
bash whisper_patch.sh
```

## 注意事项
1. 安装patch后使用whisper，请保证已经安装了适配版本的torch与torch_npu

2. 使用前请先设置CANN环境变量
```sh
    source [cann安装路径]（默认为/usr/local/Ascend/ascend-toolkit）/set_env.sh
```

## 版本依赖
| 软件             | 版本要求 |
|----------------| -------- |
| python         | >= 3.9.0 |
| openai-whisper | == 20231117 |
| torch          | >=2.0.1,<=2.1.0 |
| torch_npu      | >=2.0.1,<=2.1.0 |