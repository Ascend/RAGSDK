# 安装补丁说明

## 安装补丁步骤
```bash
bash tei_patch.sh [text-embeddings-inference包的所在路径]
```

## 注意事项
1. 安装patch前，请先设置CANN环境变量
```sh
    source [cann安装路径]（默认为/usr/local/Ascend/ascend-toolkit）/set_env.sh
```
2. TEI开源代码保持text-embeddings-inference名称不变

3. 安装patch后，在text-embeddings-inference目录下会生成NPU适配相关使用说明NPU_ADAPT.md，根据该说明进行后续相关操作

## 版本依赖
| 软件  | 版本要求            |
| ----- |-----------------|
| python | >= 3.9.0        |
| transformers | == 4.40.2       |
|TEI| 1.2.3           |
|torch_npu| 2.1.0.post3     |
|safetensors| <0.4.0, >=0.3.2 |