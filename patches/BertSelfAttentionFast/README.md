# 安装补丁说明

## 安装补丁步骤
```bash
bash bertSAFast_patch.sh
```

## 注意事项
1. 安装patch前，请先设置CANN环境变量
```sh
    source [cann安装路径]（默认为/usr/local/Ascend/ascend-toolkit）/set_env.sh
```
2. 安装patch后使用bert_model请确保已经完成了自定义算子的编译与注册。
```sh
    bash /mxRAG/ops/build.sh
```
3. 出现 aclnnBertSelfAttention not find的问题时请查看是否有设置LD_LIBRARY_PATH环境变量如下：
```sh
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
```

## 版本依赖
| 软件           | 版本要求      |
|--------------|-----------|
| pytorch      | == 2.1.0  |
| python       | >= 3.8.0  |
| transformers | >= 4.34.1 |


