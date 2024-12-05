# 1、构建
bash build.sh -v Ascend310P3

# 2、部署
将build目录下mx_rag_opp.cpython-311-aarch64-linux-gnu.so
和patch目录下bert_patch_310.py 导入到PYTHONPATH环境
例如export PYTHONPATH=/path_to_mx_rag/ops/patch:/path_to_mx_rag/ops/build:${PYTHONPATH}

# 3、测试
回到mx_rag主目录下
执行 test/test_op.py将测试单算子性能测试

# 4、运行
客户代码需要先导入bert_patch_310.py，使能加速并且transformer版本目前仅支持4.41.1