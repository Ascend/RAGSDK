# 安装openai-whisper补丁说明

## 安装环境准备

参考：https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/audio/Whisper

| 配套      | 版本要求    |
|---------|---------|
| CANN    | 8.0.RC2 |
| MindIE  | 1.0.RC2 |
| Python  | 3.10.X  |
| PyTorch | 2.1.0   |
| ffmpeg  | 4.2.7   |
| onnx    | 1.16.1  |


安装MindIE前需要先source toolkit的环境变量，然后直接安装，以默认安装路径/usr/local/Ascend为例：
```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash Ascend-mindie_*.run --install
```


## 安装补丁步骤
1.在mxRAG项目下载openai-Whisper源码
```sh
git clone https://github.com/openai/whisper.git
cd whisper
git reset --hard ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab
```

2.在mxRAG项目下载whisper推理编译文件,存放至指定目录

下载地址：https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/audio/Whisper

下载文件：compile.py, mindietorch_infer.patch, trace_model.patch

注意:

(1)建议修改compile.py文件中_MAX_TOKEN 参数值为2240,以支持长语音(超过1分钟)的识别。

(2)默认编译的device值为0，可根据需求更改在mindietorch_infer.patch文件中device的值(第17、108、156行)

3.在whisper源码项目下执行模型导出

```sh
patch -p1 < ../path/to/trace_model.patch
pip3 install .
cd ..
mkdir /tmp/models && mkdir /tmp/models/onnx
mkdir /tmp/models/onnx/encode /tmp/models/onnx/decode /tmp/models/onnx/prefill
wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
whisper zh.wav --model tiny
```
完成上述步骤将在/tmp/models目录下生成encoder.ts, decoder_prefill.ts, decoder_decode.ts, 及对应的onnx文件。


4.模型编译
```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
python3  path/to/compile.py
```
完成上述步骤将在/tmp/models目录下生成language_detection_compiled.ts, encoder_compiled.ts, decoder_prefill_compiled.ts, decoder_decode_compiled.ts四个文件。

5.模型推理
```sh
cd whisper
git reset --hard ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab
patch -p1 < ../path/to/mindietorch_infer.patch
pip3 install .
cd ..
whisper zh.wav --model tiny
```
Python调用
```python
from whisper import load_model
from whisper.transcribe import transcribe
#模型加载
model = load_model('tiny')
result = transcribe(model, audio="zh.wav", verbose=False, beam_size=5, temperature=0)
```

## 注意事项
1.如需修改模型路径，可在打完补丁后手动修改whisper/decoding.py和whisper/model.py文件，后续步骤模型推理同样需要修改对应模型的载入路径。
