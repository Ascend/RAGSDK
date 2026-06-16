# 安装部署<a name="ZH-CN_TOPIC_0000002018714781"></a>

## 安装说明<a name="ZH-CN_TOPIC_0000002018595369"></a>

RAG SDK支持物理机部署和容器部署两种方式，本文档介绍在物理机内部署的方式。
建议在容器内部署RAG SDK，在容器中进行部署则不需要执行本文档后续操作，部署指导请参考[RAG SDK镜像](https://www.hiascend.com/developer/ascendhub/detail/b875f781df984480b0385a96fa1b03c9)启动。

**注意事项**

如需安装RAG SDK软件包以外的第三方软件，请注意及时升级最新版本，关注并修补存在的漏洞。

## 安装依赖说明

### 安装NPU驱动固件和CANN

请参考[CANN安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)，使用CANN 9.0.0及对应驱动版本完成NPU驱动固件和CANN的安装。

### 其他依赖

<a id="table285894914124"></a>

<table>
<tr>
<th>软件包简称</th>
<th>安装包全名</th>
<th>配套版本</th>
<th>获取链接</th>
</tr>

<tr>
<td>npu-driver驱动包</td>
<td>Ascend-hdk-<em id="i1935205617"><a name="i1935205617"></a><a name="i1935205617"></a>&lt;chip_type&gt;</em>-npu-driver_<em id="i7935130468"><a name="i7935130468"></a><a name="i7935130468"></a>&lt;version&gt;</em>_linux-<em id="i9935706611"><a name="i9935706611"></a><a name="i9935706611"></a>&lt;arch&gt;</em>.run</td>
<td rowspan="2">25.5.0</td>
<td rowspan="2"><br><a href="https://www.hiascend.com/hardware/firmware-drivers?tag=community">获取链接</a></td>
</tr>
<tr>
<td>npu-firmware固件包</td>
<td>Ascend-hdk-<em id="i173852861"><a name="i173852861"></a><a name="i173852861"></a>&lt;chip_type&gt;</em>-npu-firmware_<em id="i37315214614"><a name="i37315214614"></a><a name="i37315214614"></a>&lt;version&gt;</em>.run</td>
</tr>
<tr>
<td>Index SDK检索软件包</td>
<td>Ascend-mindxsdk-mxindex_<em id="i13839185810816"><a name="i13839185810816"></a><a name="i13839185810816"></a>&lt;version&gt;</em>_linux-<em id="i21954331981"><a name="i21954331981"></a><a name="i21954331981"></a>&lt;arch&gt;</em>.run</td>
<td>26.0.0</td>
<td><a href="https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann">获取链接</a></td>
</tr>
<tr>
<td>Python</td>
<td>-</td>
<td>3.11</td>
<td>请从<a href="https://www.python.org/">Python官网</a>获取依赖软件</td>
</tr>
</table>

> [!NOTE]
>
>- <i><version\></i>表示软件版本号。
>- <i><arch\></i>表示CPU架构。
>- <i><chip\_type\></i>表示芯片类型。可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，将查询到的“Name”最后一位数字删除，即是<i><chip\_type\></i>的取值。
>- 为了让非root用户能够使用驱动，安装npu-driver要添加<b>--install-for-all</b>选项
>- 对于用户集成的开源和第三方软件，漏洞和问题请自行检查并及时进行修复；可以并且不限于通过[CVE（通用漏洞字典）官网](https://cve.mitre.org/cve/search_cve_list.html)确认对应开源软件版本的已知漏洞，并通过版本升级、使用patch补丁包更新等方式修复。

## 安装方式

RAG SDK 提供三种安装方式：离线安装（run 包 / Wheel 包）、源码安装和镜像安装，可根据场景选择合适的方式。

> [!NOTE] 说明
> 三种安装方式均需预先完成[安装依赖说明](#安装依赖说明)中的 NPU 驱动、CANN 及其他第三方依赖安装。所有安装方式在使用前均需加载 CANN 环境变量。

### 离线安装：run 包安装

**安装须知**

- 使用同一普通用户完成 CANN、NPU 驱动固件及 RAG SDK 的安装与运行。
- 软件包的安装、升级、卸载及版本查询相关的日志会保存至 `~/log/mxRag/deployment.log`；完整性校验、提取文件、tar 命令访问相关的日志会保存至 `~/log/makeself/makeself.log`。用户可查看相应文件，完成后续的日志跟踪及审计。

**安装准备**

1. 确保安装环境中已执行 CANN 环境变量配置脚本：

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 默认路径，请按实际安装路径修改
    ```

2. 下载 [RAG SDK 离线安装包](https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann)。

**安装步骤**

1. 以软件包的安装用户登录安装环境。

2. 将 RAG SDK 软件包上传到安装环境的任意路径下并进入软件包所在路径。

3. 增加对软件包的可执行权限：

    ```bash
    chmod u+x Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run
    ```

4. 执行如下命令，校验软件包的一致性和完整性：

    ```bash
    ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --check
    ```

    若显示如下信息，说明软件包已通过校验：

    ```text
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
    ```

5. 创建 RAG SDK 软件包的安装路径（可选）：

    > [!NOTE]
    > 不建议在 `/tmp` 路径下安装：系统重启后 `/tmp` 内容可能被清除，且该目录权限与空间不稳定，不适合持久化部署。

    - 若用户未指定安装路径，软件会默认安装到路径 `/usr/local/Ascend/mxRag`。
    - 若用户想指定安装路径，需要先创建安装路径。以安装路径 `/home/work/RAG_SDK` 为例：

    ```bash
    mkdir -p /home/work/RAG_SDK
    ```

    **表 1**  install 命令可选参数

    <a name="table7138521890"></a>
    <table><thead align="left"><tr><th class="cellrowborder" valign="top" width="35.18%">参数名</th>
    <th class="cellrowborder" valign="top" width="64.82%">参数说明</th>
    </tr>
    </thead>
    <tbody><tr><td class="cellrowborder" valign="top" width="35.18%">--install-path</td>
    <td class="cellrowborder" valign="top" width="64.82%">（可选）自定义软件包安装根目录。如未设置，默认为当前命令执行所在目录。配置的路径必须是"/"或"~"开头，路径取值合法字符为"a-zA-Z0-9_/-"。</td>
    </tr>
    <tr><td class="cellrowborder" valign="top" width="35.18%">--quiet</td>
    <td class="cellrowborder" valign="top" width="64.82%">表示静默操作。</td>
    </tr>
    <tr><td class="cellrowborder" valign="top" width="35.18%">--whitelist</td>
    <td class="cellrowborder" valign="top" width="64.82%">可选参数，表示安装白名单特性，取值可以是operator或者whl，安装多个特性时，可以用逗号分隔。</td>
    </tr>
    </tbody>
    </table>

6. 进入软件包的上传路径，参考以下命令安装 RAG SDK（安装路径的相关约束请参见[表1](#table7138521890)中 `--install-path` 的相关描述）。安装 RAG SDK 时会弹出确认是否接受下载许可协议的说明，若需要在安装时直接跳过该步骤，可在安装命令前增加 `echo y |`，表示同意[华为软件下载许可](https://www.hiascend.com/legal/softlicense)。

    - 若用户指定了安装路径。以安装路径 `/home/work/RAG_SDK` 为例：

        ```bash
        ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install --install-path=/home/work/RAG_SDK
        ```

        或者

        ```bash
        echo y | ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install --install-path=/home/work/RAG_SDK
        ```

    - 若用户未指定安装路径，将安装在当前路径：

        ```bash
        ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install
        ```

        或者

        ```bash
        echo y | ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install
        ```

    > [!NOTE]
    > --install安装命令同时支持输入可选参数，如[表1](#table7138521890)所示。输入不在列表中的参数可能正常安装或者报错。

7. 安装过程中提示 "Do you accept the LICENSE to install RAG SDK? [Y/N]" 时，输入 Y 或 y，表示同意下载协议，继续进行安装；输入其他字符时停止安装，退出程序。

8. 安装完成后，若未出现错误信息，表示软件成功安装于指定或默认路径下：

    ```text
    Install package successfully
    ```

9. 安装 RAG SDK 的依赖包：

   ```bash
   pip3 install -r <安装路径>/mxRag/requirements.txt
   ```

10. 验证 RAG SDK 依赖包安装是否成功：

    ```bash
    pip3 list | grep mxRag
    ```

    若能检索到 `mxRag` 相关包，说明安装成功。

> [!NOTE]
> 安装 RAG SDK 时可能出现报错信息：
> `ERROR: Cannot uninstall 'xxx'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.`
> 说明 `xxx` 模块是操作系统自带的组件，无法直接升级，可以尝试重新下发指令安装：
>
> ```bash
> pip3 install -r <安装路径>/mxRag/requirements.txt --ignore-installed
> ```

1. 设置 RAG SDK 运行环境变量：

   用 vim 打开文件 `~/.bashrc`，在文件最后添加如下内容。

    > [!NOTE]
    > 以下为示例配置，请根据实际安装路径替换所有路径占位符。如安装时使用了自定义路径（如 `/home/work/RAG_SDK`），请将路径中涉及 `/usr/local/Ascend` 的部分替换为实际安装路径。各路径变量说明如下：
    > - `<Ascend安装路径>`：CANN 和驱动的安装根目录，默认为 `/usr/local/Ascend`，请按实际路径修改。
    > - `<index SDK安装路径>`：Index SDK 的安装路径，默认为 `<Ascend安装路径>/mxIndex`，请按实际路径修改。
    > - `<faiss安装路径>`：faiss 的安装路径，请按实际路径修改。
    > - `<模型路径>`：模型存放路径，请按实际路径修改。

   ```bash
   export MX_INDEX_FINALIZE=0
   export PY_VERSION=python3.11
   export LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message!r}</level>'
   export MX_INDEX_MODELPATH=<模型路径>
   # 设置 index SDK 安装路径，请根据实际安装路径修改
   export MX_INDEX_INSTALL_PATH=<index SDK安装路径>
   export MX_INDEX_MULTITHREAD=1
   export ASCEND_HOME=<Ascend安装路径>
   export ASCEND_VERSION=<Ascend安装路径>/ascend-toolkit/latest
   export LD_LIBRARY_PATH=<index SDK安装路径>/lib:<faiss安装路径>/lib:<Ascend安装路径>/OpenBLAS/lib:$LD_LIBRARY_PATH
   export LD_PRELOAD=\$(find \$(python3 -c "import sklearn; print(sklearn.__path__[0])")/../scikit_learn.libs -name "libgomp-*" | head -1):\$LD_PRELOAD
   export PATH=/usr/local/bin:$PATH
   source <Ascend安装路径>/ascend-toolkit/set_env.sh
   source <Ascend安装路径>/nnal/atb/set_env.sh
   source <安装路径>/mxRag/script/set_env.sh
   ```

   保存退出后运行如下命令让环境生效：

    ```bash
    source ~/.bashrc
    ```

**安装验证**<a name="安装验证"></a>

1. 执行 `npu-smi info` 命令检查驱动是否挂载正常。如当 Health 参数的值为 OK 时，即表示当前芯片的健康状态为正常。

2. 验证 RAG SDK 是否安装成功：

    ```bash
    python3 -c "import mxRag; print('RAG SDK import: OK')"
    ```

### 离线安装：Wheel 包安装

**安装须知**

- Wheel 包不包含 CANN（`libascendcl.so` 等）及 Python 第三方依赖，使用前须按[安装依赖说明](#安装依赖说明)完成安装。
- 每次启动 Python 前，须加载 CANN 环境变量（路径以实际安装为准）：

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 或
    source /usr/local/Ascend/cann/set_env.sh
    ```

**安装准备**

1. 获取与 RAG SDK 版本配套的 `ragsdk-*.whl`：

    - 从 Release 页面下载：在 [RAG SDK Releases](https://www.hiascend.com/developer/software/mindsdk/download) 获取与 `run` 包同版本的 Wheel 文件。
    - 从 `run` 包中提取：

        ```bash
        ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --noexec --extract=/tmp/ragsdk_extract
        find /tmp/ragsdk_extract -name 'ragsdk-*.whl'
        ```

2. 确认 CANN 环境变量已加载（见上方说明）。

3. 确认[安装依赖说明](#安装依赖说明)所列 Python 依赖已安装。

**安装步骤**

1. 安装 Wheel 包（将 `ragsdk-1.0.0-py3-none-any.whl` 替换为实际文件名）：

    ```bash
    pip3 install /path/to/ragsdk-1.0.0-py3-none-any.whl
    ```

    若环境中已存在旧版本，可强制重装：

    ```bash
    pip3 install --force-reinstall --no-deps /path/to/ragsdk-1.0.0-py3-none-any.whl
    ```

    > [!NOTE]
    > 建议使用 `--no-deps`，避免 pip 自动升级或降级成固定的依赖版本。

2. 安装 RAG SDK 的依赖包：

    ```bash
    pip3 install -r <安装路径>/mxRag/requirements.txt
    ```

**安装验证**

1. 验证 RAG SDK Python 包是否可正常导入：

    ```bash
    python3 -c "import mxRag; print('RAG SDK import: OK')"
    ```

2. 验证环境变量是否设置正确：

    ```bash
    python3 -c "import os; print('MX_INDEX_INSTALL_PATH:', os.environ.get('MX_INDEX_INSTALL_PATH'))"
    ```

### 源码安装

如需从源码构建 RAG SDK，请参考 [CONTRIBUTING.md](../../CONTRIBUTING.md) 开发与测试章节进行源码构建。

### 镜像安装

提供dockerfile来构建RAG SDK镜像，详细使用指导请参阅 [docker/OVERVIEW.zh.md](../../docker/OVERVIEW.zh.md)。

# 升级<a name="ZH-CN_TOPIC_0000001983329754"></a>

## run 包升级

**注意事项<a name="section1894903161912"></a>**

升级操作涉及对安装目录的卸载再安装，如目录下存在其他文件，也会被一并删除。请在执行升级操作前，确保所有数据都已妥善处理。

**操作步骤<a name="section37391535123710"></a>**

用户如需将当前版本的RAG SDK升级至最新版本，可将最新的RAG SDK软件包上传至安装环境后，在软件包所在目录下使用命令进行版本升级，具体命令参见如下。

1. 使用<b>--upgrade</b>命令升级。

    ```bash
    bash Ascend-mindxsdk-mxrag_<version>_linux-<arch>.run --upgrade --install-path=<安装路径> --platform=<npu_type>
    ```

    <i><version\></i>为版本号，<i><arch\></i>为操作系统架构，<i><npu\_type\></i>为芯片类型。

    **表 1**  参数名及说明

    <a name="table17754104316374"></a>
    <table><thead align="left"><tr id="row575494393716"><th class="cellrowborder" valign="top" width="35.18%" id="mcps1.2.3.1.1"><p id="p1875474393717"><a name="p1875474393717"></a><a name="p1875474393717"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="64.82%" id="mcps1.2.3.1.2"><p id="p375584303712"><a name="p375584303712"></a><a name="p375584303712"></a>参数说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1975564333719"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p37557431375"><a name="p37557431375"></a><a name="p37557431375"></a>--upgrade</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p77551431377"><a name="p77551431377"></a><a name="p77551431377"></a>软件包升级操作命令，将<span id="ph1656424955418"><a name="ph1656424955418"></a><a name="ph1656424955418"></a>RAG SDK</span>升级到安装包所包含的版本。</p>
    </td>
    </tr>
    <tr id="row167552043133716"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p3755943183713"><a name="p3755943183713"></a><a name="p3755943183713"></a>--platform</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p0755943163719"><a name="p0755943163719"></a><a name="p0755943163719"></a>对应<span id="ph075594383717"><a name="ph075594383717"></a><a name="ph075594383717"></a>昇腾AI处理器</span>类型。</p>
    <p id="p1399417513288"><a name="p1399417513288"></a><a name="p1399417513288"></a>请在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是--platform的取值。</p>
    <p id="p145658372256"><a name="p145658372256"></a><a name="p145658372256"></a>若是<span id="ph12325145818223"><a name="ph12325145818223"></a><a name="ph12325145818223"></a>Atlas 800I A3 超节点服务器</span>则取值为A3。</p>
    <p id="p_new002"><a name="p_new002"></a><a name="p_new002"></a><b>注意：--platform 该参数当前版本已废弃使用，无需配置。</b></p>
    </td>
    </tr>
    <tr id="row15756174312379"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p12756114363719"><a name="p12756114363719"></a><a name="p12756114363719"></a>--install-path</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p19756943133712"><a name="p19756943133712"></a><a name="p19756943133712"></a>（可选）自定义软件包安装根目录。如未设置，默认为当前命令执行所在目录。配置的路径必须是"/"或"~"开头，路径取值合法字符为"a-zA-Z0-9_/-"。</p>
    <p id="p14756174303720"><a name="p14756174303720"></a><a name="p14756174303720"></a>如使用自定义目录安装，建议在升级操作时使用该参数。</p>
    <p id="p1637441211"><a name="p1637441211"></a><a name="p1637441211"></a>请确保配置的路径下已安装<span id="ph431844192119"><a name="ph431844192119"></a><a name="ph431844192119"></a>RAG SDK</span>。</p>
    </td>
    </tr>
    <tr id="row97569438379"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p1975694363720"><a name="p1975694363720"></a><a name="p1975694363720"></a>--quiet</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p1075634353718"><a name="p1075634353718"></a><a name="p1075634353718"></a><span id="ph264718181219"><a name="ph264718181219"></a><a name="ph264718181219"></a>表示静默操作</span><span id="ph1127692317123"><a name="ph1127692317123"></a><a name="ph1127692317123"></a>。</span></p>
    </td>
    </tr>
    <tr id="row38771538191618"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p3380154113280"><a name="p3380154113280"></a><a name="p3380154113280"></a>--whitelist</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p4380141182816"><a name="p4380141182816"></a><a name="p4380141182816"></a>可选参数，表示安装白名单特性，取值可以是operator或者whl，安装多个特性时，可以用逗号分隔。</p>
    </td>
    </tr>
    </tbody>
    </table>

    升级过程中提示Do you want to upgrade to a newer version provided by this package and the old version will be removed? \[Y/N\]时，输入Y或y表示同意升级，此时旧版本的RAG SDK将被卸载；输入其他内容表示退出升级。

2. 出现以下提示说明升级成功。

    ```text
    Upgrade RAG SDK successfully
    ```

### Wheel 包升级

1. 获取新版 Wheel 包（[RAG SDK Releases](https://gitcode.com/Ascend/RAGSDK/releases)）。

2. 加载 CANN 环境变量。

3. 执行升级命令（将 `mxrag-x.x.x-py3-none-any.whl` 替换为实际文件名）：

    ```bash
    pip3 install --upgrade /path/to/mxrag-x.x.x-py3-none-any.whl
    ```

    若需强制覆盖且不更新依赖：

    ```bash
    pip3 install --force-reinstall --no-deps /path/to/mxrag-x.x.x-py3-none-any.whl
    ```

4. 执行[安装验证](#安装验证)中的命令确认升级成功。

> [!NOTE] 说明
>
> - 建议使用 `--no-deps` 避免 pip 自动升级或降级已固定的依赖版本。
> - 若需回退，可使用 `pip3 install --force-reinstall /path/to/mxrag-旧版本.whl` 重新安装旧版本。

# 卸载<a name="ZH-CN_TOPIC_0000002018595337"></a>

## run 包卸载

用户如需移除RAG SDK软件包部署，可参考以下命令进行卸载:

```bash
bash 安装目录/mxRag/script/uninstall.sh
```

若显示如下信息，则表示软件成功卸载。

```bash
Uninstall RAG SDK package successfully.
```

### Wheel 包卸载

```bash
pip3 uninstall -y mxrag
pip3 show mxrag 2>/dev/null && echo "mxrag 仍存在" || echo "mxrag 已卸载"
```
