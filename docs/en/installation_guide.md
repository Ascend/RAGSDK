# Installation and Deployment

## Installation Instructions

RAG SDK supports two deployment methods: deployment on a physical machine and deployment in a container. This document describes how to deploy RAG SDK on a physical machine.
Deployment of RAG SDK in a container is recommended. If you deploy RAG SDK in a container, you do not need to perform the subsequent operations in this document. For deployment instructions, see [Starting the RAG SDK Image](https://www.hiascend.com/developer/ascendhub/detail/b875f781df984480b0385a96fa1b03c9).

**Notes**

If you need to install third-party software other than the RAG SDK package, upgrade it to the latest version in a timely manner, and monitor and remediate any vulnerabilities.

## Installing Dependencies

### Installing the NPU Driver, Firmware, and CANN

For details about installing the Ascend NPU driver and CANN software, including the Toolkit and `ops` packages, and configuring environment variables, see the [CANN Installation Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum).

### Other Dependencies

<a id="table285894914124"></a>

<table>
<tr>
<th>Short Package Name</th>
<th>Full Package Name</th>
<th>Matching Version</th>
<th>Download Link</th>
</tr>
<tr>
<td>Index SDK Retrieval Package</td>
<td>Ascend-mindxsdk-mxindex_<em id="i13839185810816"><a name="i13839185810816"></a><a name="i13839185810816"></a>&lt;version&gt;</em>_linux-<em id="i21954331981"><a name="i21954331981"></a><a name="i21954331981"></a>&lt;arch&gt;</em>.run</td>
<td>26.1.0</td>
<td><a href="https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann">Download Link</a></td>
</tr>
<tr>
<td>Python</td>
<td>-</td>
<td>3.11</td>
<td>Obtain the dependency software from the <a href="https://www.python.org/">Python official website</a>.</td>
</tr>
</table>

> [!NOTE]
>
> - `<version>` indicates the software version number.
> - `<arch>` indicates the CPU architecture.
> - To allow non-root users to use the driver, add the <b>--install-for-all</b> option when installing `npu-driver`.
> - For open-source and third-party software integrated by users, check for vulnerabilities and issues and remediate them in a timely manner. You can, but are not limited to, using the [CVE (Common Vulnerabilities and Exposures) website](https://cve.mitre.org/cve/search_cve_list.html) to check known vulnerabilities in the corresponding versions of open-source software, and remediate them by upgrading versions or applying patches.

## Installation Methods

RAG SDK provides three installation methods: offline installation (run package / Wheel package), source installation, and image installation. Select an appropriate method based on your scenario.

> [!NOTE]
> All three installation methods require the NPU driver, CANN, and other third-party dependencies described in [Installing Dependencies](#installing-dependencies) to be installed in advance. The CANN environment variables must be loaded before using RAG SDK regardless of the installation method.

### Offline Installation: run Package Installation

**Installation Notes**

- Use the same non-root user to install and run CANN, the NPU driver and firmware, and RAG SDK.
- Logs related to package installation, upgrade, uninstallation, and version queries are saved to `~/log/mxRag/deployment.log`. Logs related to integrity verification, file extraction, and access to the `tar` command are saved to `~/log/makeself/makeself.log`. You can view the corresponding files for subsequent log tracking and auditing.

**Installation Preparation**

1. Make sure that the CANN environment variable configuration script has been executed in the installation environment:

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Default path. Modify it according to the actual installation path.
   ```

2. Download the [RAG SDK offline installation package](https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann).

**Installation Procedure**

1. Log in to the installation environment as the user who installs the package.

2. Upload the RAG SDK package to any path in the installation environment and go to the directory containing the package.

3. Grant execute permissions to the package:

   ```bash
   chmod u+x Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run
   ```

4. Run the following command to verify the consistency and integrity of the package:

   ```bash
   ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --check
   ```

   If the following information is displayed, the package has passed verification:

   ```text
   Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
   ```

5. Create the installation path for the RAG SDK package (optional):

   > [!NOTE]
   > Installing under `/tmp` is not recommended. The contents of `/tmp` may be cleared after a system restart, and the permissions and available space of this directory are unstable, making it unsuitable for persistent deployment.

   - If the user does not specify an installation path, the software is installed to `/usr/local/Ascend/mxRag` by default.
   - If the user wants to specify an installation path, create the installation path first. For example, to use `/home/work/RAG_SDK` as the installation path:

   ```bash
   mkdir -p /home/work/RAG_SDK
   ```

   **Table 1** Optional Parameters of the `install` Command

    <a name="table7138521890"></a>
    <table><thead align="left"><tr><th class="cellrowborder" valign="top" width="35.18%">Parameter Name</th>
    <th class="cellrowborder" valign="top" width="64.82%">Parameter Description</th>
    </tr>
    </thead>
    <tbody><tr><td class="cellrowborder" valign="top" width="35.18%">--install-path</td>
    <td class="cellrowborder" valign="top" width="64.82%">(Optional) Custom root directory for package installation. If not set, the directory where the current command is executed is used by default. The configured path must start with "/" or "~". Valid characters are "a-zA-Z0-9_/-".</td>
    </tr>
    <tr><td class="cellrowborder" valign="top" width="35.18%">--quiet</td>
    <td class="cellrowborder" valign="top" width="64.82%">Performs a silent operation.</td>
    </tr>
    <tr><td class="cellrowborder" valign="top" width="35.18%">--whitelist</td>
    <td class="cellrowborder" valign="top" width="64.82%">Optional parameter that installs whitelist features. The value can be operator or whl. When installing multiple features, separate them with commas.</td>
    </tr>
    </tbody>
    </table>

6. Go to the directory where the package was uploaded and refer to the following commands to install RAG SDK. For constraints on the installation path, see the description of `--install-path` in [Table 1](#table7138521890). During RAG SDK installation, a prompt asking whether you accept the download license agreement is displayed. To skip this step during installation, add `echo y |` before the installation command to indicate acceptance of the [Huawei Software Download License](https://www.hiascend.com/legal/softlicense).

   - If the user specifies an installation path. For example, to use `/home/work/RAG_SDK` as the installation path:

     ```bash
     ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install --install-path=/home/work/RAG_SDK
     ```

     or

     ```bash
     echo y | ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install --install-path=/home/work/RAG_SDK
     ```

   - If the user does not specify an installation path, the package is installed in the current directory:

     ```bash
     ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install
     ```

     or

     ```bash
     echo y | ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --install
     ```

   > [!NOTE]
   > The `--install` command also supports the optional parameters shown in [Table 1](#table7138521890). Parameters not listed in the table may either be accepted and allow installation to proceed normally or cause an error.

7. When the prompt "Do you accept the LICENSE to install RAG SDK? [Y/N]" is displayed during installation, enter `Y` or `y` to accept the download agreement and continue the installation. Entering any other character stops the installation and exits the program.

8. After the installation is complete, if no error message is displayed, the software has been successfully installed in the specified or default path:

   ```text
   Install package successfully
   ```

9. Install the RAG SDK dependency packages:

   ```bash
   pip3 install -r <installation_path>/mxRag/requirements.txt
   ```

10. Verify whether the RAG SDK dependency packages were installed successfully:

    ```bash
    pip3 list | grep mxRag
    ```

    If packages related to `mxRag` are found, the installation is successful.

    > [!NOTE]
    > An error may occur when installing RAG SDK:
    > `ERROR: Cannot uninstall 'xxx'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.`
    > This indicates that the `xxx` module is a component provided with the operating system and cannot be upgraded directly. You can try running the installation command again:
    >
    > ```bash
    > pip3 install -r <installation_path>/mxRag/requirements.txt --ignore-installed
    > ```

11. Set the RAG SDK runtime environment variables:

    Open the `~/.bashrc` file with `vim` and add the following content to the end of the file.

    > [!NOTE]
    > The following is an example configuration. Replace all path placeholders according to the actual installation paths. If a custom installation path was used, such as `/home/work/RAG_SDK`, replace the `/usr/local/Ascend` portions in the paths with the actual installation path. The path variables are described as follows:
    > - `<Ascend installation path>`: Root directory where CANN and the driver are installed. The default is `/usr/local/Ascend`. Modify it according to the actual path.
    > - `<Index SDK installation path>`: Installation path of Index SDK. The default is `<Ascend installation path>/mxIndex`. Modify it according to the actual path.
    > - `<faiss installation path>`: Installation path of faiss. Modify it according to the actual path.
    > - `<model path>`: Path where models are stored. Modify it according to the actual path.

   ```bash
   export MX_INDEX_FINALIZE=0
   export PY_VERSION=python3.11
   export LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message!r}</level>'
   export MX_INDEX_MODELPATH=<model path>
   # Set the Index SDK installation path. Modify it according to the actual installation path.
   export MX_INDEX_INSTALL_PATH=<Index SDK installation path>
   export MX_INDEX_MULTITHREAD=1
   export ASCEND_HOME=<Ascend installation path>
   export ASCEND_VERSION=<Ascend installation path>/ascend-toolkit/latest
   export LD_LIBRARY_PATH=<Index SDK installation path>/lib:<faiss installation path>/lib:<Ascend installation path>/OpenBLAS/lib:$LD_LIBRARY_PATH
   export LD_PRELOAD=\$(find \$(python3 -c "import sklearn; print(sklearn.__path__[0])")/../scikit_learn.libs -name "libgomp-*" | head -1):\$LD_PRELOAD
   export PATH=/usr/local/bin:$PATH
   source <Ascend installation path>/ascend-toolkit/set_env.sh
   source <Ascend installation path>/nnal/atb/set_env.sh
   source <installation path>/mxRag/script/set_env.sh
   ```

   After saving and exiting, run the following command to activate the environment:

    ```bash
    source ~/.bashrc
    ```

<a id="installationverification"></a>**Installation Verification**

1. Run the `npu-smi info` command to check whether the driver is mounted correctly. If the value of the `Health` parameter is `OK`, the current chip is healthy.

2. Verify whether RAG SDK is installed successfully:

    ```bash
    python3 -c "import mxRag; print('RAG SDK import: OK')"
    ```

### Offline Installation: Wheel Package Installation

**Installation Notes**

- The Wheel package does not contain CANN (`libascendcl.so`, etc.) or Python third-party dependencies. Before use, complete the installation according to [Installing Dependencies](#installing-dependencies).
- CANN environment variables must be loaded each time Python is started. The path depends on the actual installation:

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # or
    source /usr/local/Ascend/cann/set_env.sh
    ```

**Installation Preparation**

1. Obtain the `ragsdk-*.whl` corresponding to the RAG SDK version:

    - Download from the Release page: Obtain the Wheel file with the same version as the `run` package from [RAG SDK Releases](https://www.hiascend.com/developer/software/mindsdk/download).
    - Extract it from the `run` package:

        ```bash
        ./Ascend-mindxsdk-mxrag_{version}_linux-{arch}.run --noexec --extract=/tmp/ragsdk_extract
        find /tmp/ragsdk_extract -name 'ragsdk-*.whl'
        ```

2. Confirm that the CANN environment variables have been loaded (see the preceding description).

3. Confirm that the Python dependencies listed in [Installing Dependencies](#installing-dependencies) are installed.

**Installation Procedure**

1. Install the Wheel package (replace `ragsdk-1.0.0-py3-none-any.whl` with the actual filename):

    ```bash
    pip3 install /path/to/ragsdk-1.0.0-py3-none-any.whl
    ```

    If an older version already exists in the environment, force a reinstallation:

    ```bash
    pip3 install --force-reinstall --no-deps /path/to/ragsdk-1.0.0-py3-none-any.whl
    ```

    > [!NOTE]
    > You are advised to use `--no-deps` to prevent pip from automatically upgrading or downgrading dependencies to fixed versions.

2. Install the RAG SDK dependency packages:

    ```bash
    pip3 install -r <installation_path>/mxRag/requirements.txt
    ```

**Installation Verification**

1. Verify that the RAG SDK Python package can be imported successfully:

    ```bash
    python3 -c "import mxRag; print('RAG SDK import: OK')"
    ```

2. Verify that the environment variables are set correctly:

    ```bash
    python3 -c "import os; print('MX_INDEX_INSTALL_PATH:', os.environ.get('MX_INDEX_INSTALL_PATH'))"
    ```

### Source Installation

To build RAG SDK from source code, see the development and testing section in [CONTRIBUTING.md](../../CONTRIBUTING.md) for instructions on building from source.

### Image Installation

A Dockerfile is provided for building the RAG SDK image. For detailed instructions, see [docker/OVERVIEW.en.md](../../docker/OVERVIEW.md).

## Upgrading

### run Package Upgrade

**Notes<a name="section1894903161912"></a>**

The upgrade operation involves uninstalling and then reinstalling the package in the installation directory. If other files exist in the directory, they will also be deleted. Before performing the upgrade, ensure that all data has been properly handled.

**Procedure<a name="section37391535123710"></a>**

If you need to upgrade the current version of RAG SDK to the latest version, upload the latest RAG SDK package to the installation environment and run the upgrade command in the directory containing the package. The command is as follows.

1. Use the <b>--upgrade</b> command to upgrade.

    ```bash
    bash Ascend-mindxsdk-mxrag_<version>_linux-<arch>.run --upgrade --install-path=<installation_path> --platform=<npu_type>
    ```

    `<version>` is the version number, `<arch>` is the operating system architecture, and `<npu_type>` is the chip type.

    **Table 1** Parameter Names and Descriptions

    <a name="table17754104316374"></a>
    <table><thead align="left"><tr id="row575494393716"><th class="cellrowborder" valign="top" width="35.18%" id="mcps1.2.3.1.1"><p id="p1875474393717"><a name="p1875474393717"></a><a name="p1875474393717"></a>Parameter Name</p>
    </th>
    <th class="cellrowborder" valign="top" width="64.82%" id="mcps1.2.3.1.2"><p id="p375584303712"><a name="p375584303712"></a><a name="p375584303712"></a>Parameter Description</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1975564333719"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p37557431375"><a name="p37557431375"></a><a name="p37557431375"></a>--upgrade</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p77551431377"><a name="p77551431377"></a><a name="p77551431377"></a>Package upgrade command. It upgrades <span id="ph1656424955418"><a name="ph1656424955418"></a><a name="ph1656424955418"></a>RAG SDK</span> to the version included in the installation package.</p>
    </td>
    </tr>
    <tr id="row167552043133716"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p3755943183713"><a name="p3755943183713"></a><a name="p3755943183713"></a>--platform</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p0755943163719"><a name="p0755943163719"></a><a name="p0755943163719"></a>Corresponds to the <span id="ph075594383717"><a name="ph075594383717"></a><a name="ph075594383717"></a>Ascend AI processor</span> type.</p>
    <p id="p1399417513288"><a name="p1399417513288"></a><a name="p1399417513288"></a>Run the npu-smi info command on the server where the Ascend AI processor is installed to query this value. Remove the last digit from the displayed "Name" value to obtain the value of --platform.</p>
    <p id="p145658372256"><a name="p145658372256"></a><a name="p145658372256"></a>If the server is an <span id="ph12325145818223"><a name="ph12325145818223"></a><a name="ph12325145818223"></a>Atlas 800I A3 SuperPoD server</span>, the value is A3.</p>
    <p id="p_new002"><a name="p_new002"></a><a name="p_new002"></a><b>Note: The --platform parameter is deprecated in the current version and does not need to be configured.</b></p>
    </td>
    </tr>
    <tr id="row15756174312379"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p12756114363719"><a name="p12756114363719"></a><a name="p12756114363719"></a>--install-path</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p19756943133712"><a name="p19756943133712"></a><a name="p19756943133712"></a>(Optional) Custom root directory for package installation. If not set, the directory where the current command is executed is used by default. The configured path must start with "/" or "~". Valid characters are "a-zA-Z0-9_/-".</p>
    <p id="p14756174303720"><a name="p14756174303720"></a><a name="p14756174303720"></a>If you use a custom installation directory, you are advised to specify this parameter during the upgrade.</p>
    <p id="p1637441211"><a name="p1637441211"></a><a name="p1637441211"></a>Ensure that <span id="ph431844192119"><a name="ph431844192119"></a><a name="ph431844192119"></a>RAG SDK</span> is already installed in the specified path.</p>
    </td>
    </tr>
    <tr id="row97569438379"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p1975694363720"><a name="p1975694363720"></a><a name="p1975694363720"></a>--quiet</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p1075634353718"><a name="p1075634353718"></a><a name="p1075634353718"></a>Performs a silent operation.</p>
    </td>
    </tr>
    <tr id="row38771538191618"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p3380154113280"><a name="p3380154113280"></a><a name="p3380154113280"></a>--whitelist</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p4380141182816"><a name="p4380141182816"></a><a name="p4380141182816"></a>Optional parameter that installs whitelist features. The value can be operator or whl. When installing multiple features, separate them with commas.</p>
    </td>
    </tr>
    </tbody>
    </table>

    During the upgrade, when the prompt `Do you want to upgrade to a newer version provided by this package and the old version will be removed? [Y/N]` is displayed, enter Y or y to agree to the upgrade. The old version of RAG SDK is then uninstalled. Entering any other content exits the upgrade.

2. If the following prompt is displayed, the upgrade is successful.

    ```text
    Upgrade RAG SDK successfully
    ```

### Wheel Package Upgrade

1. Obtain the new Wheel package ([RAG SDK Releases](https://gitcode.com/Ascend/RAGSDK/releases)).

2. Load the CANN environment variables.

3. Run the upgrade command (replace `mxrag-x.x.x-py3-none-any.whl` with the actual filename):

    ```bash
    pip3 install --upgrade /path/to/mxrag-x.x.x-py3-none-any.whl
    ```

    To force an overwrite without updating dependencies:

    ```bash
    pip3 install --force-reinstall --no-deps /path/to/mxrag-x.x.x-py3-none-any.whl
    ```

4. Run the commands in [Installation Verification](#installationverification) to verify that the upgrade was successful.

> [!NOTE]
>
> - You are advised to use `--no-deps` to prevent pip from automatically upgrading or downgrading fixed dependency versions.
> - To roll back to an earlier version, reinstall the earlier version using `pip3 install --force-reinstall /path/to/mxrag-<old_version>.whl`.

## Uninstallation

### run Package Uninstallation

To remove the RAG SDK package deployment, run the following command to uninstall it:

```bash
bash installation_path/mxRag/script/uninstall.sh
```

If the following information is displayed, the software has been successfully uninstalled.

```bash
Uninstall RAG SDK package successfully.
```

### Wheel Package Uninstallation

```bash
pip3 uninstall -y mxrag
pip3 show mxrag 2>/dev/null && echo "mxrag still exists" || echo "mxrag has been uninstalled"
```
