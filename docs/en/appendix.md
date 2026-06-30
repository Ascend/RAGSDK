# Appendix

## Public Network Addresses Included in the Software

The open-source software compiled by RAG SDK contains public URLs and email addresses. The SDK does not access these addresses.

For the public network addresses and email addresses, see [RAG SDK 7.3.0 Public Addresses.xlsx](./resource/RAG_SDK_7.3.0_public_network_addresses.xlsx).

## Environment Variables

**Table 1**  Environment variables

|Environment Variable|Description|
|--|--|
|PATH|Path to executables.|
|LD_LIBRARY_PATH|Path to dynamic libraries.|
|PYTHONPATH|Default search path for Python module files.|
|HOME|Current user's home directory.|
|PWD|Current working directory.|
|TMPDIR|Temporary directory.|
|LANG|Locale.|
|RAG_SDK_HOME|Working directory of RAG SDK software.|
|ATB_LOG_TO_STDOUT|When set to `1`, records operator acceleration logs to standard output.|
|ATB_LOG_TO_FILE|When set to `1`, records operator acceleration logs to a file.|
|ATB_LOG_LEVEL|Sets the operator acceleration log level. You can set it to `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, or `FATAL`.|
|ENABLE_BOOST|Whether to enable vector model inference acceleration. Set this value to `"True"` or `"False"`.|
|DISABLE_RAGS_LOGGING|When set to `0`, enables `ragas` log output. Disabled by default.|
|RAGAS_DO_NOT_TRACK|Whether to upload RAGAS reports to a remote website. RAG SDK fixes this value to `"true"`, which means reports are not uploaded to a remote website.|
|HF_HUB_OFFLINE|Whether to load only offline files when loading weights. RAG SDK fixes this value to `"1"`, which means only offline weight files are loaded.|
|HF_DATASETS_OFFLINE|Whether to support loading offline Hugging Face data. RAG SDK fixes this value to `"1"`, which means only offline datasets are loaded.|
|AUTO_DOWNLOAD_NLTK|Whether to automatically download the NLTK tokenization model the first time a Markdown document is parsed. RAG SDK fixes this value to `"false"`, which means the NLTK tokenization model is not downloaded automatically.|

> [!NOTE]
> To protect service data security, do not configure `RAGAS_DO_NOT_TRACK`, `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, or `AUTO_DOWNLOAD_NLTK`.

## User Information List

Update user passwords periodically to avoid the risks of using the same password for a long time.

**System users**

**Table 1**  User information list

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|root|OS user for driver installation.|User-defined|Use the `passwd` command to change it.|
|openGauss user name|User name for connecting to the openGauss database.|User-defined|See the official openGauss website.|
|Milvus user name|User name for connecting to the Milvus database.|User-defined|See the official Milvus website.|

**Users in the Ubuntu base image**

|User|Initial Password|Password Change Method|
|--|--|--|
|root|None|-|
|daemon|None|-|
|bin|None|-|
|sys|None|-|
|sync|None|-|
|games|None|-|
|man|None|-|
|lp|None|-|
|mail|None|-|
|news|None|-|
|uucp|None|-|
|proxy|None|-|
|www-data|None|-|
|backup|None|-|
|list|None|-|
|irc|None|-|
|gnats|None|-|
|nobody|None|-|
|_apt|None|-|

## Revision History

|Release Date|Revision Record|
|--|--|
|2025-06-30|First official release.|
