# Version Mapping

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-30T02:42:53.095Z -->

## Product Version

| Product | RAG SDK |
| --- | --- |
| Product Version | 26.0.0 |
| Version Type | Release Version |

## Related Product Versions

| Product | Version |
| --- | --- |
| Ascend HDK | 26.0.RC1 |
| CANN | 9.0.0 |

## Virus Scan Results

Virus scan passed.

# Version Compatibility

- RAG SDK: After upgrading to this version, you need to install Index SDK and generate operators before using ascendfaiss for retrieval.

**Table 1** Software Version Compatibility Description

| MindSDK Software Version | MindSDK Version to Upgrade | CANN Version Compatibility | Ascend HDK Version Compatibility |
|--|--|--|--|
| RAG SDK 26.0.0 |<li>MindSDK 6.0.RC3 and patch versions</li><li>MindSDK 6.0.0 and patch versions</li><li>MindSDK 7.0.RC1 and patch versions</li><li>MindSDK 7.1.RC1 and patch versions</li><li>MindSDK 7.2.RC1 and patch versions</li><li>MindSDK 7.3.0 and patch versions</li>|<li>CANN 8.1.RC1 and patch versions</li><li>CANN 8.2.RC1 and patch versions</li><li>CANN 8.3.RC1 and patch versions</li><li>CANN 8.5.0 and patch versions</li><li>CANN 9.0.0 and patch versions</li>|<li>Ascend HDK 25.0.RC1 and patch versions</li><li>Ascend HDK 25.2.0 and patch versions</li><li>Ascend HDK 25.3.RC1 and patch versions</li><li>Ascend HDK 25.5.0 and patch versions</li><li>Ascend HDK 26.0.RC1 and patch versions</li>|

> [!NOTE]
> Software version compatibility means that when the product software version is upgraded, other related software does not need to be upgraded or patched at the same time, and existing functions remain supported.

# Important Notes

None

# Update Notes

## New Features

| Feature | Description | Supported Product Model |
|--|------------------------------------|--|
| RAG SDK | bge series embedding and reranker acceleration. Performance optimization for the bge-reranker-v2-m3 and bge-m3 models. | Atlas 300I Duo Inference Card<br>Atlas 800I A2 Inference Server |

## Service Interface Changes

**RAG SDK**

- No interface changes are involved.

## Key Feature Changes

None

## Resolved Issues

None

## Known Issues

None

# Upgrade Impact

## Impact on the System during the Upgrade

None

## Impact on the System After the Upgrade

None

# 26.0.0 Documentation

| Document | Description | Release Notes |
|--|--|--|
| *RAG SDK 26.0.0 User Guide* | Mainly includes RAG SDK installation and deployment process, application development process, API interface descriptions, and other common operations. | For changes, see *[RAG SDK 26.0.0 User Guide](./introduction.md)*. |

# Fixed Vulnerabilities

None
