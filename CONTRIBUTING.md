# 为Ascend RAGSDK贡献

感谢您考虑为 Ascend RAGSDK 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等，甚至只是反馈。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

您可以通过多种方式支持本项目：

- 领取任务奖励：[昇腾开源软件社区任务](https://www.hiascend.com/developer/activities/details/5dbf59b2dce14f91afb157f5a52f332d#tab0)
- 贡献前，请先签署开放项目贡献者许可协议[CLA](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)。
- 审查Pull Request并协助其他贡献者。
- 传播项目：在博客文章、社交媒体上分享RAGSDK，或给仓库点个⭐。

请先提前了解社区相关规范：

- [Ascend开源项目行为守则](https://gitcode.com/Ascend/community/blob/master/docs/contributor/code-of-conduct.md)
- [Issue提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-guide.md)
- [Issue处理流程说明](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-workflow-guidelines.md)
- [Ascend社区开发者测试贡献指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)
- [Ascend开源与第三方软件建仓及分支命名指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-repo-branch-guide.md)
- [Ascend开源与第三方软件管理规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-software-management-guide.md)
- [社区安全设计规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/security-design-guideline.md)
- [代码规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
- [PR提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)

## 开发与测试

1. **Fork仓库**：在GitCode平台代码仓库右上角点击"Fork"按钮，Fork一份源代码到个人仓

2. **克隆到本地**：

    将Fork到个人仓的代码克隆到本地进行代码开发

   ```bash
   git clone https://gitcode.com/<your-username>/RAGSDK.git
   ```

3. **创建开发分支**：

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

4. **代码开发**：

    - 代码开发规范请遵循[代码开发规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)。
    - 代码安全规范请遵循[代码安全规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-secure-coding-guide.md)。

5. **开发构建验证**：

    ① 从 [AscendHub](https://www.hiascend.com/developer/ascendhub/detail/ragsdk) 下载 RAGSDK 镜像，运行容器。

    ② 在容器中克隆仓库：

    ```bash
    git clone https://gitcode.com/Ascend/RAGSDK.git
    cd RAGSDK
    ```

    ③ 本地代码开发完成后，进入 `build` 子目录，执行构建脚本：

   ```bash
   cd build
   bash build.sh
   ```

    ④ 构建完成后，进行软件包安装：

   ```bash
   cd ./output/
   pip3 uninstall mx_rag
   pip3 install mx_rag*.whl
   ```

    ⑤ 本地执行UT和补充UT，参见[RAG SDK测试指南](tests/README.md)

6. **执行pre-commit检查**

    本地提交代码前请先执行pre-commit检查，检查指导参见[pre-commit本地运行指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

7. **提交Pull Request**

    提交PR并等待代码审查。

8. **社区评审**

    如果涉及patch、头文件宏、API接口等更新，需提交社区在SIG例会进行评审，社区定期例会与活动参见[会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindSeriesSDK)。
