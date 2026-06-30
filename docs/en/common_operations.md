# Common Operations

## Log Description

RAG SDK installation packages are in .run format. Installation and uninstallation logs for RAG SDK are recorded in `~/log/mxRag/deployment.log`.

RAG SDK runtime log module uses `loguru` and outputs to the console by default. If needed, configure output redirection to a file.

To prevent log injection security issues, for example, escaped special characters such as `\n` and `\b`, configure the `LOGURU_FORMAT` environment variable. The sample configuration below mainly uses `{message!r}` in `message` to ensure that special characters are handled safely. Set the other parameters according to user preferences.

```bash
export LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message!r}</level>'
```
