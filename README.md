# asr-fast

使用 FastAPI 对外提供 faster-whisper 音频转写服务，默认运行在 **Windows x64 + NVIDIA RTX 5070 + CUDA GPU** 上。

- 默认模型：`large-v3`，计算精度：`float16`。
- Python 依赖使用清华 PyPI 镜像，模型通过 `https://hf-mirror.com` 下载。
- 提供音频上传、语言检测、分段时间戳、可选词级时间戳和英文翻译。
- 支持 Bearer API Key、单文件大小限制、音频时长限制和推理并发控制。

以下命令均使用 **PowerShell**，示例部署路径统一为 `D:\services\asr-fast`。如果服务器路径不同，请同步替换命令中的路径。

## 1. 安装系统依赖

建议按以下顺序安装，安装后重新打开 PowerShell，使新的环境变量生效。

| 依赖 | 目标配置 | 安装说明 |
| --- | --- | --- |
| Windows | Windows 11 x64 / Windows Server 2022 x64 | 使用支持目标显卡驱动的系统版本 |
| NVIDIA 驱动 | 支持 RTX 5070 和 CUDA 12.8 的驱动 | 从 [NVIDIA 驱动页面](https://www.nvidia.com/Download/index.aspx)选择实际显卡型号 |
| CUDA Toolkit | **12.8** | 从 [CUDA 12.8 下载页](https://developer.nvidia.com/cuda-12-8-0-download-archive)选择 Windows x86_64 |
| cuDNN | **9.8 或更新的兼容 9.x 版本，CUDA 12 构建** | 从 [cuDNN 下载页](https://developer.nvidia.com/cudnn-downloads)安装支持 Blackwell 的版本 |
| Visual C++ Runtime | x64 | 安装 [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) |
| Python | **3.11 x64**，也可用 3.12 x64 | 从 [Python 官网](https://www.python.org/downloads/windows/)安装，启用 Python Launcher |
| Git | Windows 版 | 从 [Git 官网](https://git-scm.com/downloads/win)安装，用于拉取和更新项目 |

RTX 5070 属于 Blackwell 架构，部署时请明确使用 CUDA **12.8**，不要只确认安装了“CUDA 12”。项目固定 `ctranslate2==4.6.3`，该版本加入了 CUDA 12.8 支持。版本依据见 [CTranslate2 4.6.3 发布说明](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.6.3)；cuDNN 与 Blackwell 的组合见 [NVIDIA cuDNN 9.8 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html)。这是部署目标配置，仍需在实际服务器完成启动和音频转写验证。

本项目使用 CTranslate2 推理，不需要额外安装 PyTorch。通过 wheel 安装的 PyAV 自带 FFmpeg 库，通常无需单独安装 `ffmpeg.exe`，参见 [faster-whisper 官方说明](https://github.com/SYSTRAN/faster-whisper#requirements)。

检查系统环境：

```powershell
nvidia-smi
nvcc --version
py -3.11 --version
```

`nvidia-smi` 应能显示 RTX 5070。它显示的 `CUDA Version` 是驱动支持的 CUDA 版本上限，不能代替 Toolkit 安装检查；Toolkit 版本以 `nvcc --version` 为准。

## 2. 拉取项目并安装 Python 依赖

```powershell
New-Item -ItemType Directory -Force D:\services | Out-Null
git clone https://gitee.com/pinxue/asr-fast.git D:\services\asr-fast
Set-Location D:\services\asr-fast

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check

Copy-Item .env.example .env
notepad .env
```

全程直接调用虚拟环境的 Python，不需要执行 `Activate.ps1` 或修改 PowerShell 执行策略。若安装的是 Python 3.12，把创建环境时的 `py -3.11` 改为 `py -3.12`。

`requirements.txt` 已内置清华源。**pip 不会自动读取 `.env`**；单独安装其他包时，可以显式添加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

## 3. 配置 .env

完整配置和注释见 [`.env.example`](.env.example)。修改 `.env` 后必须重启服务。系统环境变量优先于 `.env`；不要将 `.env` 提交到仓库。

### GPU 与 Windows DLL 路径

保留以下 GPU 默认值：

```dotenv
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_DEVICE_INDEX=0
WHISPER_COMPUTE_TYPE=float16
APP_MAX_CONCURRENT_REQUESTS=1
```

将 CUDA 和 cuDNN 的 **DLL 所在目录**填入 `CUDA_DLL_DIRS`，多个目录用分号分隔。例如：

```dotenv
CUDA_DLL_DIRS="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin;C:/Program Files/NVIDIA/CUDNN/v9.8/bin/12.8"
```

上述路径仅为示例。不同 cuDNN 安装方式可能使用不同目录，请找到实际包含 `cudnn64_9.dll` 等文件的目录；CUDA 目录应能找到 `cublas64_12.dll` 等文件。配置中的每个目录都必须真实存在，不能填写通配符或 DLL 文件名。

如果这些目录已经加入运行账户的系统 `PATH`，`CUDA_DLL_DIRS` 可以留空。作为后台任务运行时，建议显式配置，避免登录账户与后台账户的环境不同。

程序会在导入原生库前注册 DLL 目录。GPU 不可用时启动失败，不会自动切到 CPU。若需手动使用 CPU，须同时改成 `WHISPER_DEVICE=cpu` 和 `WHISPER_COMPUTE_TYPE=int8`。

### 模型镜像与缓存

```dotenv
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=.cache/huggingface
HF_HUB_DISABLE_XET=1
HF_HUB_ETAG_TIMEOUT=30
HF_HUB_DOWNLOAD_TIMEOUT=120
HF_HUB_OFFLINE=0
HF_TOKEN=
WHISPER_DOWNLOAD_ROOT=./models
WHISPER_LOCAL_FILES_ONLY=false
```

首次启动需联网下载模型，公开 Whisper 模型无需填写 `HF_TOKEN`。模型下载和加载完成前，HTTP 服务不会完成启动；此时健康检查可能连接失败。

请保留 `models` 和 `.cache` 目录。相对路径以进程工作目录为基准，前台运行和后台任务均应使用项目根目录作为工作目录。

需要离线部署时，可以在联网机器上使用相同配置完成一次启动，再复制项目、`models` 和 `.cache` 到目标机器；Python 依赖也需提前安装。确认模型文件齐全后设置：

```dotenv
HF_HUB_OFFLINE=1
WHISPER_LOCAL_FILES_ONLY=true
```

也可以将 `WHISPER_MODEL` 指向完整的本地 **CTranslate2 格式**模型目录，其中应包含模型权重、配置和 tokenizer 等文件。它不是原始 PyTorch checkpoint 目录。

### 服务与默认转写参数

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | 监听地址与端口；仅本机访问可改成 `127.0.0.1` |
| `APP_LOG_LEVEL` | `info` | 服务日志级别 |
| `APP_API_KEY` | 空 | 空值关闭鉴权；设置后转写接口需要 Bearer Token |
| `APP_MAX_UPLOAD_MB` | `100` | 单音频文件上限，单位 MiB，在 multipart 解析后检查 |
| `APP_MAX_AUDIO_SECONDS` | `1800` | 解码后的音频时长上限，单位秒 |
| `APP_MAX_CONCURRENT_REQUESTS` | `1` | 同时执行的推理数；调高会增加 GPU 显存压力 |
| `APP_QUEUE_TIMEOUT_SECONDS` | `30` | 等待推理槽位的时限，不是推理执行时限 |
| `WHISPER_CPU_THREADS` | `4` | CTranslate2 CPU 线程数 |
| `WHISPER_LANGUAGE` | 空 | 自动检测语言；主要识别中文可设 `zh` |
| `WHISPER_BEAM_SIZE` | `5` | 解码 beam size，范围 1–20 |
| `WHISPER_VAD_FILTER` | `true` | 使用 VAD 过滤静音 |
| `WHISPER_VAD_MIN_SILENCE_MS` | `500` | VAD 最短静音时长，单位毫秒 |
| `WHISPER_WORD_TIMESTAMPS` | `false` | 是否返回词级时间戳 |

## 4. 前台启动并验证

```powershell
Set-Location D:\services\asr-fast
.\.venv\Scripts\python.exe run.py
```

保留此窗口，在另一个 PowerShell 窗口验证：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

就绪后返回 `{"status":"ok"}`。接口文档位于 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。健康检查表示模型已加载；还应上传实际音频完成一次转写，以验证 GPU 推理及解码链路。

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@D:/audio/test.wav" `
  -F "language=zh" `
  -F "word_timestamps=true"
```

设置了 `APP_API_KEY` 时，调用增加请求头：

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions `
  -H "Authorization: Bearer YOUR_API_KEY" `
  -F "file=@D:/audio/test.wav" `
  -F "language=zh"
```

`/health` 和 `/docs` 不要求 API Key。前台按 `Ctrl+C` 停止服务；完成验证后先停止前台进程，再配置后台运行，避免端口冲突和重复加载模型。

## 5. 后台部署与开机启动

使用 Windows 自带的任务计划程序，无需额外安装进程管理工具。以下配置以 `SYSTEM` 账户运行，开机启动，异常退出后每隔 1 分钟重试，最多重试 3 次；日志追加到项目的 `logs\server.log`。

在服务器上用**管理员 PowerShell**执行一次：

```powershell
$asrRoot = "D:\services\asr-fast"
New-Item -ItemType Directory -Force "$asrRoot\logs" | Out-Null

# cmd 的外层双引号用于包裹完整命令，内部路径也单独加引号。
$asrCommand = '""{0}\.venv\Scripts\python.exe" -u "{0}\run.py" >> "{0}\logs\server.log" 2>&1"' -f $asrRoot
$asrAction = New-ScheduledTaskAction `
  -Execute "$env:SystemRoot\System32\cmd.exe" `
  -Argument "/d /s /c $asrCommand" `
  -WorkingDirectory $asrRoot
$asrTrigger = New-ScheduledTaskTrigger -AtStartup
$asrPrincipal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$asrSettings = New-ScheduledTaskSettingsSet `
  -StartWhenAvailable `
  -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -RestartCount 3 `
  -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "AsrFast" `
  -Action $asrAction -Trigger $asrTrigger `
  -Principal $asrPrincipal -Settings $asrSettings `
  -Description "GPU faster-whisper FastAPI service"
Start-ScheduledTask -TaskName "AsrFast"
```

`SYSTEM` 必须能读取项目、Python 环境和 CUDA/cuDNN DLL，且能写入 `models`、`.cache`、`logs`。若依赖用户专属的代理、网络盘或用户目录，请改用有相应权限的运行账户，或将依赖放到本机固定目录。

此配置通过 `WorkingDirectory` 固定项目目录，并取消计划任务的执行时长上限。参数依据见 Microsoft 的 [任务动作](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtaskaction)和[任务设置](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasksettingsset)文档。

常用维护命令：

```powershell
# 查看状态与日志；Running 不等于模型已就绪，还需检查 /health。
Get-ScheduledTask -TaskName "AsrFast"
Get-ScheduledTaskInfo -TaskName "AsrFast"
Get-Content D:\services\asr-fast\logs\server.log -Tail 100 -Wait

# 停止 / 启动。
Stop-ScheduledTask -TaskName "AsrFast"
Start-ScheduledTask -TaskName "AsrFast"

# 取消开机启动并停止。
Disable-ScheduledTask -TaskName "AsrFast"
Stop-ScheduledTask -TaskName "AsrFast"
```

`Get-Content -Wait` 会持续跟踪日志，按 `Ctrl+C` 结束日志查看。任务停止可能终止正在处理的请求，维护前应先停止接收新请求，等已有转写完成。日志文件不会自动轮转，请定期归档清理。

### 允许其他机器访问

确认 `.env` 中 `APP_HOST=0.0.0.0`，并设置 `APP_API_KEY`。如需允许局域网直接访问，在管理员 PowerShell 中添加防火墙规则；下面网段仅为示例，应替换成实际客户端网段：

```powershell
New-NetFirewallRule -DisplayName "AsrFast TCP 8000" `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 `
  -RemoteAddress 192.168.1.0/24
```

客户端改为访问 `http://服务器IP:8000`；`0.0.0.0` 是监听地址，不是客户端访问地址。云服务器还需配置对应的安全组。

公网部署应在 HTTPS 反向代理后提供服务，并在代理上设置总请求体大小和适合长音频的响应等待时间。应用的上传限制是在 multipart 解析后检查，不能替代代理的总请求体限制；音频时长限制也在解码后生效。客户端断开连接不会立即中止已经开始的原生推理。

`run.py` 固定使用 **1 个进程**，单张 GPU 不要额外启动多个 worker。需要增加并发时，先根据实际显存和音频负载评估，再调整 `APP_MAX_CONCURRENT_REQUESTS`。

### 更新部署

以下命令在管理员 PowerShell 中按顺序执行，任一步失败应先处理错误，再继续：

```powershell
Stop-ScheduledTask -TaskName "AsrFast"
Set-Location D:\services\asr-fast
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
# 对照新版 .env.example 手动补充 .env，保留现有路径和 API Key。
Start-ScheduledTask -TaskName "AsrFast"
```

更新后重新检查日志、`/health` 和一段真实音频的转写结果。不要用 `Copy-Item .env.example .env` 覆盖已有生产配置。

## 6. 接口说明

### GET /health

模型加载完成时返回 `200` 和 `{"status":"ok"}`。启动尚未完成时可能无法连接；模型未就绪时接口返回 `503`。

### POST /v1/audio/transcriptions

使用 `multipart/form-data` 上传音频，不使用 JSON 请求体。支持格式取决于 PyAV 解码能力，常见格式包括 WAV、MP3、M4A、FLAC、OGG、WebM。

| 字段 | 必填 | 默认与说明 |
| --- | --- | --- |
| `file` | 是 | 音频文件 |
| `language` | 否 | 例如 `zh` / `en`；缺省使用 `WHISPER_LANGUAGE`，配置为空时自动检测 |
| `task` | 否 | `transcribe` 保留原语言；`translate` 将语音翻译为英文 |
| `beam_size` | 否 | 使用环境默认值，范围 1–20 |
| `temperature` | 否 | `0`，范围 0–1 |
| `initial_prompt` | 否 | 提示文本，最长 2000 个字符 |
| `vad_filter` | 否 | 使用环境默认值，传 `true` / `false` |
| `word_timestamps` | 否 | 使用环境默认值，传 `true` / `false` |

响应结构示例（数值仅用于说明格式）：

```json
{
  "text": "你好。",
  "language": "zh",
  "language_probability": 1.0,
  "duration": 2.5,
  "duration_after_vad": 1.5,
  "segments": [
    {"id": 1, "start": 0.0, "end": 1.5, "text": "你好。", "words": null}
  ]
}
```

时间及音频时长单位均为秒。启用词级时间戳后，`words` 返回包含 `start`、`end`、`word`、`probability` 的对象数组。此服务实现以上表单接口，未实现 OpenAI 接口的全部参数或流式返回能力。

| HTTP 状态码 | 含义 |
| --- | --- |
| `400` | 上传文件为空 |
| `401` | 缺少或提供了错误的 API Key |
| `413` | 文件过大或解码后时长超限 |
| `422` | 表单参数错误、音频无法解码或语言不受支持 |
| `503` | 模型未就绪，或等待推理槽位超时；容量不足时响应含 `Retry-After` |
| `500` | 推理异常，详细原因需查看服务日志 |

## 7. 常见问题

| 现象 | 检查方式与处理 |
| --- | --- |
| 找不到 `cublas64_12.dll` / `cudnn64_9.dll` | 确认 CUDA 12.8 和匹配的 cuDNN 已安装，检查 `CUDA_DLL_DIRS` 是否指向实际 DLL 目录；后台账户需能访问这些路径 |
| DLL 存在但仍报加载失败 | 检查 VC++ x64 Runtime、DLL 的依赖，以及 PATH 中是否混入旧版 CUDA/cuDNN；修改后重启进程 |
| `Configured CUDA device unavailable` | 先确认 `nvidia-smi` 可用，再检查驱动、DLL 路径和 `WHISPER_DEVICE_INDEX`；只有一张卡时通常设为 `0` |
| `no kernel image`、架构或精度相关错误 | 检查是否仍安装了旧版 CTranslate2 或较早的 CUDA 12；确认使用本项目虚拟环境、固定的 CTranslate2 4.6.3 和 `float16` |
| 首次启动长时间未就绪 | 查看日志确认模型下载进度；检查 HF Mirror 网络连通性、磁盘空间和下载目录写权限；首次下载不要启用离线模式 |
| GPU 显存不足 | 确认没有重复服务进程或其他 GPU 任务，保持并发为 1；仍不足时可改用 `turbo` / `small` 并重启 |
| 本机可访问，其他机器不可访问 | 检查监听地址、Windows 防火墙、服务器 IP、客户端网段和云安全组 |
| 返回 `503` | 查看是否已就绪；若为容量不足，可按 `Retry-After` 重试、减少请求数或调整排队等待时间 |
| 计划任务显示运行，但接口失败 | 查看 `logs\server.log`；检查运行账户权限、实际 DLL 路径、工作目录以及模型下载是否完成 |

仓库目前提供代码、配置和部署步骤；Windows + RTX 5070 的实际安装、开机启动及真实音频转写需在目标服务器验收。
