# SV4Agent 安装与部署指南

[项目介绍](README.md) · [English introduction](README.en.md) · [配置示例](.env.example)

使用 FastAPI 对外提供 faster-whisper 音频转写服务，默认运行在 **Windows x64 + NVIDIA RTX 5070 + CUDA GPU** 上。

- 默认模型：[ModelScope 的 Systran/faster-whisper-large-v3](https://www.modelscope.cn/models/Systran/faster-whisper-large-v3)，计算精度：`float16`。
- Python 依赖使用清华 PyPI 镜像，模型通过 ModelScope 下载。
- 提供音频上传、语言检测、分段时间戳、可选词级时间戳和英文翻译。
- 支持 Bearer API Key、单文件大小限制、音频时长限制和推理并发控制。
- 提供 FFmpeg GPU 视频转码接口：CUDA 解码/缩放、NVENC H.264/H.265 编码，返回 MP4。
- 支持上传图片时间线和音频，直接用 NVENC 生成照片视频，无需本地先编码。

以下命令均使用 **PowerShell**，示例部署路径统一为 `D:\services\SV4Agent`。如果服务器路径不同，请同步替换命令中的路径；既有部署可以保留原来的项目目录。

## 1. 安装系统依赖

建议按以下顺序安装，安装后重新打开 PowerShell，使新的环境变量生效。

| 依赖 | 目标配置 | 安装说明 |
| --- | --- | --- |
| Windows | Windows 11 x64 / Windows Server 2022 x64 | 使用支持目标显卡驱动的系统版本 |
| NVIDIA 驱动 | 支持 RTX 5070 和 CUDA 12.8 的驱动 | 从 [NVIDIA 驱动页面](https://www.nvidia.com/Download/index.aspx)选择实际显卡型号 |
| CUDA Toolkit | **12.8** | 从 [CUDA 12.8 下载页](https://developer.nvidia.com/cuda-12-8-0-download-archive)选择 Windows x86_64 |
| cuDNN | **9.8 或更新的兼容 9.x 版本，CUDA 12 构建** | 从 [cuDNN 下载页](https://developer.nvidia.com/cudnn-downloads)安装支持 Blackwell 的版本 |
| Visual C++ Runtime | x64 | 安装 [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) |
| Python | **标准 CPython 3.13 x64** | 从 [Python 官网](https://www.python.org/downloads/windows/)安装，启用 Python Launcher；不要选择 free-threaded `3.13t` 构建 |
| Git | Windows 版 | 从 [Git 官网](https://git-scm.com/downloads/win)安装，用于拉取和更新项目 |

RTX 5070 属于 Blackwell 架构，部署时请明确使用 CUDA **12.8**，不要只确认安装了“CUDA 12”。项目固定 `ctranslate2==4.6.3`，该版本加入了 CUDA 12.8 支持。版本依据见 [CTranslate2 4.6.3 发布说明](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.6.3)；cuDNN 与 Blackwell 的组合见 [NVIDIA cuDNN 9.8 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html)。这是部署目标配置，仍需在实际服务器完成启动和音频转写验证。

本项目使用 CTranslate2 推理，不需要额外安装 PyTorch。音频转写使用的 PyAV wheel 自带 FFmpeg 库，无需单独安装 `ffmpeg.exe`，参见 [faster-whisper 官方说明](https://github.com/SYSTRAN/faster-whisper#requirements)。新增的视频转码接口需要另外安装下文的 FFmpeg 可执行程序。

Windows 依赖固定 `av==16.0.1` 作为兼容性基线，该版本提供 [CPython 3.13 / Windows x64 wheel](https://pypi.org/project/av/16.0.1/#files)。这用于对照排查 PyAV 19.0.1 在目标服务器单独导入也长时间未完成的问题；尚不能仅凭版本回退认定服务器故障已解决。

检查系统环境：

```powershell
nvidia-smi
nvcc --version
py -3.13 --version
```

`nvidia-smi` 应能显示 RTX 5070。它显示的 `CUDA Version` 是驱动支持的 CUDA 版本上限，不能代替 Toolkit 安装检查；Toolkit 版本以 `nvcc --version` 为准。

## 2. 拉取项目并安装 Python 依赖

```powershell
New-Item -ItemType Directory -Force D:\services | Out-Null
git clone https://github.com/Hold-API/SV4Agent.git D:\services\SV4Agent
Set-Location D:\services\SV4Agent

py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check

Copy-Item .env.example .env
notepad .env
```

全程直接调用虚拟环境的 Python，不需要执行 `Activate.ps1` 或修改 PowerShell 执行策略。若已有其他 Python 版本创建的 `.venv`，先停止服务并移走旧环境，再用 Python 3.13 重新创建，不能直接复用旧环境。

项目固定的 CTranslate2 4.6.3 已发布 `cp313-cp313-win_amd64` wheel，见 [PyPI 文件列表](https://pypi.org/project/ctranslate2/4.6.3/#files)。这里使用普通 CPython 3.13，未验证 free-threaded 构建。

安装后确认虚拟环境实际使用的版本：

```powershell
.\.venv\Scripts\python.exe --version
```

`requirements.txt` 已内置清华源。**pip 不会自动读取 `.env`**；单独安装其他包时，可以显式添加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

## 3. 配置 .env

完整配置和注释见 [`.env.example`](.env.example)。修改 `.env` 后必须重启服务。系统环境变量优先于 `.env`；不要将 `.env` 提交到仓库。

### GPU 与 Windows DLL 路径

保留以下 GPU 默认值：

```dotenv
WHISPER_MODEL=Systran/faster-whisper-large-v3
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

### ModelScope 模型下载与缓存

```dotenv
MODELSCOPE_ENDPOINT=https://modelscope.cn
MODELSCOPE_API_TOKEN=
MODELSCOPE_REVISION=master
MODELSCOPE_API_TIMEOUT=30
MODELSCOPE_DOWNLOAD_TIMEOUT=120
WHISPER_MODEL=Systran/faster-whisper-large-v3
WHISPER_DOWNLOAD_ROOT=./models/modelscope
WHISPER_LOCAL_FILES_ONLY=false
```

首次启动会从指定的 ModelScope 仓库下载模型，公开仓库无需填写 `MODELSCOPE_API_TOKEN`。模型下载和加载完成前，HTTP 服务不会完成启动；此时健康检查可能连接失败。

程序通过 ModelScope SDK 显式传入 endpoint、token、revision、缓存目录及仅本地模式。`MODELSCOPE_REVISION` 为仓库分支、标签或版本，默认 `master`；两个超时参数由 SDK 在导入时读取，单位为秒，不是整个启动过程的总时限。应用不会在下载失败后切换到 Hugging Face；ModelScope 自身仍可能将文件请求重定向到对象存储/CDN。

启动日志会显示 `.env` 的绝对路径、文件是否存在，以及 `Model download source: ModelScope; endpoint=...`。若显示文件不存在，请确认文件名确实是 `.env`，而不是 `.env.txt` 或只有 `.env.example`。若进程环境覆盖了文件中的 `MODELSCOPE_ENDPOINT`，会输出覆盖警告；需要调整启动进程的同名环境变量并重启。编辑 `.env.example` 不会改变运行配置。

可在项目目录单独检查有效配置，不会下载模型或输出 token：

```powershell
.\.venv\Scripts\python.exe -c "from app.config import ENV_FILE, get_settings; s=get_settings(); print('env_file:', ENV_FILE); print('exists:', ENV_FILE.is_file()); print('MODELSCOPE_ENDPOINT:', s.modelscope_endpoint); print('WHISPER_MODEL:', s.whisper_model); print('WHISPER_LOCAL_FILES_ONLY:', s.whisper_local_files_only)"
```

`WHISPER_MODEL` 可填写完整 ModelScope 仓库 ID，或填写 `large-v3` / `large`（均映射到 `Systran/faster-whisper-large-v3`）。也支持 `tiny` / `base` / `small` / `medium` 及对应 `.en` 版本、`large-v1`、`large-v2`；`turbo` / `large-v3-turbo` 映射到 ModelScope 的 `pengzhendong/faster-whisper-large-v3-turbo`。选择其他仓库时，必须确认它提供 CTranslate2 格式的权重和 tokenizer。

请保留 `models` 目录；相对的本地模型路径和 `WHISPER_DOWNLOAD_ROOT` **以项目根目录为基准**。前台运行和后台任务仍统一使用项目根目录作为工作目录。

### 手动下载与离线加载

安装项目依赖后，可在项目目录执行以下命令，下载同一个 Systran 模型到指定目录。命令依据见 [ModelScope 官方下载说明](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md#download)。

```powershell
.\.venv\Scripts\modelscope.exe download --model Systran/faster-whisper-large-v3 --local_dir .\models\large-v3
```

下载成功后，将 `.env` 改为：

```dotenv
WHISPER_MODEL=./models/large-v3
WHISPER_LOCAL_FILES_ONLY=true
```

程序会校验 `config.json`、`model.bin`、`tokenizer.json` 和词表文件（`vocabulary.json` 或 `vocabulary.txt`），并拒绝空的必要文件、Git LFS 指针权重。校验通过后仅从本地目录加载推理模型；这项文件检查不等于完成了真实推理验证。

需要离线部署时，复制下载完整的模型目录到服务器，将 `WHISPER_MODEL` 指向它，并设置 `WHISPER_LOCAL_FILES_ONLY=true`；Python 依赖和 CUDA/cuDNN 仍需提前安装。也可保留仓库 ID，使用已有的 ModelScope 缓存配合仅本地模式启动；缺失缓存会报错，不会自动联网。

### 从旧 HF 配置升级

更新代码后重新执行 `pip install -r requirements.txt`，并对照新版 `.env.example` 更新配置：

- `WHISPER_MODEL` 使用 `Systran/faster-whisper-large-v3`，或保留已有完整本地模型目录。
- 自动下载使用 `WHISPER_DOWNLOAD_ROOT=./models/modelscope`，旧 HF 缓存目录不会自动迁移。
- 本地已有完整的 CTranslate2 模型可以继续复用，不需要因为更换下载平台而重新下载。
- 旧 `HF_ENDPOINT`、`HF_TOKEN`、`HF_HOME`、`HF_HUB_OFFLINE` 等配置不再控制本项目下载流程；按上文配置 ModelScope 和 `WHISPER_LOCAL_FILES_ONLY` 即可。

faster-whisper 自身仍依赖 `huggingface-hub` 包，因此安装列表中出现它属于正常情况；项目下载调用已切换到 ModelScope，传入推理引擎的始终是经过校验的本地目录。

### 服务与默认转写参数

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | 监听地址与端口；仅本机访问可改成 `127.0.0.1` |
| `APP_LOG_LEVEL` | `info` | 服务日志级别 |
| `APP_API_KEY` | 空 | 空值关闭鉴权；设置后转写和视频接口需要 Bearer Token |
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
Set-Location D:\services\SV4Agent
.\.venv\Scripts\python.exe run.py
```

保留此窗口，在另一个 PowerShell 窗口验证：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

就绪后返回 `{"status":"ok"}`。接口文档位于 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。健康检查表示模型已加载；还应上传实际音频完成一次转写，以验证 GPU 推理及解码链路。

`/docs`（Swagger UI）和 `/redoc`（ReDoc）的 JS、CSS、图标随项目放在 `app/static/docs`，通过同一服务的 `/static/docs/...` 提供，无需浏览器访问外部 CDN，也不会在启动时下载前端资源。ReDoc 不加载 Google Fonts，Swagger UI 不调用外部 schema validator。部署时请保留整个静态资源目录；版本、来源、许可证和校验值见 [资源说明](app/static/docs/README.md)。实现方式参考 [FastAPI 本地文档资源说明](https://fastapi.tiangolo.com/how-to/custom-docs-ui-assets/)。

如果旧页面因 CDN 超时而空白，更新代码并重启后按 `Ctrl+F5` 强制刷新。可分别访问 `/static/docs/swagger-ui-bundle.js`、`/static/docs/swagger-ui.css` 和 `/openapi.json`，检查是否由当前服务器返回成功。使用反向代理时，这些路径也需转发到同一服务；应用配置了 ASGI `root_path` 时，文档资源与 OpenAPI 地址会带上对应前缀。

如果只看到 `Started server process` 和 `Waiting for application startup`，说明 HTTP 启动尚未完成，不能仅凭这两行判断进程已经失败。保持 `APP_LOG_LEVEL=info`，程序会输出 DLL 配置、原生库导入、CUDA 检查、模型文件下载、GPU 模型加载等阶段；等待期间每 15 秒输出一次当前阶段和累计耗时。等待提示不代表文件下载进度，也不会强制中断正在运行的原生操作。

排查时可用以下方式前台启动，将标准输出和错误输出一起保存：

```powershell
Set-Location D:\services\SV4Agent
.\.venv\Scripts\python.exe -u -X faulthandler run.py 2>&1 | Tee-Object -FilePath startup.log
```

最后一个 `Startup stage` 可用于定位停顿位置：下载阶段先检查镜像网络和磁盘，CUDA 检查或模型加载阶段检查驱动、DLL 和显存。Python 启动异常会记录失败阶段及堆栈；`run.py` 也启用了 faulthandler，便于捕获部分原生扩展崩溃。排查时请同时保留末尾日志，并确认进程是仍在运行还是已退出。

### PyAV 原生模块导入停顿

若日志停在 `import PyAV (FFmpeg native libraries)`，且单独 `import av` 也很慢，此时不涉及模型下载或 large-v3 显存占用。先停止服务，在项目根目录重新安装固定版本的二进制包：

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir --only-binary=:all: "av==16.0.1" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -u -c "import faulthandler, time; from importlib.metadata import version; print('Installed PyAV:', version('av'), flush=True); faulthandler.dump_traceback_later(30); t=time.perf_counter(); import av; faulthandler.cancel_dump_traceback_later(); print('PyAV import OK:', av.__version__, 'elapsed:', round(time.perf_counter()-t, 2), 'seconds', flush=True)"
```

确认输出 `Installed PyAV: 16.0.1` 和 `PyAV import OK` 后，再运行 `python run.py` 并验证真实音频转写。如果仍卡住，保留 30 秒后的堆栈，继续检查本机 DLL 加载、磁盘和安全软件行为。主线程单独导入也复现时，不应直接归因于 FastAPI 后台线程。

Windows PowerShell 将 stderr 合并进管道时，可能把正常的 Uvicorn INFO 日志包装成 `NativeCommandError`；应结合后续 Python 异常、进程退出和启动阶段判断是否失败。

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

## FFmpeg GPU 视频转码

### Windows 安装和配置

从 [FFmpeg 官方下载页的 Windows 构建入口](https://ffmpeg.org/download.html#build-windows)选择包含 NVIDIA CUDA、NVENC 和 `scale_cuda` 的构建，解压，例如放在 `D:\tools\ffmpeg`。不需要安装 `ffmpeg-python`，本项目不新增 Python 依赖，也不会自动下载 FFmpeg。

在 PowerShell 中检查实际解压路径及编译能力：

```powershell
& "D:\tools\ffmpeg\bin\ffmpeg.exe" -version
& "D:\tools\ffmpeg\bin\ffmpeg.exe" -hide_banner -encoders | Select-String "h264_nvenc|hevc_nvenc"
& "D:\tools\ffmpeg\bin\ffmpeg.exe" -hide_banner -hwaccels
& "D:\tools\ffmpeg\bin\ffmpeg.exe" -hide_banner -filters | Select-String "scale_cuda"
```

应能看到两个 NVENC 编码器、`cuda` 硬件加速和 `scale_cuda` 滤镜。这只证明构建包含相应功能，仍需真实视频验证驱动和显卡可用。FFmpeg 所用 Video Codec SDK 对驱动有最低版本要求，请按构建说明更新 NVIDIA 驱动，参见 [NVIDIA FFmpeg 加速指南](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)。FFmpeg 硬件视频编码主要使用 NVENC/NVDEC 单元，任务管理器请观察 Video Encode / Video Decode；ASR 主要使用 CUDA/Tensor 计算，两者仍会共享显存和部分资源。

将以下内容加入**实际 `.env`**，然后重启服务：

```dotenv
FFMPEG_ENABLED=true
FFMPEG_PATH="D:/tools/ffmpeg/bin/ffmpeg.exe"
FFMPEG_GPU_INDEX=0
FFMPEG_MAX_UPLOAD_MB=1024
FFMPEG_MAX_OUTPUT_MB=2048
FFMPEG_TIMEOUT_SECONDS=600
FFMPEG_MAX_CONCURRENT_REQUESTS=1
FFMPEG_QUEUE_TIMEOUT_SECONDS=30
FFMPEG_TEMP_DIR=D:/asr-fast-temp
```

`FFMPEG_PATH` 也可填系统 `PATH` 中的命令名 `ffmpeg`；后台计划任务建议填写完整路径。`FFMPEG_TEMP_DIR` 可省略，默认使用系统临时目录；指定目录时会自动创建，运行账户需有写入权限。该目录存放上传副本和转码结果，需预留磁盘空间；响应发送完成或处理失败后清理各任务目录。进程被强制结束可能留下临时目录，可在停止服务后清理 `asr-fast-video-*`。

视频与 ASR 使用独立并发队列和上传限制，默认各执行一个任务，因此可能同时占用同一张 GPU。根据实际显存和吞吐调整并发；视频默认允许 1 GiB 输入和 2 GiB 输出，超限返回 413，输出达到上限时丢弃结果。`FFMPEG_TIMEOUT_SECONDS` 是 FFmpeg 执行时限，排队另有时限。反向代理需要同时放行对应的请求体大小和等待时长。

视频功能在请求时调用 FFmpeg，缺失程序不会阻止原有 ASR 启动；但服务仍需完成 Whisper 模型加载后才能接受 HTTP 请求。`FFMPEG_ENABLED=false` 可关闭视频接口。

### 调用接口

`POST /v1/video/transcode` 接收 `multipart/form-data`，成功直接返回 `video/mp4` 文件。支持自包含的 MP4/MOV、MKV/WebM、AVI、TS、MPEG、FLV、OGV 视频，上传内容必须包含视频流。只处理首条视频流和首条可选音轨；音频转为双声道 AAC 192k，不保留字幕、章节和附加音轨。不接受 URL、播放列表或任意 FFmpeg 参数。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `file` | 必填 | 视频文件 |
| `codec` | `h264` | `h264` 或 `hevc`（H.265）；H.264 兼容性更广 |
| `preset` | `p4` | `p1`–`p7`，p1 最快，p7 更重视压缩效率 |
| `quality` | `23` | CQ 质量值 1–51，越低质量越高、通常文件越大；不保证小于原文件 |
| `height` | 留空 | 可选目标高度，144–4320 的偶数；宽度按比例计算，留空保留尺寸 |

```powershell
# H.264 GPU 转码，保留尺寸
curl.exe --fail http://127.0.0.1:8000/v1/video/transcode `
  -F "file=@D:/video/input.mp4" `
  -o D:/video/output.mp4

# H.265 GPU 转码并等比例缩放至 1080 高
curl.exe --fail http://127.0.0.1:8000/v1/video/transcode `
  -F "file=@D:/video/input.mp4" `
  -F "codec=hevc" `
  -F "height=1080" `
  -F "preset=p4" `
  -F "quality=23" `
  -o D:/video/output-hevc.mp4
```

启用 `APP_API_KEY` 后，两个命令均需添加 `-H "Authorization: Bearer YOUR_API_KEY"`。也可在 `/docs` 的 `video` 分组上传视频并下载结果。此接口用于加快处理速度，输出保持原始播放速度，不做视频倍速播放或 ASR 转写。

当前输出为 8-bit YUV 4:2:0，面向 SDR 视频；未实现 HDR 到 SDR 色调映射。不支持硬件解码的输入、缺少 CUDA/NVENC 功能的构建或不可用的 GPU 会报错，不自动切到软件编码。HTTP 422 表示输入或转码失败，查看服务器 FFmpeg 错误日志区分；503 表示功能关闭、程序无法启动或队列已满；504 表示转码超时。缺少必填文件或参数不合法也返回 422。

部署验收应上传一段带音轨的视频及一段无音轨视频，检查输出可播放、音画同步、分辨率和编码正确，并观察 GPU Video Encode / Decode。仓库自动化测试覆盖接口、参数、子进程超时和清理；本地开发环境未验证 Windows/5070 实际硬件转码速度。

### 图片时间线 + 音频直接生成视频

`POST /v1/video/slideshow` 接收多张图片、一段音频和一个 `concat.txt`，成功直接返回 MP4。它将“本地 libx264 先编码 → 上传 raw MP4 → GPU 再编码”合并为一次服务端制作：图片只预处理一次，按时间线切换，最终使用 NVENC 编码。照片预处理使用 CPU，长视频编码使用 GPU；不需要在客户端安装 FFmpeg，也不新增 Python 依赖。

| 表单字段 | 默认值 | 含义 |
| --- | --- | --- |
| `images` | 必填，可重复 | JPG/JPEG、PNG 或 WebP 图片，按文件名匹配时间线；大小写不敏感 |
| `audio` | 必填 | 录音或配音；默认适合 AAC 编码的 `.m4a` |
| `timeline` | 必填 | UTF-8（允许 BOM）的 `concat.txt`，最大 256 KiB |
| `codec` | `h264` | `h264` 或 `hevc` |
| `preset` | `p7` | `p1`–`p7`；较快编码可先试 `p4` |
| `quality` | `23` | NVENC CQ 质量值，1–51；不是 libx264 的 CRF |
| `width` / `height` | `1920` / `1080` | 偶数画布尺寸；每边 144–3840，总像素不超过 3840×2160 |
| `fps` | `24` | 输出帧率，1–60 |
| `audio_codec` | `copy` | `copy` 无损复制音轨，需兼容 MP4；设为 `aac` 可转换其他音频格式 |

时间线严格按文件顺序执行，`duration` 单位是秒，必须为有限正数且不少于一帧。图片等比例适配画布并补黑边，不拉伸；动画图片只取第一帧。时间线只接受 `file`、`duration`、可选的 `ffconcat version 1.0` 首部及注释，不接受任意 FFmpeg 指令。

例如 `concat.txt`：

```text
ffconcat version 1.0
file 'C:\photos\first.jpg'
duration 5.000
file 'C:\photos\second.jpg'
duration 8.000
```

上传 `first.jpg`、`second.jpg` 和音频后，第一张从 0 秒显示至 5 秒，第二张继续至 13 秒。**Windows 路径仅用于提取图片文件名，不会读取服务器上的同名路径**；无需将客户端目录复制到服务器。不同目录中的同名图片需要先重命名。缺失图片、重复文件名、上传了时间线未引用的图片、缺少时长都会报错，不会跳过某段。

服务会重新生成只引用本次上传图片的 concat 清单，并重复最后一张作为结束时间戳，保证最后一段按指定时长显示，无需手动追加重复行。也兼容客户端已追加最后一张、无 `duration` 的标准结束行。成片在**音频结束和时间线结束中较早的位置**停止，按帧对齐；音频较短时不会补静音，较长时会截断。行为依据见 [FFmpeg concat demuxer](https://ffmpeg.org/ffmpeg-formats.html#concat-1) 和 [shortest 选项](https://ffmpeg.org/ffmpeg.html#Advanced-options)。

上传一个短示例：

```powershell
curl.exe --fail http://10.234.10.20:8000/v1/video/slideshow `
  -F "timeline=@D:/photos/concat.txt" `
  -F "images=@D:/photos/first.jpg" `
  -F "images=@D:/photos/second.jpg" `
  -F "audio=@D:/audio/recording.m4a" `
  -F "audio_codec=copy" `
  -o D:/video/slideshow.mp4
```

启用鉴权时，增加 `-H "Authorization: Bearer YOUR_API_KEY"`。可在 `/docs` 的 `video` 分组选择多张图片并上传时间线、音频；这是同步请求，连接需要持续到编码完成，不是后台任务或任务轮询接口。

#### 使用 2026-10-12 的 14 张照片和录音

已保存完整时间线到 [`examples/concat_20261012.txt`](examples/concat_20261012.txt)，首张为 `IMG_20261012_084558.jpg`，14 段共 **5164 秒（86 分 4 秒）**。调整下方客户端素材路径，运行 PowerShell 辅助脚本即可上传全部引用的图片：

```powershell
Set-Location D:\services\SV4Agent
.\examples\create_slideshow.ps1 `
  -PhotoDirectory "C:\path\to\photos_1012" `
  -AudioPath "C:\path\to\20261012_082607.m4a" `
  -ServerUrl "http://10.234.10.20:8000" `
  -OutputPath ".\20261012_082607.mp4"
```

脚本从时间线提取图片文件名，从 `PhotoDirectory` 找到实际文件，因此示例 concat 里的占位目录可以保留。自定义时间线通过 `-TimelinePath` 指定；脚本要求 `file` 路径使用示例中的单引号格式。启用鉴权时增加 `-ApiKey "YOUR_API_KEY"`。如本机 PowerShell 策略不允许执行脚本，可用上方 `curl.exe` 方式按 `images` 字段逐张上传。音频不是 MP4 支持的编码时，在 `/docs` 或直接请求中选择 `audio_codec=aac`。

在实际 `.env` 添加并重启：

```dotenv
FFMPEG_SLIDESHOW_MAX_IMAGES=100
FFMPEG_SLIDESHOW_MAX_DURATION_SECONDS=86400
FFMPEG_SLIDESHOW_TIMEOUT_SECONDS=3600
```

`FFMPEG_MAX_UPLOAD_MB` 控制图片、音频和时间线的**合计上传大小**；`FFMPEG_MAX_OUTPUT_MB` 仍限制 MP4 输出大小。图片合成与 `/v1/video/transcode` 共用 `FFMPEG_MAX_CONCURRENT_REQUESTS` 队列。`FFMPEG_SLIDESHOW_TIMEOUT_SECONDS` 是所有图片预处理和最终编码的整体执行时限，不含上传、排队和下载，与普通转码的 `FFMPEG_TIMEOUT_SECONDS` 分开。超过时限返回 504 并结束子进程，输入/输出超限返回 413，失败或响应发送完成后清理临时文件。长视频使用反向代理时也需要调大等待时限；客户端请勿设置比编码时间更短的超时。

直接从原始图片制作可避免低画质中间视频造成的二次有损压缩。把 CRF 35 的视频重新编码为 CQ 23 并不能恢复已丢失的细节；`p7` 也不保证固定的体积或速度。请先制作短片验证画面、音轨、切换时刻和时长，再处理完整录音。当前自动化测试不等于已完成 Windows/5070 硬件编码或 86 分钟素材的实际部署验收。

## 5. 后台部署与开机启动

使用 Windows 自带的任务计划程序，无需额外安装进程管理工具。以下配置以 `SYSTEM` 账户运行，开机启动，异常退出后每隔 1 分钟重试，最多重试 3 次；日志追加到项目的 `logs\server.log`。

在服务器上用**管理员 PowerShell**执行一次：

```powershell
$asrRoot = "D:\services\SV4Agent"
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
Get-Content D:\services\SV4Agent\logs\server.log -Tail 100 -Wait

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
Set-Location D:\services\SV4Agent
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
| 首次启动长时间未就绪 | 查看日志确认模型下载阶段；检查 ModelScope 网络连通性、磁盘空间和缓存目录写权限；首次下载保持 `WHISPER_LOCAL_FILES_ONLY=false` |
| ModelScope 配置似乎未生效 | 查看 `.env` 的实际路径及 `Model download source: ModelScope`；排除 `.env.txt`、只修改 `.env.example`、进程环境变量覆盖和修改后未重启 |
| 本地模型提示文件不完整 | 确认目录内包含 `model.bin`、`config.json`、`tokenizer.json` 和词表；不要把 Hugging Face 缓存根目录或未下载的 Git LFS 指针当成模型目录 |
| 停在 `Waiting for application startup` | 保持 `APP_LOG_LEVEL=info`，查看最后的 `Startup stage` 和每 15 秒的等待提示；使用前台日志命令保存输出，确认进程是否退出 |
| 停在 PyAV 导入，CPU 占用低 | 按上方 PyAV 排查步骤确认固定版本和独立导入表现；等待时输出的堆栈用于定位，不能把十几分钟默认视为正常编译 |
| `/docs` 页面空白、JS/CSS 请求超时 | 更新后重启并强制刷新；确认 `/static/docs/...` 和 `/openapi.json` 可访问，静态目录已部署且代理正确转发；新版文档页使用本地资源 |
| GPU 显存不足 | 确认没有重复服务进程或其他 GPU 任务，保持并发为 1；仍不足时可改用 `turbo` / `small` 并重启 |
| 本机可访问，其他机器不可访问 | 检查监听地址、Windows 防火墙、服务器 IP、客户端网段和云安全组 |
| 返回 `503` | 查看是否已就绪；若为容量不足，可按 `Retry-After` 重试、减少请求数或调整排队等待时间 |
| 计划任务显示运行，但接口失败 | 查看 `logs\server.log`；检查运行账户权限、实际 DLL 路径、工作目录以及模型下载是否完成 |

已使用清华源，针对 Windows x64 / CPython 3.13 完成全部依赖的二进制包解析检查；这验证了依赖可解析和 wheel 可用。Windows + RTX 5070 的实际安装、开机启动及真实音频转写仍需在目标服务器验收。
