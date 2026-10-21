# SV4Agent

[English](README.en.md) · [安装与部署](start.md) · [配置示例](.env.example)

SV4Agent 是一个基于 **FastAPI、faster-whisper 和 FFmpeg** 的语音与视频处理服务，为 Agent、脚本和自动化工作流提供 HTTP API。

在一个服务中完成音频转写、GPU 视频转码，以及“图片时间线 + 录音 → MP4”制作。默认面向 **Windows + NVIDIA GPU** 部署，语音模型通过 ModelScope 下载，Python 依赖使用清华 PyPI 镜像。

## 核心功能

- **语音转写**：默认使用 `Systran/faster-whisper-large-v3`，支持自动语言检测、分段时间戳、可选词级时间戳，以及语音翻译为英文。
- **GPU 视频转码**：使用 NVIDIA CUDA 解码和缩放、NVENC 编码，支持 H.264 / H.265、画质设置和分辨率调整。
- **照片视频制作**：上传图片、音频和 `ffconcat` 时间线，按指定停留时间直接生成 MP4。图片统一尺寸、等比例补黑边，最终由 NVENC 编码。
- **服务控制**：支持 Bearer API Key、上传和输出限制、独立的 ASR 与视频队列、视频任务执行超时及临时文件清理。
- **本地接口文档**：Swagger UI 和 ReDoc 的静态资源随项目提供，浏览器无需访问外部 CDN。

## 接口概览

| 方法 | 路径 | 功能 | 成功响应 |
| --- | --- | --- | --- |
| `GET` | `/health` | 检查语音模型是否就绪 | JSON |
| `POST` | `/v1/audio/transcriptions` | 音频转写或翻译为英文 | 文本、语言、时间戳 JSON |
| `POST` | `/v1/video/transcode` | GPU 视频转码与缩放 | MP4 文件 |
| `POST` | `/v1/video/slideshow` | 图片时间线与音频合成视频 | MP4 文件 |
| `GET` | `/docs` | Swagger UI 交互文档 | HTML |
| `GET` | `/redoc` | ReDoc 接口文档 | HTML |

三个处理接口均使用 `multipart/form-data`。设置 `APP_API_KEY` 后，调用需携带 `Authorization: Bearer YOUR_API_KEY`；健康检查与文档页面无需鉴权。

## 使用示例

服务启动后，上传音频进行中文转写：

```powershell
curl.exe --fail http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@D:/audio/recording.m4a" `
  -F "language=zh" `
  -F "word_timestamps=true"
```

上传视频进行 H.264 GPU 转码：

```powershell
curl.exe --fail http://127.0.0.1:8000/v1/video/transcode `
  -F "file=@D:/video/input.mp4" `
  -F "codec=h264" `
  -F "preset=p4" `
  -F "height=1080" `
  -o D:/video/output.mp4
```

上传图片、时间线和录音，直接制作照片视频：

```powershell
curl.exe --fail http://127.0.0.1:8000/v1/video/slideshow `
  -F "images=@D:/photos/first.jpg" `
  -F "images=@D:/photos/second.jpg" `
  -F "timeline=@D:/photos/concat.txt" `
  -F "audio=@D:/audio/recording.m4a" `
  -F "preset=p4" `
  -F "fps=1" `
  -o D:/video/slideshow.mp4
```

上例 `concat.txt` 的内容：

```text
ffconcat version 1.0
file 'first.jpg'
duration 5.000
file 'second.jpg'
duration 8.000
```

`fps=1` 适合本例按整数秒切换的静态照片；接口默认是 24 fps。图片按文件名匹配，服务会补齐末张图片的结束时间戳。成片在音频或时间线较早结束的位置停止。音频默认直接复制，非 MP4 兼容编码可增加 `audio_codec=aac`。完整参数及批量上传脚本见 [部署指南](start.md) 和 [examples](examples)。

## 安装与部署

```powershell
git clone https://github.com/Hold-API/SV4Agent.git
Set-Location SV4Agent
```

默认部署配置为 **标准 CPython 3.13 x64、Windows x64、CUDA 12.8、兼容的 cuDNN 9.x 和 NVIDIA GPU**。视频功能另需安装支持 CUDA / NVENC 的 FFmpeg 可执行程序。

请按照 [start.md](start.md) 完成依赖安装、`.env` 配置、模型下载和服务部署。指南也包含 Windows 开机启动、接口参数及故障排查。首次启动会下载并加载语音模型，完成前 HTTP 服务不会就绪；已有完整本地模型时可配置离线加载。

默认 ASR 和视频功能可以使用同一张 GPU，分别控制并发。视频制作使用同步请求，需保持连接直到生成并下载完成。

## 项目结构

```text
app/                 FastAPI 接口、模型加载与视频处理
app/static/docs/     本地 Swagger UI / ReDoc 资源与许可证
examples/            图片时间线与 PowerShell 上传脚本
tests/               接口、配置、下载与进程管理测试
.env.example         完整环境变量说明
requirements.txt     Python 依赖与清华镜像配置
run.py               服务启动入口
start.md             安装、部署及使用指南
README.en.md         英文项目介绍
```

## 相关项目

本项目基于 [FastAPI](https://github.com/fastapi/fastapi)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[CTranslate2](https://github.com/OpenNMT/CTranslate2)、[ModelScope](https://github.com/modelscope/modelscope) 和 [FFmpeg](https://ffmpeg.org/)。各依赖及随仓库分发的文档资源适用其各自许可证。
