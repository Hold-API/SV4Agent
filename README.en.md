# SV4Agent

[简体中文](README.md) · [Deployment guide (Chinese)](start.md) · [Configuration](.env.example)

SV4Agent is a speech and video processing service built with **FastAPI, faster-whisper, and FFmpeg**. It exposes HTTP APIs for agents, scripts, and automated workflows.

Use one service to transcribe audio, transcode video on an NVIDIA GPU, and turn a photo timeline with a recording into an MP4. The default deployment targets **Windows with an NVIDIA GPU**. Models are downloaded through ModelScope, and Python dependencies use the Tsinghua PyPI mirror.

## Features

- **Speech recognition**: uses `Systran/faster-whisper-large-v3` by default, with automatic language detection, segment timestamps, optional word timestamps, and speech translation into English.
- **GPU video transcoding**: CUDA decoding and scaling, NVENC H.264 / H.265 encoding, configurable quality, and output resolution.
- **Photo slideshows**: upload photos, audio, and an `ffconcat` timeline to create an MP4 directly. Photos are fitted to a consistent canvas with black padding, and NVENC handles final video encoding.
- **Service controls**: optional Bearer authentication, upload and output limits, separate ASR and video queues, video execution timeouts, and temporary file cleanup.
- **Self-hosted API documentation**: Swagger UI and ReDoc assets are bundled with the service, so browsers do not need access to an external CDN.

## API overview

| Method | Path | Purpose | Successful response |
| --- | --- | --- | --- |
| `GET` | `/health` | Check speech model readiness | JSON |
| `POST` | `/v1/audio/transcriptions` | Transcribe speech or translate it into English | JSON with text, language, and timestamps |
| `POST` | `/v1/video/transcode` | GPU video transcoding and resizing | MP4 file |
| `POST` | `/v1/video/slideshow` | Combine a photo timeline with audio | MP4 file |
| `GET` | `/docs` | Interactive Swagger UI | HTML |
| `GET` | `/redoc` | ReDoc API documentation | HTML |

All three processing endpoints accept `multipart/form-data`. When `APP_API_KEY` is configured, send `Authorization: Bearer YOUR_API_KEY`. Health and documentation endpoints do not require authentication.

## Examples

Transcribe a recording with word timestamps:

```powershell
curl.exe --fail http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@D:/audio/recording.m4a" `
  -F "language=en" `
  -F "word_timestamps=true"
```

Transcode a video to H.264 using the GPU:

```powershell
curl.exe --fail http://127.0.0.1:8000/v1/video/transcode `
  -F "file=@D:/video/input.mp4" `
  -F "codec=h264" `
  -F "preset=p4" `
  -F "height=1080" `
  -o D:/video/output.mp4
```

Create a slideshow from two photos, a timeline, and a recording:

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

Contents of `concat.txt`:

```text
ffconcat version 1.0
file 'first.jpg'
duration 5.000
file 'second.jpg'
duration 8.000
```

`fps=1` suits these static photos with transitions on whole seconds; the endpoint defaults to 24 fps. Photos are matched by filename. The service adds the final timestamp needed to preserve the last photo's duration. Output ends when either the audio or the timeline ends, whichever comes first. Audio is copied by default; add `audio_codec=aac` for audio that needs conversion to an MP4-compatible codec. See the [deployment guide](start.md) and [examples](examples) for all parameters and the upload script.

## Installation and deployment

```powershell
git clone https://github.com/Hold-API/SV4Agent.git
Set-Location SV4Agent
```

The default deployment configuration uses **standard CPython 3.13 x64, Windows x64, CUDA 12.8, compatible cuDNN 9.x libraries, and an NVIDIA GPU**. Video processing additionally requires an FFmpeg executable with CUDA / NVENC support.

Follow [start.md](start.md), currently in Chinese, for dependency installation, `.env` configuration, model downloads, Windows startup tasks, API details, and troubleshooting. On first startup, the service downloads and loads the speech model before accepting HTTP requests. Complete local model files can be used for offline startup.

ASR and video processing can share one GPU and have separate concurrency limits. Video creation uses a synchronous request: keep the connection open until processing and download finish.

## Repository layout

```text
app/                 FastAPI routes, model loading, and video processing
app/static/docs/     Bundled Swagger UI / ReDoc assets and licenses
examples/            Photo timeline and PowerShell upload script
tests/               API, configuration, download, and process tests
.env.example         Environment variables and configuration notes
requirements.txt     Python dependencies and Tsinghua mirror settings
run.py               Service entry point
start.md             Installation, deployment, and usage guide
README.md            Chinese project introduction
```

## Related projects

Built with [FastAPI](https://github.com/fastapi/fastapi), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2), [ModelScope](https://github.com/modelscope/modelscope), and [FFmpeg](https://ffmpeg.org/). Dependencies and bundled documentation assets remain subject to their respective licenses.
