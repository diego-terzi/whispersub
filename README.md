# whispersub

CLI tool to generate `.srt` subtitle files from video using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on an NVIDIA GPU (with automatic CPU fallback).

## Features

- GPU-accelerated transcription via CUDA (float16)
- Automatic fallback to CPU (int8) when CUDA is unavailable or VRAM is exhausted
- Live progress bar during transcription
- Outputs `.srt` file alongside the input video

## Requirements

- Windows 10/11 (native — no WSL required)
- Python 3.10+
- NVIDIA GPU with CUDA 12.1 drivers (optional — CPU fallback available)
- [FFmpeg](https://ffmpeg.org/) on `PATH`

---

## Installation

### 1. Install Python

Download Python 3.10 or later from [python.org](https://www.python.org/downloads/windows/).  
During installation, check **"Add Python to PATH"**.

Verify:
```cmd
python --version
```

### 2. Install FFmpeg

1. Download the latest FFmpeg build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (choose `ffmpeg-release-essentials.zip`).
2. Extract the archive (e.g. to `C:\ffmpeg`).
3. Add `C:\ffmpeg\bin` to your system `PATH`:
   - Search for **"Environment Variables"** in the Start menu.
   - Under **System variables**, edit `Path` and add `C:\ffmpeg\bin`.
4. Open a new terminal and verify:
   ```cmd
   ffmpeg -version
   ```

### 3. Install NVIDIA CUDA drivers (GPU only)

Install the latest [NVIDIA Game Ready or Studio driver](https://www.nvidia.com/Download/index.aspx) for your GPU.  
CUDA 12.1 runtime is bundled with PyTorch — no separate CUDA Toolkit installation is required.

### 4. Clone the repository

```cmd
git clone https://github.com/diego-terzi/whispersub.git
cd whispersub
```

### 5. Create and activate a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

Your prompt should now show `(venv)`.

### 6. Install dependencies

```cmd
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA 12.1 support will be downloaded (~2 GB). The faster-whisper model (`small`, ~460 MB) is downloaded automatically on first run.

---

## Usage

```cmd
python main.py path\to\video.mp4
```

The `.srt` file is saved in the same folder as the input video.

**Example:**

```cmd
python main.py C:\Videos\lecture.mp4
# Output: C:\Videos\lecture.srt
```

### Output

```
[info] Input:  C:\Videos\lecture.mp4
[info] Output: C:\Videos\lecture.srt
[info] Loading model (device=cuda, compute_type=float16)...
[info] Duration: 3642.3s | Language: en (99%)
Transcribing: 100%|████████████████████| 3642/3642 [04:12<00:00, 14.4s/s]
[done] 847 segment(s) written to 'C:\Videos\lecture.srt'
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA not available` | Script falls back to CPU automatically. Check that NVIDIA drivers are up to date. |
| `VRAM out of memory` | Script retries on CPU automatically. Alternatively close other GPU workloads. |
| `FileNotFoundError: ffmpeg` | Make sure FFmpeg is installed and `C:\ffmpeg\bin` is on your `PATH`. |
| Slow CPU transcription | Expected without a GPU. For the `small` model, ~20× real-time on CPU. |

---

## License

MIT
