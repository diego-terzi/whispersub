#!/usr/bin/env python3
"""CLI tool to generate .srt subtitle files from video using faster-whisper."""

import sys
import argparse
from pathlib import Path


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp format HH:MM:SS,mmm."""
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(segments: list, output_path: Path) -> int:
    """Write transcription segments to an SRT file. Returns number of entries."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(
                f"{i}\n"
                f"{_seconds_to_srt_time(seg.start)} --> {_seconds_to_srt_time(seg.end)}\n"
                f"{seg.text.strip()}\n\n"
            )
            count = i
    return count


def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) based on CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
        print("[warn] CUDA not available — using CPU (int8).")
    except ImportError:
        print("[warn] PyTorch not found — using CPU (int8).")
    return "cpu", "int8"


def _free_gpu_memory() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_transcription(video_path: Path, device: str, compute_type: str, output_path: Path) -> int:
    """Load model, transcribe video, write SRT. Returns segment count."""
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    print(f"[info] Loading model (device={device}, compute_type={compute_type})...")
    model = WhisperModel("small", device=device, compute_type=compute_type)

    segments_gen, info = model.transcribe(str(video_path), language="en", beam_size=5)
    print(
        f"[info] Duration: {info.duration:.1f}s | "
        f"Language: {info.language} ({info.language_probability:.0%})"
    )

    segments: list = []
    with tqdm(total=round(info.duration), unit="s", desc="Transcribing", dynamic_ncols=True) as pbar:
        prev_end = 0.0
        for seg in segments_gen:
            segments.append(seg)
            pbar.update(max(0, round(seg.end - prev_end)))
            prev_end = seg.end

    return _write_srt(segments, output_path)


def transcribe(video_path: Path, output_path: Path) -> None:
    """Transcribe video, auto-falling back to CPU on CUDA failure."""
    device, compute_type = _detect_device()

    try:
        count = _run_transcription(video_path, device, compute_type, output_path)
        print(f"[done] {count} segment(s) written to '{output_path}'")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device == "cuda":
            print("[warn] VRAM out of memory — retrying on CPU (int8)...")
            _free_gpu_memory()
            count = _run_transcription(video_path, "cpu", "int8", output_path)
            print(f"[done] {count} segment(s) written to '{output_path}'")
        else:
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate .srt subtitle files from video using faster-whisper."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file")
    args = parser.parse_args()

    video_path = args.video.resolve()

    if not video_path.exists():
        print(f"[error] File not found: '{video_path}'", file=sys.stderr)
        sys.exit(1)

    output_path = video_path.with_suffix(".srt")

    print(f"[info] Input:  {video_path}")
    print(f"[info] Output: {output_path}")

    try:
        transcribe(video_path, output_path)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
