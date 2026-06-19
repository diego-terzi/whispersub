#!/usr/bin/env python3
"""CLI tool to generate .srt subtitle files from video using faster-whisper."""

import json
import sys
import argparse
from pathlib import Path

_PROGRESS_FILE = ".whispersub_progress.json"


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


def _load_progress(folder: Path) -> dict[str, list | dict]:
    """Load batch progress from the progress file in folder."""
    progress_path = folder / _PROGRESS_FILE
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": {}}


def _save_progress(folder: Path, progress: dict[str, list | dict]) -> None:
    """Persist batch progress to the progress file in folder."""
    progress_path = folder / _PROGRESS_FILE
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def transcribe_folder(folder_path: Path) -> None:
    """Transcribe all MP4 files in folder_path, skipping already-completed ones."""
    mp4_files = sorted(folder_path.glob("*.mp4"))
    if not mp4_files:
        print(f"[warn] No MP4 files found in '{folder_path}'")
        return

    progress = _load_progress(folder_path)
    completed: list[str] = progress["completed"]
    failed: dict[str, str] = progress["failed"]

    total = len(mp4_files)
    print(f"[info] Found {total} MP4 file(s) in '{folder_path}'")
    print(f"[info] Already completed: {len(completed)}/{total}")
    if failed:
        print(f"[info] Previously failed (will retry): {list(failed.keys())}")
    print()

    for idx, video_path in enumerate(mp4_files, start=1):
        filename = video_path.name

        output_path = video_path.with_suffix(".srt")

        if filename in completed or output_path.exists():
            if filename not in completed:
                completed.append(filename)
                _save_progress(folder_path, {"completed": completed, "failed": failed})
            print(f"[skip] ({idx}/{total}) {filename} — already transcribed")
            continue

        print(f"[info] ({idx}/{total}) Processing: {filename}")

        try:
            transcribe(video_path, output_path)
            completed.append(filename)
            failed.pop(filename, None)  # clear previous failure if retried
        except Exception as exc:
            failed[filename] = str(exc)
            print(f"[error] Failed '{filename}': {exc}", file=sys.stderr)
            print("[info] Continuing with next file...")
        finally:
            _save_progress(folder_path, {"completed": completed, "failed": failed})

        print()

    success = len(completed)
    print(f"[done] Batch complete: {success}/{total} succeeded, {len(failed)} failed")
    if failed:
        print("[warn] Failed files:")
        for name, err in failed.items():
            print(f"       • {name}: {err}")


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
        description="Generate .srt subtitle files from video using faster-whisper. "
                    "Pass a video file for single transcription, or a folder to batch-process all MP4s inside it."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a video file OR a folder containing MP4 files",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()

    if not input_path.exists():
        print(f"[error] Path not found: '{input_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        if input_path.is_dir():
            transcribe_folder(input_path)
        else:
            output_path = input_path.with_suffix(".srt")
            print(f"[info] Input:  {input_path}")
            print(f"[info] Output: {output_path}")
            if output_path.exists():
                print(f"[skip] Subtitle file already exists: '{output_path}'")
            else:
                transcribe(input_path, output_path)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
