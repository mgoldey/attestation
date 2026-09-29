"""Stitch the demo recordings into one narrated film.

    PIPER=/path/to/piper PIPER_VOICE=/path/to/en_US-lessac-medium.onnx \
        python demos/film/build.py [OUT.mp4]

For each scene in scenes.toml: synthesize its narration with Piper (a local,
offline voice -- the text never leaves the machine), render the clip (or a
title card) at 1280x720, stretched by `slow` and held on its last frame until
the narration has finished, then concatenate every scene with its audio. A
.srt of the narration is written beside the film, for playing it muted.

Needs ffmpeg/ffprobe on PATH and Piper (`pip install piper-tts`, then
`python -m piper.download_voices en_US-lessac-medium`). Clips are looked up in
the repo-root demo/ directory the recorders write to, then docs/media/.
Not run by any test: it produces a video, and needs a voice model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEARCH = (REPO / "demo", REPO / "docs" / "media")
SIZE = "1280:720"
BG = "0x282a36"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LEAD_IN = 0.4  # seconds of picture before the voice starts
TAIL = 1.2  # seconds of picture after the voice ends


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def duration(path: Path) -> float:
    out = run(
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)
    )
    return float(out.strip())


def clip_path(name: str) -> Path:
    for base in SEARCH:
        if (base / name).is_file():
            return base / name
    raise SystemExit(f"no clip {name!r} in {', '.join(map(str, SEARCH))} -- record it first")


def speak(text: str, out: Path) -> float:
    piper, voice = os.environ.get("PIPER", "piper"), os.environ["PIPER_VOICE"]
    words = " ".join(text.split())
    subprocess.run(
        [piper, "-m", voice, "-f", str(out)],
        input=words,
        text=True,
        check=True,
        capture_output=True,
    )
    return duration(out)


def card_video(lines: list[str], seconds: float, out: Path) -> None:
    title, *rest = lines
    draw = [
        f"drawtext=fontfile={FONT}:text='{title}':fontcolor=white:fontsize=64:"
        "x=(w-text_w)/2:y=(h-text_h)/2-50"
    ]
    draw += [
        f"drawtext=fontfile={FONT}:text='{line}':fontcolor=0xbbbbbb:fontsize=28:"
        f"x=(w-text_w)/2:y=(h/2)+30+{i * 44}"
        for i, line in enumerate(rest)
    ]
    run(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={BG}:s={SIZE.replace(':', 'x')}:d={seconds:.2f}:r=25",
        "-vf",
        ",".join(draw),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        str(out),
    )


def clip_video(src: Path, slow: float, seconds: float, out: Path) -> None:
    hold = max(0.0, seconds - duration(src) * slow)
    vf = (
        f"setpts={slow}*PTS,scale={SIZE}:force_original_aspect_ratio=decrease,"
        f"pad={SIZE}:(ow-iw)/2:(oh-ih)/2:color={BG},"
        f"tpad=stop_mode=clone:stop_duration={hold:.2f},fps=25"
    )
    run(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-t",
        f"{seconds:.2f}",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        str(out),
    )


def mux(video: Path, voice: Path, seconds: float, out: Path) -> None:
    delay = int(LEAD_IN * 1000)
    run(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(voice),
        "-filter_complex",
        f"[1:a]adelay={delay},apad,atrim=0:{seconds:.2f},aresample=48000[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "1",
        str(out),
    )


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main(out: Path) -> None:
    scenes = tomllib.loads((HERE / "scenes.toml").read_text())["scene"]
    work = Path(tempfile.mkdtemp(prefix="attest-film-"))
    parts, subs, clock = [], [], 0.0
    for i, scene in enumerate(scenes, 1):
        voice = work / f"{i:02d}.wav"
        spoken = speak(scene["say"], voice)
        if "card" in scene:
            seconds = LEAD_IN + spoken + TAIL
            video = work / f"{i:02d}-v.mp4"
            card_video(scene["card"], seconds, video)
        else:
            src = clip_path(scene["clip"])
            seconds = max(duration(src) * scene.get("slow", 1.0) + TAIL, LEAD_IN + spoken + TAIL)
            video = work / f"{i:02d}-v.mp4"
            clip_video(src, scene.get("slow", 1.0), seconds, video)
        part = work / f"{i:02d}.mp4"
        mux(video, voice, seconds, part)
        parts.append(part)
        start = clock + LEAD_IN
        subs.append((start, start + spoken, " ".join(scene["say"].split())))
        clock += seconds
        print(f"  scene {i:2d}  {seconds:5.1f}s  {scene.get('clip') or scene['card'][0]}")

    listing = work / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    run(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(out),
    )
    out.with_suffix(".srt").write_text(
        "".join(
            f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n"
            for n, (a, b, text) in enumerate(subs, 1)
        )
    )
    print(f"wrote {out} ({duration(out):.0f}s) and {out.with_suffix('.srt').name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "demo" / "attestation-film.mp4")
