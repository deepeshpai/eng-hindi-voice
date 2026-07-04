"""
Entry point for the English → Hindi real-time voice translation pipeline.

All models run locally — no API keys required.
Models are downloaded from HuggingFace on first run.

Usage
-----
    python main.py                                     # default (NLLB + piper)
    python main.py --text-only                         # print translations, no TTS
    python main.py --translation-backend opus          # faster, smaller model
    python main.py --whisper-model small               # better ASR accuracy
    python main.py --tone formal                       # force formal Hindi (आप-form)
    python main.py --piper-voice hi_IN-sreenivas-medium
    python main.py --list-devices                      # list audio input devices
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def _print_banner() -> None:
    banner = Text.assemble(
        ("Gnani", "bold white"),
        ("  ·  ", "dim"),
        ("Real-Time Voice Translation", "bold"),
        ("  ·  ", "dim"),
        ("English  →  हिन्दी", "bold cyan"),
        ("  ·  ", "dim"),
        ("100% local, no API keys", "dim green"),
    )
    console.print(Panel(banner, border_style="dim", padding=(0, 2)))
    console.print()


def _list_devices() -> None:
    import sounddevice as sd  # type: ignore

    console.print("[bold]Available audio input devices:[/bold]")
    for i, device in enumerate(sd.query_devices()):  # type: ignore
        if device["max_input_channels"] > 0:
            console.print(
                f"  [{i:>2}]  {device['name']}  "
                f"[dim]({device['max_input_channels']} ch, "
                f"{device['default_samplerate']:.0f} Hz)[/dim]"
            )
    console.print()
    console.print("Set [bold]INPUT_DEVICE=<index>[/bold] in your .env to select one.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gnani",
        description="Real-time English → Hindi voice translation (fully local, open-source)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print Hindi translations without playing audio (testing mode).",
    )
    parser.add_argument(
        "--whisper-model",
        metavar="MODEL",
        default=None,
        help=(
            "Whisper ASR model (default: base.en).\n"
            "  tiny.en   ~39 MB, fastest\n"
            "  base.en   ~74 MB  ← default\n"
            "  small.en  ~244 MB, better accuracy\n"
            "  medium.en ~769 MB, best accuracy\n"
            "  large-v3  ~1.5 GB, multilingual"
        ),
    )
    parser.add_argument(
        "--translation-backend",
        choices=["nllb", "opus"],
        default=None,
        help=(
            "Local translation model (default: nllb).\n"
            "  nllb  facebook/nllb-200-distilled-600M  ~1.2 GB, best quality\n"
            "  opus  Helsinki-NLP/opus-mt-en-hi         ~300 MB, fastest"
        ),
    )
    parser.add_argument(
        "--tone",
        choices=["formal", "casual", "auto"],
        default=None,
        help="Hindi register (default: auto — inferred per sentence).",
    )
    parser.add_argument(
        "--piper-voice",
        metavar="VOICE",
        default=None,
        help=(
            "Piper Hindi voice model (default: hi_IN-rohan-medium).\n"
            "  hi_IN-rohan-medium      male voice,   ~63 MB  ← default\n"
            "  hi_IN-pratham-medium    male voice,   ~63 MB\n"
            "  hi_IN-priyamvada-medium female voice, ~63 MB"
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphone devices and exit.",
    )
    parser.add_argument(
        "--no-latency",
        action="store_true",
        help="Suppress per-utterance latency reports.",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    from config import config
    from pipeline import VoiceTranslationPipeline

    if args.whisper_model:
        config.asr.model = args.whisper_model
    if args.translation_backend:
        config.translation.backend = args.translation_backend  # type: ignore[assignment]
    if args.tone:
        config.translation.tone = args.tone  # type: ignore[assignment]
    if args.piper_voice:
        config.tts.voice = args.piper_voice
    if args.text_only:
        config.text_only = True
    if args.no_latency:
        config.show_latency = False

    console.print(
        f"[dim]ASR:[/dim] faster-whisper [bold]{config.asr.model}[/bold]  "
        f"[dim]Translation:[/dim] [bold]{config.translation.backend}[/bold]  "
        f"[dim]TTS:[/dim] piper [bold]{config.tts.voice}[/bold]"
    )
    console.print()

    async with VoiceTranslationPipeline(config) as pipeline:
        console.print(
            "[bold green]Listening…[/bold green]  "
            "[dim]Speak in English. Press Ctrl-C to stop.[/dim]"
        )
        await pipeline.run()


def main() -> None:
    args = _parse_args()

    if args.list_devices:
        _list_devices()
        sys.exit(0)

    _print_banner()

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
