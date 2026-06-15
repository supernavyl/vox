"""Command-line entry point for the vox voice engine.

vox talk [--llm SPEC] [--voice NAME]   start a voice conversation
vox say "text" [--voice NAME]          synthesize and play one phrase
vox devices                            list audio input/output devices
vox providers                          list supported llm-spec providers
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from vox.config import VoxConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vox", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    talk = sub.add_parser("talk", help="start a voice conversation")
    talk.add_argument("--llm", help="llm spec, e.g. ollama:llama3.2 or anthropic:claude-opus-4-8")
    talk.add_argument("--voice", help="Kokoro voice id (default af_nicole)")
    talk.add_argument("--system", help="override the system prompt")

    say = sub.add_parser("say", help="synthesize and play one phrase")
    say.add_argument("text", help="text to speak")
    say.add_argument("--voice", help="Kokoro voice id (default af_nicole)")

    sub.add_parser("devices", help="list audio devices")
    sub.add_parser("providers", help="list supported llm providers")
    return parser


def _config_from_args(args: argparse.Namespace) -> VoxConfig:
    config = VoxConfig.from_env()
    if getattr(args, "llm", None):
        config.llm = args.llm
    if getattr(args, "voice", None):
        config.tts_voice = args.voice
    if getattr(args, "system", None):
        config.system_prompt = args.system
    return config


async def _talk(config: VoxConfig) -> None:
    from vox.conversation import VoiceEngine

    engine = VoiceEngine(config)
    print(f"vox — wrapping {config.llm}. Speak; Ctrl-C to quit.")
    await engine.run()


async def _say(config: VoxConfig, text: str) -> None:
    from vox.audio.speaker import Speaker
    from vox.providers.tts.kokoro import KokoroTTS

    tts = KokoroTTS(voice=config.tts_voice, lang=config.tts_lang, device=config.tts_device)
    speaker = Speaker(device=config.output_device)
    try:
        audio = await tts.synthesize(text)
        await speaker.play(audio, tts.sample_rate)
    finally:
        await speaker.close()
        await tts.aclose()


def _devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def _providers() -> None:
    from vox.providers.llm.spec import available_providers

    print("llm-spec providers:")
    for name in available_providers():
        print(f"  {name}")
    print("\nspec form: provider:model[@base_url]")
    print(
        "examples: ollama:qwen3:8b · anthropic:claude-opus-4-8 · openai-compat:m@http://host:8080/v1"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch the chosen subcommand."""
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "talk":
            asyncio.run(_talk(_config_from_args(args)))
        elif args.command == "say":
            asyncio.run(_say(_config_from_args(args), args.text))
        elif args.command == "devices":
            _devices()
        elif args.command == "providers":
            _providers()
    except KeyboardInterrupt:
        print("\nbye.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
