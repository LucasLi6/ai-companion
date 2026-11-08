"""Command line entry point for the independent real-time ASR demo.

Usage::

    python -m companion.asr.demo --list-devices
    python -m companion.asr.demo
    python -m companion.asr.demo --device "AirPods"
    python -m companion.asr.demo --device 2 --debug

This demo is deliberately self-contained: it never imports Ava's Speaker,
Router, TTS, Memory or Orchestrator, and it never writes captured audio to
disk.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from companion.asr.config import ASRConfigError, load_asr_config
from companion.asr.microphone import MicrophoneError, list_input_devices
from companion.asr.vad import VADError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m companion.asr.demo",
        description=(
            "独立实时语音识别 Demo：麦克风 → Silero VAD → Qwen 流式 ASR → 终端。"
            "不接入 Ava 的 Speaker、Router、TTS 或 Memory。"
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出音频输入设备后退出（不需要 API Key，也不会连接网络）",
    )
    parser.add_argument(
        "--device",
        default="",
        help="输入设备索引或名称片段，例如 2 或 AirPods；默认使用系统输入",
    )
    parser.add_argument(
        "--model",
        default="",
        help="覆盖 ASR 模型（默认读取 AVA_ASR_MODEL）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示设备接口、分段状态和 VAD→最终结果延迟",
    )
    return parser


def _show_devices() -> int:
    try:
        devices = list_input_devices()
    except MicrophoneError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    if not devices:
        print("没有找到音频输入设备。", file=sys.stderr)
        return 1
    for device in devices:
        flags = []
        if device.is_default_input:
            flags.append("系统默认输入")
        if device.max_output_channels > 0:
            flags.append("可输出")
        if device.max_input_channels > 1:
            flags.append(f"{device.max_input_channels} 声道输入")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"{device.index:>2}  {device.name}  "
            f"默认采样率 {device.default_sample_rate} Hz  "
            f"输入 {device.max_input_channels} / 输出 "
            f"{device.max_output_channels}{suffix}"
        )
    return 0


def _print_summary(manager, *, debug: bool) -> None:
    stats = manager.stats
    print("", file=sys.stderr)
    print(
        f"会话结束：共 {stats.utterances} 段语音，"
        f"{stats.finalized} 段已确认，"
        f"{stats.aborted} 段中止，{stats.errors} 次错误。",
        file=sys.stderr,
    )
    latencies = stats.final_latencies_ms
    if latencies:
        average = sum(latencies) / len(latencies)
        print(
            "VAD 结束 → 最终结果："
            f"平均 {average:.0f} ms，最小 {min(latencies):.0f} ms，"
            f"最大 {max(latencies):.0f} ms（{len(latencies)} 段）。",
            file=sys.stderr,
        )
    if debug:
        print(
            f"调试：输入丢弃块 {manager.dropped_audio_chunks}，"
            f"控制事件丢弃 {stats.dropped_control_events}，"
            f"延迟会话 {stats.deferred}。",
            file=sys.stderr,
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_devices:
        return _show_devices()

    try:
        config = load_asr_config(
            device=args.device,
            debug=args.debug,
            model=args.model or None,
        )
    except ASRConfigError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    # Imported here so --list-devices stays usable without dashscope/numpy.
    from companion.asr.manager import ASRManager

    manager = ASRManager(config)
    interrupted = False
    try:
        manager.run()
    except KeyboardInterrupt:
        interrupted = True
        manager.stop()
        print("\n已停止。", file=sys.stderr)
    except (MicrophoneError, VADError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level safe failure
        print(
            f"错误: 实时识别异常退出（{type(exc).__name__}）。",
            file=sys.stderr,
        )
        return 1
    finally:
        manager.shutdown()
        _print_summary(manager, debug=args.debug)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
