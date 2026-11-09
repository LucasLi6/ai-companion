"""Command line entry point for the independent real-time ASR demo.

Usage::

    python -m companion.asr.demo --list-devices
    python -m companion.asr.demo
    python -m companion.asr.demo --device "AirPods"
    python -m companion.asr.demo --device 2 --debug

    # optional local speaker identification
    python -m companion.asr.demo --speaker-id
    python -m companion.asr.demo --speaker-id --speaker-debug
    python -m companion.asr.demo --speaker-list
    python -m companion.asr.demo --speaker-delete TEMP-0001
    python -m companion.asr.demo --speaker-clear --yes

This demo is deliberately self-contained: it never imports Ava's Speaker,
Router, TTS, Memory or Orchestrator, and it never writes captured audio to
disk. Speaker identification is an opt-in extra; without ``--speaker-id`` no
voice-print model is loaded and the original behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from companion.asr.config import ASRConfigError, load_asr_config
from companion.asr.microphone import MicrophoneError, list_input_devices
from companion.asr.vad import VADError
from companion.config import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m companion.asr.demo",
        description=(
            "独立实时语音识别 Demo：麦克风 → Silero VAD → Qwen 流式 ASR → 终端。"
            "不接入 Ava 的 Speaker、Router、TTS 或 Memory。"
            "加 --speaker-id 时额外使用本地声纹模型（不保存原始音频）。"
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
    # ------------------------------------------------------ speaker options
    parser.add_argument(
        "--speaker-id",
        action="store_true",
        help="启用本地声纹识别（需要 speaker 可选依赖；请先获得被录音者同意）",
    )
    parser.add_argument(
        "--speaker-db",
        default=None,
        help="声纹 SQLite 路径（默认 ~/.cache/ava-speaker/speakers.sqlite3）",
    )
    parser.add_argument(
        "--speaker-model",
        default=None,
        help="本地声纹模型目录（离线使用；默认由 ModelScope 缓存管理）",
    )
    parser.add_argument(
        "--speaker-device",
        choices=("cpu", "mps", "auto"),
        default=None,
        help="声纹推理设备，默认 cpu；mps 不可用时自动回退 cpu",
    )
    parser.add_argument(
        "--speaker-debug",
        action="store_true",
        help="输出声纹识别的调试日志",
    )
    parser.add_argument(
        "--speaker-list",
        action="store_true",
        help="列出声纹库中的身份后退出（不需要 API Key）",
    )
    parser.add_argument(
        "--speaker-delete",
        default=None,
        metavar="ID",
        help="删除一个声纹身份后退出（例如 TEMP-0001）",
    )
    parser.add_argument(
        "--speaker-clear",
        action="store_true",
        help="清空整个声纹库（必须同时加 --yes）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="与 --speaker-clear 一起使用，确认删除全部声纹",
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


def _open_store(args):
    from companion.speaker_id.config import load_speaker_config
    from companion.speaker_id.store import SpeakerStore, SpeakerStoreError

    config = load_speaker_config(
        db_path=args.speaker_db,
        device=args.speaker_device,
    )
    store = SpeakerStore(
        config.db_path,
        max_embeddings_per_speaker=config.max_embeddings_per_speaker,
        duplicate_sample_similarity=config.duplicate_sample_similarity,
        minimum_enrollment_quality=config.minimum_enrollment_quality,
    )
    try:
        store.initialize()
    except SpeakerStoreError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return None
    return config, store


def _speaker_maintenance(args) -> Optional[int]:
    """Explicit list/delete/clear commands; no ASR or model is started."""
    if not (
        args.speaker_list or args.speaker_delete or args.speaker_clear
    ):
        return None
    opened = _open_store(args)
    if opened is None:
        return 2
    config, store = opened
    try:
        if args.speaker_list:
            speakers = store.list_speakers()
            if not speakers:
                print("声纹库为空。")
            for speaker in speakers:
                print(
                    f"{speaker['speaker_id']}  {speaker['status']}  "
                    f"样本 {speaker['samples']}  观察 "
                    f"{speaker['observation_count']}  "
                    f"更新 {speaker['updated_at']:.0f}"
                )
        if args.speaker_delete:
            deleted = store.delete_speaker(args.speaker_delete)
            print(
                f"已删除 {args.speaker_delete}。"
                if deleted
                else f"未找到 {args.speaker_delete}。"
            )
        if args.speaker_clear:
            if not args.yes:
                print(
                    "拒绝清空：请同时加 --yes 明确确认。",
                    file=sys.stderr,
                )
                return 2
            removed = store.clear()
            print(f"已清空声纹库（删除 {removed} 个身份）。")
        print(f"声纹库: {config.db_path}")
        return 0
    finally:
        store.close()


def _setup_speaker(args):
    """Build the deferred observer; the model loads in the background.

    Returns ``None`` only on a configuration error. A model/store failure is
    reported by the observer and does not stop the ASR demo.
    """
    from companion.speaker_id.config import (
        SpeakerConfigError,
        load_speaker_config,
    )
    from companion.speaker_id.runtime import DeferredSpeakerObserver

    try:
        config = load_speaker_config(
            db_path=args.speaker_db,
            model_path=args.speaker_model,
            device=args.speaker_device,
            debug=args.speaker_debug,
        )
    except SpeakerConfigError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return None

    print(
        "已启用本地声纹识别：说话音频只用于提取声纹向量并保存在 "
        f"{config.db_path}（不保存原始音频）。请确保被录音者知情并同意。",
        file=sys.stderr,
    )
    print(
        "声纹模型将在后台加载；加载期间的语音仍会正常识别，"
        "但不会附带说话人身份。",
        file=sys.stderr,
    )
    return DeferredSpeakerObserver(config)


def _print_summary(manager, *, debug: bool, observer=None) -> None:
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
    if observer is not None:
        timings = observer.timings
        if timings["model_load_ms"] is not None:
            print(
                "声纹启动：模型加载 "
                f"{timings['model_load_ms']:.0f} ms，"
                f"预热 {timings['warmup_ms'] or 0.0:.0f} ms。",
                file=sys.stderr,
            )
        if timings["error"]:
            print(
                f"声纹功能不可用：{timings['error']}",
                file=sys.stderr,
            )
        engine = observer.engine
        if engine is not None:
            speaker = engine.stats()
            identity = speaker["first_identity_latencies_ms"]
            if identity:
                average = sum(identity) / len(identity)
                print(
                    "说话开始 → 首次身份结果："
                    f"平均 {average:.0f} ms（{len(identity)} 段）。",
                    file=sys.stderr,
                )
            print(
                f"声纹：分析 {speaker['analyses']} 次，"
                f"错误 {speaker['errors']} 次，"
                f"丢弃唤醒 {speaker['dropped_wakeups']} 次。",
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

    maintenance = _speaker_maintenance(args)
    if maintenance is not None:
        return maintenance

    try:
        config = load_asr_config(
            device=args.device,
            debug=args.debug,
            model=args.model or None,
        )
    except ASRConfigError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    # `--debug` surfaces structured ``asr_lifecycle`` lines (session state,
    # close origin, primary/secondary error) on stderr; without it only
    # concise user-facing errors are shown and the file log stays quiet.
    configure_logging(args.debug)
    if args.speaker_debug:
        logging.getLogger("companion.speaker").setLevel(logging.DEBUG)

    # Imported here so --list-devices stays usable without dashscope/numpy.
    from companion.asr.manager import ASRManager

    observer = None
    if args.speaker_id:
        observer = _setup_speaker(args)
        if observer is None:
            return 3

    manager = ASRManager(config, speaker_observer=observer)
    if observer is not None:
        observer.set_on_result(manager.post_speaker_result)
        observer.start_loading()

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
        if observer is not None:
            observer.shutdown()
        _print_summary(manager, debug=args.debug, observer=observer)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
