"""Offline tests for the live ASR terminal renderer.

The renderer is the only component allowed to treat text as final, so these
tests lock down line clearing, Unicode width, non-TTY fallback and the rule
that a provisional partial never becomes a transcript line by itself.
"""

from __future__ import annotations

import io

from companion.asr.terminal import (
    CLEAR_LINE,
    RenderEvent,
    TerminalRenderer,
    display_width,
    fit_line,
)


def make_renderer(*, is_tty: bool = True, width: int = 80):
    out = io.StringIO()
    status = io.StringIO()
    renderer = TerminalRenderer(
        stream=out,
        status_stream=status,
        is_tty=is_tty,
        width=width,
    )
    return renderer, out, status


def test_display_width_handles_cjk_combining_and_ascii() -> None:
    assert display_width("abc") == 3
    assert display_width("你好") == 4
    assert display_width("é") == 1  # combining acute
    assert display_width("") == 0


def test_fit_line_keeps_newest_text_and_fits_width() -> None:
    fitted = fit_line("abcdef", 4)
    assert fitted == "…def"
    assert display_width(fitted) == 4
    assert display_width(fit_line("你好世界", 5)) <= 5
    assert fit_line("short", 80) == "short"
    assert fit_line("text", 0) == ""


def test_partial_updates_use_carriage_return_clear_line() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(1, "你")
    renderer.partial(1, "你好")
    renderer.partial(1, "你好世")
    raw = out.getvalue()
    assert raw.count(CLEAR_LINE) == 3
    assert raw.endswith(CLEAR_LINE + "你好世")
    assert renderer.partial_text == "你好世"
    assert renderer.confirmed == []


def test_shorter_partial_replaces_longer_text() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(1, "你好世界")
    renderer.partial(1, "你好")
    assert out.getvalue().endswith(CLEAR_LINE + "你好")


def test_long_partial_is_truncated_to_terminal_width() -> None:
    renderer, out, _ = make_renderer(width=10)
    renderer.partial(1, "一二三四五六七八九十十一十二")
    last = out.getvalue().split(CLEAR_LINE)[-1]
    assert last.startswith("…")
    assert display_width(last) <= 10


def test_final_is_never_truncated_even_when_wider_than_terminal() -> None:
    renderer, out, _ = make_renderer(width=10)
    text = "一二三四五六七八九十十一十二"
    renderer.partial(1, "一二三四五")
    renderer.final(1, text)
    raw = out.getvalue()
    assert out.getvalue().split(CLEAR_LINE)[-1] == text + "\n"
    assert "…" not in raw.split(CLEAR_LINE)[-1]
    assert renderer.confirmed == [text]


def test_older_partial_cannot_overwrite_a_newer_partial() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(2, "新的内容")
    before = out.getvalue()
    renderer.partial(1, "旧的内容")
    assert renderer.partial_text == "新的内容"
    assert renderer._current_uid == 2
    assert out.getvalue() == before


def test_newer_partial_replaces_older_partial() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(1, "旧")
    renderer.partial(2, "新")
    assert renderer.partial_text == "新"
    assert out.getvalue().endswith(CLEAR_LINE + "新")


def test_final_replaces_partial_and_adds_one_newline() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.final(1, "你好世界")
    raw = out.getvalue()
    assert raw == "你好世界\n"
    assert renderer.confirmed == ["你好世界"]

    renderer.partial(1, "下一句")
    renderer.final(1, "下一句完整")
    raw = out.getvalue()
    assert raw.endswith(CLEAR_LINE + "下一句完整\n")
    assert renderer.confirmed == ["你好世界", "下一句完整"]
    assert renderer.partial_text == ""


def test_empty_final_never_writes_a_blank_transcript_line() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.final(1, "")
    renderer.final(1, "   ")
    assert out.getvalue() == ""
    assert renderer.confirmed == []


def test_non_tty_prints_only_confirmed_lines() -> None:
    renderer, out, status = make_renderer(is_tty=False)
    renderer.partial(1, "临时")
    renderer.partial(1, "临时修正")
    assert out.getvalue() == ""
    renderer.final(1, "最终文本")
    assert out.getvalue() == "最终文本\n"
    renderer.status("正在监听")
    assert "正在监听" in status.getvalue()
    assert "正在监听" not in out.getvalue()


def test_status_and_error_use_the_separate_status_stream() -> None:
    renderer, out, status = make_renderer(width=40)
    renderer.partial(1, "临时")
    before = out.getvalue()
    renderer.status("设备已连接")
    renderer.status("连接失败", is_error=True)
    after = out.getvalue()
    assert "[status] 设备已连接" in status.getvalue()
    assert "[error] 连接失败" in status.getvalue()
    assert "连接失败" not in after
    # The provisional line is cleared and redrawn around the status write.
    assert after.startswith(before)
    assert after.endswith(CLEAR_LINE + "临时")


def test_late_older_final_is_printed_before_newer_partial() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(2, "第二段临时")
    renderer.final(1, "第一段最终")
    raw = out.getvalue()
    assert "第一段最终\n" in raw
    assert raw.endswith("第二段临时")
    assert renderer.partial_text == "第二段临时"
    assert renderer.confirmed == ["第一段最终"]


def test_newer_sentence_final_keeps_existing_partial_when_asked() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(1, "第二句临时")
    renderer.final(1, "第一句最终", replace_partial=False)
    raw = out.getvalue()
    assert "第一句最终\n" in raw
    assert raw.endswith("第二句临时")
    assert renderer.partial_text == "第二句临时"


def test_clear_partial_discards_provisional_text_without_newline() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(1, "临时")
    renderer.clear_partial(1)
    raw = out.getvalue()
    assert raw.endswith(CLEAR_LINE)
    assert "\n" not in raw
    assert renderer.partial_text == ""
    assert renderer.confirmed == []


def test_clear_partial_ignores_a_different_utterance() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(2, "新的")
    before = out.getvalue()
    renderer.clear_partial(1)
    assert out.getvalue() == before
    assert renderer.partial_text == "新的"


def test_render_dispatches_events() -> None:
    renderer, out, status = make_renderer(width=40)
    renderer.render(RenderEvent(kind="partial", utterance_id=1, text="临"))
    renderer.render(RenderEvent(kind="final", utterance_id=1, text="定"))
    renderer.render(RenderEvent(kind="error", text="出错了"))
    assert out.getvalue().endswith(CLEAR_LINE + "定\n")
    assert "[error] 出错了" in status.getvalue()


def test_close_clears_an_unfinished_partial() -> None:
    renderer, out, _ = make_renderer(width=40)
    renderer.partial(1, "未完成")
    renderer.close()
    assert out.getvalue().endswith(CLEAR_LINE)
    renderer.close()  # idempotent
