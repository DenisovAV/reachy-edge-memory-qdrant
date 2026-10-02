"""The shared HTTP reason-phrase sanitizer.

http.server writes the reason phrase verbatim into the status line, so a
dynamic error message must be made safe: no non-ASCII (would crash the latin-1
encoder) and no control chars (CR/LF could split the response).
"""

from demo.http_util import ascii_reason


def test_replaces_each_non_ascii_codepoint_with_one_question_mark():
    assert ascii_reason("é☃") == "??"      # é, snowman
    assert ascii_reason("ab☃cd") == "ab?cd"


def test_neutralizes_crlf_so_it_cannot_split_the_response():
    out = ascii_reason("bad\r\nX-Injected: 1")
    assert "\r" not in out and "\n" not in out


def test_neutralizes_all_control_chars():
    assert ascii_reason("a\tb\x00c\x7f") == "a b c "


def test_result_is_always_pure_printable_ascii():
    out = ascii_reason("\U0001f525\r\n\x07 é" * 50)
    assert all(0x20 <= ord(c) < 0x7f for c in out)


def test_caps_length():
    assert len(ascii_reason("x" * 500, limit=200)) == 200


def test_a_reason_never_names_the_home_directory():
    # An exception about a file carries its path, and the path the user's
    # name — to anyone on the network the laptop's services answer.
    from pathlib import Path

    from demo.http_util import ascii_reason, public

    text = f"FileNotFoundError: {Path.home()}/Work/repo/assets/m.tflite is missing"
    assert public(text) == "FileNotFoundError: ~/Work/repo/assets/m.tflite is missing"
    assert str(Path.home()) not in ascii_reason(text)
    assert "~/Work/repo/assets/m.tflite" in ascii_reason(text)
