"""
tests/test_auth.py

Phase 10 login logic (datatrust/web/auth.py), without Django.
"""
from __future__ import annotations

import pytest

from datatrust.web.auth import (
    LOCKOUT_SECONDS,
    MAX_AGE,
    LoginThrottle,
    check_password,
    check_token,
    make_token,
    safe_next,
)

SECRET, PW, NOW = "django-secret-key", "s3cret pässword", 1_800_000_000


def test_token_round_trip_and_expiry():
    t = make_token(SECRET, PW, now=NOW)
    assert check_token(t, SECRET, PW, now=NOW)
    assert check_token(t, SECRET, PW, now=NOW + MAX_AGE - 1)
    assert not check_token(t, SECRET, PW, now=NOW + MAX_AGE + 1)


def test_token_is_bound_to_secret_and_password():
    t = make_token(SECRET, PW, now=NOW)
    assert not check_token(t, "other-secret", PW, now=NOW)
    assert not check_token(t, SECRET, "new password", now=NOW)      # changing the password logs everyone out
    assert PW not in t and "s3cret" not in t                          # nothing secret in the cookie


@pytest.mark.parametrize("bad", [None, "", "nodot", "abc.def", f"{NOW + 99}.{'0' * 64}", "-5.x", "1e9.x"])
def test_malformed_or_forged_tokens_fail(bad):
    assert not check_token(bad, SECRET, PW, now=NOW)


def test_extending_the_expiry_breaks_the_signature():
    expiry, sig = make_token(SECRET, PW, now=NOW).split(".")
    assert not check_token(f"{int(expiry) + 3600}.{sig}", SECRET, PW, now=NOW)


def test_check_password():
    assert check_password(PW, PW)
    assert not check_password("s3cret", PW) and not check_password(None, PW) and not check_password("", PW)


@pytest.mark.parametrize("url, expected", [
    ("/runs/3/", "/runs/3/"),
    ("/datasets/a%20b.csv/?x=1", "/datasets/a%20b.csv/?x=1"),
    (None, "/"), ("", "/"),
    ("https://evil.example/", "/"), ("//evil.example/", "/"), ("/\\evil.example", "/"),
    ("javascript:alert(1)", "/"), ("/ok\r\nSet-Cookie: x=1", "/"),
])
def test_safe_next_blocks_open_redirects(url, expected):
    assert safe_next(url) == expected


def test_throttle_locks_after_five_failures_then_releases():
    th = LoginThrottle()
    for i in range(4):
        th.failed("1.2.3.4", now=NOW + i)
    assert th.locked_for("1.2.3.4", now=NOW + 4) == 0
    th.failed("1.2.3.4", now=NOW + 4)
    assert 0 < th.locked_for("1.2.3.4", now=NOW + 5) <= LOCKOUT_SECONDS
    assert th.locked_for("5.6.7.8", now=NOW + 5) == 0                 # other clients are not affected
    assert th.locked_for("1.2.3.4", now=NOW + 4 + LOCKOUT_SECONDS + 1) == 0


def test_throttle_success_resets():
    th = LoginThrottle()
    for i in range(4):
        th.failed("c", now=NOW + i)
    th.succeeded("c")
    th.failed("c", now=NOW + 10)
    assert th.locked_for("c", now=NOW + 10) == 0
