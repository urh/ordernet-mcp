"""Tests for bot.ordernet.

Two flavors of test:

1. Behavioral: stub out the network and verify auth, parsing, and the
   credentials loader.
2. **Security invariants** (the important ones): the client cannot
   issue any HTTP write besides the single auth POST, and cannot GET
   any path that isn't on the explicit allowlist. These are the tests
   that earn the right to call this module 'read-only'.

Adding a method that breaks these invariants must fail loudly here.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ordernet_mcp import client as ordernet
from ordernet_mcp.client import (
    _ALLOWED_GET_PATHS,
    _AUTH_PATH,
    Account,
    OrdernetClient,  # back-compat alias
    OrdernetError,
    OrdernetReadOnlyClient,
    OrdernetSecurityError,
    SecuritiesTotals,
    load_credentials,
    pull_all_balances,
)


def _mock_response(status: int, payload):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
    return r


# ============================== security invariants ===========================
# The whole point of this module: our code path can only ever read.


def test_back_compat_alias_resolves_to_readonly_client():
    """``OrdernetClient`` is kept as an alias only - the underlying class
    is the read-only one. Anyone importing the old name still gets the
    safe surface."""
    assert OrdernetClient is OrdernetReadOnlyClient


def test_only_allowed_get_paths_can_be_called(monkeypatch):
    """Calling _get with an off-list path must raise ``OrdernetSecurityError``
    BEFORE any network call. Fakes ``self._session.get`` so we can
    detect (and fail) any sneaky network call as well."""
    c = OrdernetReadOnlyClient("meitav")
    c._token = "x"
    network_calls: list[str] = []

    def fake_get(url, *args, **kwargs):
        network_calls.append(url)
        return _mock_response(200, {})

    monkeypatch.setattr(c._session, "get", fake_get)

    forbidden = [
        "Order/Place",
        "Trade/Execute",
        "Account/Cancel",
        "Order/Submit",
        "Auth/Authenticate",  # POST endpoint - never legal as a GET either
        "DataProvider/GetStaticData/../Order/Place",
    ]
    for path in forbidden:
        with pytest.raises(OrdernetSecurityError):
            c._get(path)
    assert network_calls == [], (
        f"client made network calls when it shouldn't have: {network_calls}"
    )


def test_allowlist_contains_only_read_paths():
    """Belt-and-braces: the allowlist itself shouldn't ever contain any
    obvious-write path. If someone accidentally adds 'Order/Place' to
    the allowlist this test catches it."""
    forbidden_substrings = ["Order", "Trade", "Cancel", "Submit", "Place"]
    for path in _ALLOWED_GET_PATHS:
        for f in forbidden_substrings:
            assert f.lower() not in path.lower(), (
                f"path {path!r} on read-only allowlist contains write-like "
                f"substring {f!r}"
            )
    # The auth POST path also must not be on the GET allowlist.
    assert _AUTH_PATH not in _ALLOWED_GET_PATHS


def test_no_post_helper_exists():
    """We deliberately do not have a generalized ``_post`` / ``_put`` /
    ``_delete`` / ``_patch`` / ``place_order`` / ``cancel_order`` /
    ``deposit`` / ``withdraw``. If you ever add one this test fails -
    re-think before doing that."""
    members = {n for n, _ in inspect.getmembers(OrdernetReadOnlyClient)}
    forbidden = {
        "_post", "_put", "_delete", "_patch",
        "place_order", "submit_order", "cancel_order",
        "place_trade", "execute_trade",
        "deposit", "withdraw", "transfer",
    }
    leaked = members & forbidden
    assert not leaked, f"OrdernetReadOnlyClient grew write-capable methods: {leaked}"


def test_module_source_has_no_post_calls_outside_authenticate():
    """Last line of defense: a regex sweep of the module source. The ONLY
    POST in the file should be inside ``authenticate``. Any other
    ``self._session.post(...)`` (or .put/.patch/.delete) is a bug."""
    src = Path(ordernet.__file__).read_text(encoding="utf-8")
    # Find every "<session>.<verb>(" call.
    write_verbs = ("post", "put", "patch", "delete")
    pattern = re.compile(r"_session\.(post|put|patch|delete)\s*\(", re.IGNORECASE)
    matches = [
        (line_no, line.strip())
        for line_no, line in enumerate(src.splitlines(), 1)
        if pattern.search(line)
    ]
    # We expect exactly one match - the post inside authenticate().
    assert len(matches) == 1, (
        f"expected exactly one HTTP write call (the auth POST), "
        f"found {len(matches)}: {matches}"
    )
    # ...and that one match must be a `.post(` (not put/patch/delete).
    assert ".post(" in matches[0][1].lower(), (
        f"the single write call is not a POST: {matches[0]}"
    )


# ============================== behavioral tests =============================


def test_authenticate_sets_bearer():
    c = OrdernetReadOnlyClient("meitav")
    with patch.object(c._session, "post", return_value=_mock_response(200, {"l": "abc.def.ghi"})) as post:
        c.authenticate("user", "pass")
    assert c._token == "abc.def.ghi"
    assert c._auth_headers == {"Authorization": "Bearer abc.def.ghi"}
    post.assert_called_once()
    call_url = post.call_args.args[0]
    assert call_url == "https://sparkmeitav.ordernet.co.il/api/Auth/Authenticate"


def test_authenticate_failure_raises():
    c = OrdernetReadOnlyClient("meitav")
    with patch.object(c._session, "post", return_value=_mock_response(401, {"error": "bad creds"})):
        with pytest.raises(OrdernetError, match="auth failed"):
            c.authenticate("user", "pass")


def test_authenticate_missing_token_raises():
    c = OrdernetReadOnlyClient("meitav")
    with patch.object(c._session, "post", return_value=_mock_response(200, {"not_l": "x"})):
        with pytest.raises(OrdernetError, match="missing 'l'"):
            c.authenticate("u", "p")


def test_unsupported_broker_rejected():
    with pytest.raises(ValueError, match="unsupported broker"):
        OrdernetReadOnlyClient("fakebroker")


def test_get_accounts_parses_static_data_shape():
    """The Spark `GetStaticData` payload is a list of blocks; the ACC block
    has an `a` field with one entry per account, and each entry's `_k` is
    the account key while `a.b` is the number and `a.e` is the display name."""
    payload = [
        {"b": "OTHER", "a": [{"_k": "skip", "a": {"b": "0", "e": "noise"}}]},
        {
            "b": "ACC",
            "a": [
                {"_k": "ACC_001-1111111111", "a": {"b": 1111111111, "e": "primary"}},
                {"_k": "ACC_001-3333333333", "a": {"b": 3333333333, "e": "joint"}},
                {"_k": "ACC_001-2222222222", "a": {"b": 2222222222, "e": "secondary"}},
            ],
        },
    ]
    c = OrdernetReadOnlyClient("meitav")
    c._token = "x"
    with patch.object(c._session, "get", return_value=_mock_response(200, payload)):
        accs = c.get_accounts()
    assert [a.number for a in accs] == ["1111111111", "3333333333", "2222222222"]
    assert accs[0].key == "ACC_001-1111111111"
    assert accs[2].name == "secondary"


def test_get_accounts_no_acc_block_raises():
    c = OrdernetReadOnlyClient("meitav")
    c._token = "x"
    with patch.object(c._session, "get", return_value=_mock_response(200, [{"b": "OTHER", "a": []}])):
        with pytest.raises(OrdernetError, match="no ACC block"):
            c.get_accounts()


def test_get_balance_reads_a_o():
    """The Node lib reads `result.data.a.o`; we must do the same."""
    c = OrdernetReadOnlyClient("meitav")
    c._token = "x"
    acc = Account(key="ACC_001-1111111111", number="1111111111", name="primary")
    with patch.object(c._session, "get", return_value=_mock_response(200, {"a": {"o": 1875432.55}})):
        bal = c.get_balance(acc)
    assert bal == pytest.approx(1875432.55)


def test_get_balance_unexpected_shape_raises():
    c = OrdernetReadOnlyClient("meitav")
    c._token = "x"
    acc = Account(key="ACC_001-1111111111", number="1111111111", name="primary")
    with patch.object(c._session, "get", return_value=_mock_response(200, {"unexpected": "body"})):
        with pytest.raises(OrdernetError, match="unexpected GetAccountSecurities shape"):
            c.get_balance(acc)


def test_get_balance_uses_auth_header():
    """Confirm the bearer token actually goes out on each request."""
    c = OrdernetReadOnlyClient("meitav")
    c._token = "tok123"
    acc = Account(key="K", number="N", name="X")
    with patch.object(c._session, "get", return_value=_mock_response(200, {"a": {"o": 0}})) as get:
        c.get_balance(acc)
    headers = get.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok123"


# ------------------------------------------------------------- credentials


def test_load_credentials_rejects_fill_me_password(tmp_path):
    """Password as FILL_ME must always be rejected - there is no
    fallback for it."""
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({
        "broker": "meitav",
        "accounts": [{"label": "primary", "username": "1", "password": "FILL_ME", "account_number": "1"}],
    }))
    with pytest.raises(OrdernetError, match="not filled in"):
        load_credentials(p)


def test_load_credentials_rejects_unknown_broker(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"broker": "fakebroker", "accounts": [{"username": "u", "password": "p"}]}))
    with pytest.raises(OrdernetError, match="bad broker"):
        load_credentials(p)


def test_load_credentials_accepts_filled(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({
        "broker": "meitav",
        "accounts": [{"label": "primary", "username": "u", "password": "p", "account_number": "1"}],
    }))
    cfg = load_credentials(p)
    assert cfg["broker"] == "meitav"


def test_load_credentials_backfills_username_from_account_number(tmp_path):
    """Spark logs in with the account number as the username; the file
    is allowed to omit ``username`` entirely."""
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({
        "broker": "meitav",
        "accounts": [
            {"label": "primary", "password": "p", "account_number": "1111111111"},
        ],
    }))
    cfg = load_credentials(p)
    assert cfg["accounts"][0]["username"] == "1111111111"


def test_load_credentials_explicit_username_wins(tmp_path):
    """If the file does specify username, we keep it (don't clobber
    with account_number)."""
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({
        "broker": "meitav",
        "accounts": [
            {"label": "primary", "username": "explicit_user",
             "password": "p", "account_number": "1111111111"},
        ],
    }))
    cfg = load_credentials(p)
    assert cfg["accounts"][0]["username"] == "explicit_user"


def test_load_credentials_no_username_no_account_number_rejected(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({
        "broker": "meitav",
        "accounts": [{"label": "primary", "password": "p"}],
    }))
    with pytest.raises(OrdernetError, match="not filled in"):
        load_credentials(p)


def test_load_credentials_fill_me_username_with_account_number_falls_back(tmp_path):
    """Even if the file leaves ``username`` as the FILL_ME placeholder,
    a valid account_number lets us proceed. Common path on first
    run after the user trimmed the field."""
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({
        "broker": "meitav",
        "accounts": [
            {"label": "primary", "username": "FILL_ME", "password": "p",
             "account_number": "1111111111"},
        ],
    }))
    cfg = load_credentials(p)
    assert cfg["accounts"][0]["username"] == "1111111111"


# ------------------------------------------------------------- pull_all


def test_pull_all_balances_matches_by_account_number(tmp_path, monkeypatch):
    """Each cred should pick the matching Spark account by configured number."""
    cfg = {
        "broker": "meitav",
        "accounts": [
            {"label": "primary", "username": "u", "password": "p", "account_number": "1111111111"},
            {"label": "secondary", "username": "u", "password": "p", "account_number": "2222222222"},
            {"label": "joint", "username": "u", "password": "p", "account_number": "3333333333"},
        ],
    }
    spark_accounts_per_login = [
        # Each login typically sees only its own account.
        [Account(key="ACC_001-1111111111", number="1111111111", name="primary")],
        [Account(key="ACC_001-2222222222", number="2222222222", name="secondary")],
        [Account(key="ACC_001-3333333333", number="3333333333", name="joint")],
    ]
    balances = [1_875_000.0, 646_000.0, 1_284_000.0]

    call_count = {"i": 0}

    def fake_authenticate(self, username, password):
        self._token = "tok"

    def fake_get_accounts(self):
        i = call_count["i"]
        return spark_accounts_per_login[i]

    def fake_get_balance(self, account):
        i = call_count["i"]
        bal = balances[i]
        call_count["i"] += 1
        return bal

    monkeypatch.setattr(OrdernetReadOnlyClient, "authenticate", fake_authenticate)
    monkeypatch.setattr(OrdernetReadOnlyClient, "get_accounts", fake_get_accounts)
    monkeypatch.setattr(OrdernetReadOnlyClient, "get_balance", fake_get_balance)

    out = pull_all_balances(cfg)
    assert [r["balance_ils"] for r in out] == balances
    assert all(r["error"] is None for r in out)


def test_pull_all_balances_records_mismatched_account_number(monkeypatch):
    """If the login sees an account whose number != the configured one, and
    there is more than one account on the login, we record an error rather
    than blindly picking the first."""
    cfg = {
        "broker": "meitav",
        "accounts": [
            {"label": "primary", "username": "u", "password": "p", "account_number": "9999999999"},
        ],
    }

    def fake_authenticate(self, username, password):
        self._token = "tok"

    def fake_get_accounts(self):
        return [
            Account(key="ACC_001-1111", number="1111", name="A"),
            Account(key="ACC_001-2222", number="2222", name="B"),
        ]

    monkeypatch.setattr(OrdernetReadOnlyClient, "authenticate", fake_authenticate)
    monkeypatch.setattr(OrdernetReadOnlyClient, "get_accounts", fake_get_accounts)

    out = pull_all_balances(cfg)
    assert out[0]["balance_ils"] is None
    assert "no spark account matches" in out[0]["error"]


def test_pull_all_balances_falls_back_to_single_account(monkeypatch):
    """When the login has exactly one account, use it even if the number
    doesn't match exactly (defensive: Spark sometimes exposes the number
    formatted differently)."""
    cfg = {
        "broker": "meitav",
        "accounts": [
            {"label": "primary", "username": "u", "password": "p", "account_number": "doesnt-match"},
        ],
    }

    def fake_authenticate(self, username, password):
        self._token = "tok"

    def fake_get_accounts(self):
        return [Account(key="ACC_001-1111111111", number="1111111111", name="primary")]

    def fake_get_balance(self, account):
        return 999.0

    monkeypatch.setattr(OrdernetReadOnlyClient, "authenticate", fake_authenticate)
    monkeypatch.setattr(OrdernetReadOnlyClient, "get_accounts", fake_get_accounts)
    monkeypatch.setattr(OrdernetReadOnlyClient, "get_balance", fake_get_balance)

    out = pull_all_balances(cfg)
    assert out[0]["balance_ils"] == 999.0
    assert out[0]["matched_account"]["number"] == "1111111111"
