"""Tests for the read-only Ordernet MCP server.

Two layers:

1. **Surface invariants**: tool names, descriptions, schemas. We
   absolutely do not want a write-capable tool to slip in.
2. **Behavioral**: each tool, when invoked, calls the right read-only
   method on the underlying client. Networking is faked.
"""

from __future__ import annotations

import asyncio
import json
import re
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ordernet_mcp import server as ordernet_mcp
from ordernet_mcp.client import (
    Account,
    Holding,
    OrdernetReadOnlyClient,
    SecuritiesTotals,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# =========================================================== surface invariants


def test_tool_names_are_exactly_the_three_read_tools():
    tools = _run(ordernet_mcp.list_tools())
    names = sorted(t.name for t in tools)
    assert names == ["get_current_stocks", "get_total_amount", "get_total_cash"], names


def test_tool_descriptions_advertise_read_only():
    """Every tool description must mention read-only-ness so any LLM
    consuming the tool list sees the constraint."""
    tools = _run(ordernet_mcp.list_tools())
    for t in tools:
        assert "READ-ONLY" in t.description.upper(), (
            f"tool {t.name!r} description should advertise read-only: {t.description!r}"
        )


def test_no_write_keywords_in_tool_names():
    """Belt-and-braces: tool names mustn't suggest write capability."""
    forbidden = {"place", "trade", "order", "submit", "cancel",
                 "buy", "sell", "deposit", "withdraw", "transfer"}
    tools = _run(ordernet_mcp.list_tools())
    for t in tools:
        lc = t.name.lower()
        for f in forbidden:
            assert f not in lc, (
                f"tool name {t.name!r} contains forbidden keyword {f!r}"
            )


def test_module_does_not_import_requests():
    """The MCP server should ONLY touch the network through
    ``OrdernetReadOnlyClient``. Importing ``requests`` directly would
    open a side channel; this test catches it."""
    src = Path(ordernet_mcp.__file__).read_text(encoding="utf-8")
    # Match `import requests` or `from requests import ...` at line start
    # (allowing leading whitespace).
    bad = re.findall(r"^\s*(?:import\s+requests|from\s+requests\s+import)", src, re.MULTILINE)
    assert not bad, f"ordernet_mcp imports requests directly: {bad}"


# =========================================================== behavioral


@pytest.fixture
def fake_creds(monkeypatch):
    cfg = {
        "broker": "meitav",
        "accounts": [
            {"label": "primary", "username": "u", "password": "p",
             "account_number": "1111111111"},
            {"label": "secondary", "username": "u", "password": "p",
             "account_number": "2222222222"},
            {"label": "joint", "username": "u", "password": "p",
             "account_number": "3333333333"},
        ],
    }
    monkeypatch.setattr(ordernet_mcp, "load_credentials", lambda *a, **kw: cfg)
    return cfg


@pytest.fixture
def stub_spark(monkeypatch):
    """Stub the read-only client so each instance returns a deterministic
    account + totals + holdings keyed by the configured account_number.

    We override ``_resolve_one`` itself rather than monkeypatching every
    method on the client; this gives us per-account fixtures without a
    bunch of boilerplate while still exercising the rest of
    ``ordernet_mcp.call_tool``.
    """
    def _t(cash, total, withdraw=0.0):
        return SecuritiesTotals(
            cash_ils=cash, total_ils=total,
            immediate_withdraw_ils=withdraw, raw={},
        )

    def _h(symbol, name, qty, mv, cost=None, pct=None, cls="StockShekel"):
        return Holding(
            symbol=symbol, name=name, quantity=qty,
            market_value_ils=mv, cost_basis_ils=cost,
            portfolio_pct=pct, asset_class=cls, raw={},
        )

    fixtures = {
        "1111111111": {
            "totals": _t(15000.0, 1875000.0, withdraw=14000.0),
            "holdings": [
                _h("AAPL", "Apple", 100, 900000.0, cost=600000.0, pct=48.0),
                _h("GOOG", "Alphabet", 50, 960000.0, cost=400000.0, pct=51.2),
            ],
        },
        "2222222222": {
            "totals": _t(46000.0, 646000.0, withdraw=45000.0),
            "holdings": [
                _h("VOO", "Vanguard S&P", 200, 600000.0, cost=500000.0, pct=92.9),
            ],
        },
        "3333333333": {
            "totals": _t(84000.0, 1284000.0, withdraw=80000.0),
            "holdings": [],  # joint account holds only cash here
        },
    }

    def fake_resolve_one(broker, acc_cfg):
        num = str(acc_cfg["account_number"])
        f = fixtures[num]

        client = MagicMock(spec=OrdernetReadOnlyClient)
        client.get_securities.return_value = f["totals"]
        client.get_holdings.return_value = f["holdings"]

        spark_account = Account(
            key=f"ACC_001-{num}",
            number=num,
            name=acc_cfg.get("label", ""),
        )
        return ordernet_mcp._Resolved(
            label=acc_cfg.get("label", ""),
            account_number=num,
            spark_account=spark_account,
            client=client,
        )

    monkeypatch.setattr(ordernet_mcp, "_resolve_one", fake_resolve_one)
    return fixtures


def test_get_total_amount_all_accounts(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("get_total_amount", {}))
    assert len(out) == 1
    text = out[0].text
    # JSON block at the end has the structured payload.
    payload = json.loads(text.split("```json")[1].split("```")[0])
    assert payload["currency"] == "ILS"
    assert payload["grand_total_ils"] == pytest.approx(1875000 + 646000 + 1284000)
    labels = [r["label"] for r in payload["accounts"]]
    assert labels == ["primary", "secondary", "joint"]


def test_get_total_amount_filtered_by_label(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("get_total_amount", {"account_label": "secondary"}))
    payload = json.loads(out[0].text.split("```json")[1].split("```")[0])
    assert len(payload["accounts"]) == 1
    assert payload["accounts"][0]["label"] == "secondary"
    assert payload["accounts"][0]["total_ils"] == 646000.0
    # Single-account responses don't include grand totals (no point).
    assert "grand_total_ils" not in payload


def test_get_total_amount_label_case_insensitive(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("get_total_amount", {"account_label": "pRiMaRy"}))
    payload = json.loads(out[0].text.split("```json")[1].split("```")[0])
    assert payload["accounts"][0]["label"] == "primary"


def test_get_total_amount_unknown_label_errors_loudly(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("get_total_amount", {"account_label": "Bogus"}))
    text = out[0].text
    assert "Ordernet error" in text
    assert "unknown account label" in text


def test_get_total_cash_returns_only_cash(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("get_total_cash", {}))
    payload = json.loads(out[0].text.split("```json")[1].split("```")[0])
    cashes = {r["label"]: r["cash_ils"] for r in payload["accounts"]}
    assert cashes == {"primary": 15000.0, "secondary": 46000.0, "joint": 84000.0}
    assert payload["grand_cash_ils"] == pytest.approx(15000 + 46000 + 84000)
    # The tool intentionally does NOT expose total_ils / securities_ils.
    for r in payload["accounts"]:
        assert "total_ils" not in r
        assert "securities_ils" not in r


def test_get_current_stocks_returns_holdings_per_account(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("get_current_stocks", {"account_label": "primary"}))
    text = out[0].text
    assert "AAPL" in text
    assert "GOOG" in text
    # Sorted by market value descending - GOOG (960k) before AAPL (900k).
    assert text.index("GOOG") < text.index("AAPL")


def test_get_current_stocks_handles_empty_account(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("get_current_stocks", {"account_label": "joint"}))
    text = out[0].text
    assert "no positions" in text


def test_unknown_tool_returns_error(fake_creds, stub_spark):
    out = _run(ordernet_mcp.call_tool("place_order", {}))
    assert "Unknown tool" in out[0].text


def test_call_tool_does_not_invoke_get_transactions(fake_creds, stub_spark):
    """We exposed get_transactions on the client (might want it later
    for the monthly refresh) but did NOT expose it via MCP. None of
    the three tools should call it."""
    out = _run(ordernet_mcp.call_tool("get_total_amount", {}))
    out2 = _run(ordernet_mcp.call_tool("get_total_cash", {}))
    out3 = _run(ordernet_mcp.call_tool("get_current_stocks", {}))
    # The fake client mocks have call records.
    # Each fixture returns 3 client mocks per call (one per account).
    # Easier to test that the MCP module functions don't *reference*
    # get_transactions in the dispatch path.
    src = Path(ordernet_mcp.__file__).read_text()
    # Allowed to be imported (we re-export) but never called.
    calls = re.findall(r"\.get_transactions\s*\(", src)
    assert calls == [], f"ordernet_mcp dispatcher calls get_transactions: {calls}"


# =========================================================== entrypoint smoke


def test_module_has_main_for_python_dash_m():
    """Must be runnable as ``python -m bot.ordernet_mcp`` (mcp.json
    invocation)."""
    assert hasattr(ordernet_mcp, "main")
    assert callable(ordernet_mcp.main)
