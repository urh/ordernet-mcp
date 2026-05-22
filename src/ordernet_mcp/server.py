"""Read-only Ordernet MCP server (stdio).

Exposes three read tools backed by ``bot.ordernet.OrdernetReadOnlyClient``:

- ``get_total_amount`` - total equity (cash + securities) per account
- ``get_total_cash``   - just the NIS cash balance
- ``get_current_stocks`` - per-instrument holdings

What this server does NOT do, and is structurally incapable of doing:

- Place / cancel / modify any order.
- Transfer money between accounts.
- Even *read* anything beyond the four whitelisted GET endpoints in
  ``bot.ordernet`` (the only way it talks to Spark).

If a future tool wants to add write capability the only path is to add
new code to ``bot.ordernet`` (which is read-only by construction and
test-enforced), then expose it here. That's a deliberate, reviewable
change - not something an LLM client can sneak past.

Run as ``python -m ordernet_mcp.server``. Configure in your MCP client's
``mcp.json`` (see README).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ordernet_mcp.client import (
    Account,
    Holding,
    OrdernetError,
    OrdernetReadOnlyClient,
    OrdernetSecurityError,
    SecuritiesTotals,
    load_credentials,
)

log = logging.getLogger(__name__)

server: Server = Server("ordernet-readonly")


# ---------------------------------------------------------------- credentials

# Re-resolved at each tool invocation so the user can fix a busted creds
# file without restarting the MCP. Cheap (file read + JSON parse).


def _label_lookup(cfg: dict, label: str | None) -> list[dict]:
    """Return the cred entries matching ``label`` (case-insensitive). If
    ``label`` is None, returns all entries. Raises if the label is given
    but matches nothing - we want loud failures, not silent empty
    results."""
    accs = cfg["accounts"]
    if label is None:
        return list(accs)
    needle = label.strip().lower()
    matches = [a for a in accs if str(a.get("label", "")).lower() == needle]
    if not matches:
        valid = sorted({str(a.get("label", "")) for a in accs})
        raise OrdernetError(
            f"unknown account label {label!r}; expected one of {valid}"
        )
    return matches


# ---------------------------------------------------------------- helpers

@dataclass(frozen=True)
class _Resolved:
    label: str
    account_number: str
    spark_account: Account
    client: OrdernetReadOnlyClient  # already authenticated


def _resolve_one(broker: str, acc_cfg: dict) -> _Resolved:
    """Authenticate one credentials entry and find its Spark account."""
    client = OrdernetReadOnlyClient(broker)
    client.authenticate(acc_cfg["username"], acc_cfg["password"])
    spark_accounts = client.get_accounts()
    configured = str(acc_cfg.get("account_number", ""))
    match = next((a for a in spark_accounts if a.number == configured), None)
    if match is None and len(spark_accounts) == 1:
        match = spark_accounts[0]
    if match is None:
        raise OrdernetError(
            f"no spark account matches configured account_number={configured!r}; "
            f"spark sees {[a.number for a in spark_accounts]}"
        )
    return _Resolved(
        label=str(acc_cfg.get("label", "")),
        account_number=match.number,
        spark_account=match,
        client=client,
    )


def _resolve_many(label: str | None) -> list[_Resolved]:
    cfg = load_credentials()
    broker = cfg["broker"]
    return [_resolve_one(broker, a) for a in _label_lookup(cfg, label)]


# ---------------------------------------------------------------- formatting

def _fmt_ils(v: float | None) -> str:
    if v is None:
        return "?"
    return f"{v:,.0f} ₪"


def _serialize_holding(h: Holding) -> dict:
    return {
        "symbol": h.symbol,
        "name": h.name,
        "asset_class": h.asset_class,
        "quantity": h.quantity,
        "market_value_ils": h.market_value_ils,
        "cost_basis_ils": h.cost_basis_ils,
        "average_cost_ils": h.average_cost_ils,
        "portfolio_pct": h.portfolio_pct,
    }


def _serialize_totals(t: SecuritiesTotals) -> dict:
    return {
        "cash_ils": t.cash_ils,
        "securities_ils": t.securities_ils,
        "total_ils": t.total_ils,
        "immediate_withdraw_ils": t.immediate_withdraw_ils,
    }


# ---------------------------------------------------------------- tools

_LABEL_DESCRIPTION = (
    "Optional account label from your credentials file (e.g. 'primary', "
    "'joint'). Omit to query all configured accounts. Match is "
    "case-insensitive."
)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_total_amount",
            description=(
                "READ-ONLY. Total portfolio equity (cash + securities) in "
                "ILS for one or all configured Ordernet/Spark broker accounts. "
                "Returns per-account rows plus a grand total when multiple "
                "accounts are queried. This server cannot place trades or "
                "move money - it only reads balances."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "account_label": {
                        "type": "string",
                        "description": _LABEL_DESCRIPTION,
                    },
                },
            },
        ),
        Tool(
            name="get_total_cash",
            description=(
                "READ-ONLY. NIS cash balance for one or all configured "
                "Ordernet/Spark accounts. Cash here means the broker's "
                "uninvested ILS pool - settled funds available to invest "
                "or withdraw. Does not include open buy orders or pending "
                "settlements."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "account_label": {
                        "type": "string",
                        "description": _LABEL_DESCRIPTION,
                    },
                },
            },
        ),
        Tool(
            name="get_current_stocks",
            description=(
                "READ-ONLY. List of current holdings (positions) per "
                "Ordernet/Spark account. For each instrument returns "
                "symbol, name, quantity, market value (ILS) and average "
                "cost where Spark exposes it. Empty list means no "
                "open positions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "account_label": {
                        "type": "string",
                        "description": _LABEL_DESCRIPTION,
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    label = arguments.get("account_label")
    try:
        if name == "get_total_amount":
            return _do_total_amount(label)
        if name == "get_total_cash":
            return _do_total_cash(label)
        if name == "get_current_stocks":
            return _do_current_stocks(label)
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except OrdernetSecurityError as e:
        # Should be impossible at runtime; if it ever happens we want a
        # very loud signal in the MCP output.
        log.exception("ordernet security error")
        return [TextContent(
            type="text",
            text=f"SECURITY: ordernet client refused a non-allowlisted call: {e}",
        )]
    except OrdernetError as e:
        log.exception("ordernet error")
        return [TextContent(type="text", text=f"Ordernet error: {e}")]
    except Exception as e:
        log.exception("unexpected error in ordernet MCP")
        return [TextContent(type="text", text=f"Unexpected error: {e}")]


def _do_total_amount(label: str | None) -> list[TextContent]:
    resolved = _resolve_many(label)
    rows: list[dict] = []
    grand_total = 0.0
    grand_cash = 0.0
    grand_sec = 0.0
    for r in resolved:
        t = r.client.get_securities(r.spark_account)
        rows.append({
            "label": r.label,
            "account_number": r.account_number,
            **_serialize_totals(t),
        })
        grand_total += t.total_ils
        grand_cash += t.cash_ils
        grand_sec += t.securities_ils
    payload: dict = {
        "accounts": rows,
        "currency": "ILS",
    }
    if len(rows) > 1:
        payload["grand_total_ils"] = grand_total
        payload["grand_cash_ils"] = grand_cash
        payload["grand_securities_ils"] = grand_sec
    pretty_lines = ["Total amount (ILS):"]
    for r in rows:
        pretty_lines.append(
            f"  - {r['label']} ({r['account_number']}): {_fmt_ils(r['total_ils'])}"
            f" (cash {_fmt_ils(r['cash_ils'])}, securities {_fmt_ils(r['securities_ils'])})"
        )
    if len(rows) > 1:
        pretty_lines.append(f"  -- TOTAL: {_fmt_ils(grand_total)}")
    pretty_lines.append("")
    pretty_lines.append("```json")
    pretty_lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    pretty_lines.append("```")
    return [TextContent(type="text", text="\n".join(pretty_lines))]


def _do_total_cash(label: str | None) -> list[TextContent]:
    resolved = _resolve_many(label)
    rows: list[dict] = []
    grand = 0.0
    for r in resolved:
        t = r.client.get_securities(r.spark_account)
        rows.append({
            "label": r.label,
            "account_number": r.account_number,
            "cash_ils": t.cash_ils,
        })
        grand += t.cash_ils
    pretty = ["Cash (ILS):"]
    for r in rows:
        pretty.append(f"  - {r['label']} ({r['account_number']}): {_fmt_ils(r['cash_ils'])}")
    if len(rows) > 1:
        pretty.append(f"  -- TOTAL: {_fmt_ils(grand)}")
    payload = {"accounts": rows, "currency": "ILS"}
    if len(rows) > 1:
        payload["grand_cash_ils"] = grand
    pretty.extend(["", "```json", json.dumps(payload, ensure_ascii=False, indent=2), "```"])
    return [TextContent(type="text", text="\n".join(pretty))]


def _do_current_stocks(label: str | None) -> list[TextContent]:
    resolved = _resolve_many(label)
    sections: list[str] = []
    for r in resolved:
        holdings = r.client.get_holdings(r.spark_account)
        rows = [_serialize_holding(h) for h in holdings]
        rows.sort(key=lambda h: h["market_value_ils"], reverse=True)
        sections.append(f"### {r.label} ({r.account_number})")
        if not rows:
            sections.append("(no positions)")
        else:
            for h in rows:
                qty = h["quantity"]
                qty_s = f"{qty:,.0f}" if qty == int(qty) else f"{qty:,.4f}"
                sym = h.get("symbol") or "?"
                name = h.get("name") or "?"
                cost = h.get("cost_basis_ils")
                cost_s = f", cost {_fmt_ils(cost)}" if cost else ""
                pct = h.get("portfolio_pct")
                pct_s = f", {pct:.1f}% of port" if pct is not None and pct > 0 else ""
                cls = h.get("asset_class") or ""
                cls_s = f" [{cls}]" if cls else ""
                sections.append(
                    f"- {sym} ({name}){cls_s}: qty {qty_s}, "
                    f"value {_fmt_ils(h['market_value_ils'])}{cost_s}{pct_s}"
                )
        sections.append("")
        sections.append("```json")
        sections.append(json.dumps({"label": r.label, "account_number": r.account_number, "holdings": rows}, ensure_ascii=False, indent=2))
        sections.append("```")
        sections.append("")
    return [TextContent(type="text", text="\n".join(sections).rstrip())]


# ---------------------------------------------------------------- entrypoint


async def _main() -> None:
    logging.basicConfig(
        level=os.environ.get("ORDERNET_MCP_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stdio is reserved for MCP framing
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
