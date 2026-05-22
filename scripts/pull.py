#!/usr/bin/env python3
"""Dump current Ordernet/Spark balances for configured accounts."""

from __future__ import annotations

import argparse
import json
import sys

from ordernet_mcp.client import (
    OrdernetReadOnlyClient,
    load_credentials,
    pull_all_balances,
)


def discover(cfg: dict) -> list[dict]:
    out = []
    for acc_cfg in cfg["accounts"]:
        client = OrdernetReadOnlyClient(cfg["broker"])
        try:
            client.authenticate(acc_cfg["username"], acc_cfg["password"])
            accounts = client.get_accounts()
            out.append({
                "label": acc_cfg.get("label"),
                "configured_account_number": acc_cfg.get("account_number"),
                "spark_accounts": [
                    {"key": a.key, "number": a.number, "name": a.name} for a in accounts
                ],
            })
        except Exception as e:
            out.append({"label": acc_cfg.get("label"), "error": str(e)})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretty", action="store_true", help="indent JSON output")
    p.add_argument("--discover", action="store_true", help="list Spark accounts per login")
    args = p.parse_args()

    cfg = load_credentials()
    result = discover(cfg) if args.discover else pull_all_balances(cfg)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
