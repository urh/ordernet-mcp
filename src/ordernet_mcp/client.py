"""Read-only client for Spark/Ordernet (Israeli broker trading platform).

Spark exposes the same bearer token to read account data AND to place
trades; there is no read-only credential at the API level. This module is
how we make the *code path* read-only by construction:

1. The ONLY ``POST`` request the module is capable of issuing is
   ``/api/Auth/Authenticate`` (required to obtain the session token).
2. Every other API call goes through ``_get`` which validates the path
   against the explicit allowlist ``_ALLOWED_GET_PATHS``. Anything not on
   that list - notably any trade / order / cancel endpoint - raises
   immediately, before ``requests`` is even called.
3. There is intentionally no ``_post`` method, no ``place_order``, no
   ``cancel_order``, no ``deposit`` / ``withdraw``. To add one, you'd
   have to add a brand-new HTTP code path - which is meant to make any
   such addition obvious in code review.

These invariants are enforced by the tests in ``test_ordernet.py``.

Endpoints in scope (all read):

- ``Auth/Authenticate``        - login (only POST allowed)
- ``DataProvider/GetStaticData`` - list accounts
- ``Account/GetAccountSecurities`` - cash + securities + equity totals
- ``Account/GetHoldings``      - per-instrument positions
- ``Account/GetHoldingsSummery`` [sic, Spark API typo] - by-class summary
- ``Account/GetAccountTransactions`` - historical transactions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import requests

DEFAULT_CRED_PATH = Path(
    os.environ.get(
        "ORDERNET_CREDENTIALS_PATH",
        str(Path.home() / ".config" / "ordernet" / "credentials.json"),
    )
)
SUPPORTED_BROKERS = frozenset({"meitav", "psagot", "nesua"})

# The allowlist that turns this from "client" into "read-only client".
# Every method that hits the network MUST resolve to one of these paths.
# Adding a new path here is a deliberate change that must pass code
# review - and trade endpoints (which we never want) are simply absent.
_ALLOWED_GET_PATHS = frozenset({
    "DataProvider/GetStaticData",
    "Account/GetAccountSecurities",
    "Account/GetHoldings",
    "Account/GetHoldingsSummery",
    "Account/GetAccountTransactions",
})

# The only POST we ever issue. Inlined in `authenticate`; not callable
# from any helper. Listed here for documentation only.
_AUTH_PATH = "Auth/Authenticate"

log = logging.getLogger(__name__)


class OrdernetError(Exception):
    pass


class OrdernetSecurityError(OrdernetError):
    """Raised when something tried to take a code path that would let it
    write to the broker (e.g. an unknown GET path, or a path on the
    explicit deny list). This should be impossible at runtime - if you
    ever see it in a log, treat it as a serious bug."""


@dataclass(frozen=True)
class Account:
    """A single Spark account."""

    key: str       # "ACC_XXX-YYYYYY", required for API calls.
    number: str    # "YYYYYY", the human-friendly account number.
    name: str      # Whatever Spark labels the account in the UI.


@dataclass(frozen=True)
class SecuritiesTotals:
    """Headline totals from ``GetAccountSecurities``.

    All values in ILS. The naming inside Spark's response is misleading
    on purpose - ``NezilimMorning`` ("liquids") sounds like it's the
    total portfolio but is actually the margin/collateral figure. The
    real total is ``AccountValueMorning``. We map to clean names here
    and keep ``raw`` around for callers that need the rest.

    - ``cash_ils``: NIS cash pool currently in the account
    - ``total_ils``: total portfolio value (cash + all positions)
    - ``immediate_withdraw_ils``: how much can be withdrawn now
      (cash plus already-settled equities); pulled from
      ``GetHoldingsSummery`` rather than ``GetAccountSecurities``,
      may be 0 if we couldn't fetch the summary
    """

    cash_ils: float
    total_ils: float
    immediate_withdraw_ils: float
    raw: dict[str, Any]

    @property
    def securities_ils(self) -> float:
        """Total minus cash - the value of everything that isn't cash.
        Kept as a property for back-compat; the real source of truth is
        the per-instrument breakdown from ``get_holdings``."""
        return max(self.total_ils - self.cash_ils, 0.0)


@dataclass(frozen=True)
class Holding:
    """One row of ``GetHoldings`` - a single instrument position.

    Field mapping cross-references RMType in the Spark response:

    - ``symbol`` <- ``i`` (SYMBOL_NAM), e.g. "GOOGL US" / "ISH.FRF SP 500"
    - ``name`` <- ``j`` (BNO_NAME), Hebrew/English display name
    - ``quantity`` <- ``bd`` (NV), nominal value / units held
    - ``market_value_ils`` <- ``bf`` (VL)
    - ``cost_basis_ils`` <- ``be`` (COST), total ILS spent
    - ``portfolio_pct`` <- ``bk`` (ID_PCNT)
    - ``asset_class`` <- ``cg`` (SugBno), one of:
      ``StockShekel`` (Israeli/TASE stock or ETF), ``ForeignStockFrgn``
      (foreign stock e.g. NASDAQ), ``MatachMezuman`` (foreign cash),
      ``Pahak`` (tax-shelter voucher / "מגן מס"), and friends.
    """

    symbol: str | None
    name: str | None
    quantity: float
    market_value_ils: float
    cost_basis_ils: float | None
    portfolio_pct: float | None
    asset_class: str | None
    raw: dict[str, Any]

    @property
    def average_cost_ils(self) -> float | None:
        """Per-unit cost. None when quantity is zero (e.g. tax vouchers)."""
        if not self.cost_basis_ils or not self.quantity:
            return None
        return self.cost_basis_ils / self.quantity


class OrdernetReadOnlyClient:
    """Spark client capable of reading - and ONLY reading - account data.

    Each instance holds one auth session. Spark issues short-lived
    bearer tokens, so for batch jobs prefer one client per account
    rather than re-using a long-lived session.
    """

    def __init__(self, broker: str, *, timeout: float = 30.0):
        broker = broker.lower().strip()
        if broker not in SUPPORTED_BROKERS:
            raise ValueError(
                f"unsupported broker {broker!r}; expected one of {sorted(SUPPORTED_BROKERS)}"
            )
        self.broker = broker
        self.api_url = f"https://spark{broker}.ordernet.co.il/api"
        self._token: str | None = None
        self._session = requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------ auth
    #
    # The ONE place we issue a POST. There's no generalized ``_post``
    # helper; if you ever want to add another POST you'll have to write
    # one from scratch, which is the point.

    def authenticate(self, username: str, password: str) -> None:
        url = f"{self.api_url}/{_AUTH_PATH}"
        r = self._session.post(
            url,
            json={"username": username, "password": password},
            timeout=self._timeout,
        )
        if r.status_code != 200:
            raise OrdernetError(f"auth failed for {username}: {r.status_code} {r.text[:200]}")
        body = r.json()
        token = body.get("l")
        if not token:
            raise OrdernetError(f"auth response missing 'l' (token) field: {body}")
        self._token = token

    @property
    def _auth_headers(self) -> dict[str, str]:
        if not self._token:
            raise OrdernetError("not authenticated; call authenticate() first")
        return {"Authorization": f"Bearer {self._token}"}

    # ------------------------------------------------------------ guarded GET
    #
    # All non-auth network calls funnel through here. The allowlist
    # check is what makes this client read-only-by-construction.

    def _get(self, path: str, params: dict | None = None) -> Any:
        if path not in _ALLOWED_GET_PATHS:
            raise OrdernetSecurityError(
                f"refusing to GET {path!r}: not in read-only allowlist "
                f"({sorted(_ALLOWED_GET_PATHS)})"
            )
        url = f"{self.api_url}/{path}"
        r = self._session.get(
            url,
            params=params or {},
            headers=self._auth_headers,
            timeout=self._timeout,
        )
        if r.status_code != 200:
            raise OrdernetError(f"{path} failed: {r.status_code} {r.text[:200]}")
        return r.json()

    # ---------------------------------------------------------- accounts

    def get_accounts(self) -> list[Account]:
        data = self._get("DataProvider/GetStaticData")
        acc_entry = next((x for x in data if x.get("b") == "ACC"), None)
        if not acc_entry:
            raise OrdernetError("no ACC block in GetStaticData response")
        out: list[Account] = []
        for x in acc_entry.get("a", []):
            key = x.get("_k", "")
            inner = x.get("a", {})
            out.append(Account(
                key=key,
                number=str(inner.get("b", "")),
                name=str(inner.get("e", "")),
            ))
        return out

    # ---------------------------------------------------------- balances

    def get_securities(self, account: Account) -> SecuritiesTotals:
        """Cash, total portfolio value, and immediate withdrawability.

        Field mapping (Spark API response keys, verified against live
        Meitav responses):

        - ``a`` = CashMorning (start-of-day NIS cash)
        - ``d`` = CashCurrent (current NIS cash, what we want)
        - ``b`` = NezilimMorning - **collateral / margin**, NOT total
        - ``e`` = NezilimCurrent - same, current
        - ``o`` = AccountValueMorning - **the actual total**

        ``immediate_withdraw_ils`` requires a second call to
        ``GetHoldingsSummery``; we do it inline so callers don't have
        to. If that fails we fall back to 0 rather than blowing up.
        """
        data = self._get("Account/GetAccountSecurities", {"accountKey": account.key})
        a = data.get("a") or {}
        try:
            total = float(a["o"])
        except (KeyError, TypeError, ValueError) as e:
            raise OrdernetError(
                f"unexpected GetAccountSecurities shape for {account.key}: {data!r}"
            ) from e

        def _maybe_float(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        # Prefer the "current" snapshot to the "morning" one - matches
        # what the web client shows.
        cash = _maybe_float(a.get("d", a.get("a", 0)))

        # ``ImmediateWithdraw`` only lives on GetHoldingsSummery. Best-
        # effort - some accounts return errors here even though the
        # rest of the API works.
        immediate = 0.0
        summary: dict | None = None
        try:
            summary = self._get(
                "Account/GetHoldingsSummery", {"accountKey": account.key}
            )
            # RMTotalType: h = ImmediateWithdraw
            immediate = _maybe_float((summary or {}).get("h", 0))
        except OrdernetError as e:
            log.warning("GetHoldingsSummery failed for %s: %s", account.key, e)

        raw: dict[str, Any] = {"securities": data}
        if summary is not None:
            raw["summary"] = summary
        return SecuritiesTotals(
            cash_ils=cash,
            total_ils=total,
            immediate_withdraw_ils=immediate,
            raw=raw,
        )

    def get_balance(self, account: Account) -> float:
        """Just the equity total, for back-compat with the original lib."""
        return self.get_securities(account).total_ils

    # ---------------------------------------------------------- holdings

    def get_holdings(self, account: Account) -> list[Holding]:
        """Per-instrument positions.

        The underlying response is an array of objects with single-letter
        keys (Spark's space-saving convention). We map only the keys we
        care about and stash the full row in ``raw`` for callers that
        want more.

        Field mapping (Spark RMType keys, verified against live responses):

        - ``i`` = SYMBOL_NAM  (e.g. ``"GOOGL US"``, ``"ISH.FRF SP 500"``)
        - ``j`` = BNO_NAME    (display name, often Hebrew)
        - ``bd`` = NV         (nominal value / units held)
        - ``bf`` = VL         (current market value, ILS)
        - ``be`` = COST       (cost basis total, ILS)
        - ``bk`` = ID_PCNT    (% of portfolio)
        - ``cg`` = SugBno     (asset class code, e.g. ``StockShekel``,
          ``ForeignStockFrgn``, ``MatachMezuman``, ``Pahak``)
        """
        data = self._get("Account/GetHoldings", {"accountKey": account.key})
        if not isinstance(data, list):
            raise OrdernetError(
                f"GetHoldings returned non-list for {account.key}: {type(data).__name__}"
            )

        def _f(row: dict[str, Any], key: str) -> float:
            v = row.get(key)
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        def _maybe(row: dict[str, Any], key: str) -> float | None:
            v = row.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        out: list[Holding] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            symbol = row.get("i")
            name = row.get("j")
            asset_class = row.get("cg")
            out.append(Holding(
                symbol=str(symbol) if symbol is not None else None,
                name=str(name) if name is not None else None,
                quantity=_f(row, "bd"),
                market_value_ils=_f(row, "bf"),
                cost_basis_ils=_maybe(row, "be"),
                portfolio_pct=_maybe(row, "bk"),
                asset_class=str(asset_class) if asset_class is not None else None,
                raw=row,
            ))
        return out

    def get_holdings_summary(self, account: Account) -> dict:
        """By-class summary (equities / bonds / cash etc.). Returns the
        raw dict because we don't currently slice it."""
        return self._get("Account/GetHoldingsSummery", {"accountKey": account.key})

    # ---------------------------------------------------------- transactions

    def get_transactions(
        self,
        account: Account,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Historical transactions in [start, end]. Spark requires both
        dates be in the same calendar year; callers should chunk by
        year if they need multi-year history."""
        if start.year != end.year:
            raise ValueError(
                "Spark requires start.year == end.year for GetAccountTransactions; "
                "split the request by calendar year"
            )
        data = self._get(
            "Account/GetAccountTransactions",
            {
                "accountKey": account.key,
                "startDate": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "endDate": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            },
        )
        if not isinstance(data, list):
            raise OrdernetError(
                f"GetAccountTransactions returned non-list: {type(data).__name__}"
            )
        return data


# ---------------------------------------------------------- credentials helpers


def load_credentials(path: Path | None = None) -> dict:
    """Read the JSON credentials file, validate shape, and normalize.

    For Spark/Meitav the login *username* IS the account number, so the
    credentials file is allowed to omit ``username``; we backfill it
    from ``account_number`` here. Anything that was filled in
    explicitly wins.
    """
    p = path or DEFAULT_CRED_PATH
    raw = p.read_text(encoding="utf-8")
    cfg = json.loads(raw)
    if cfg.get("broker") not in SUPPORTED_BROKERS:
        raise OrdernetError(f"credentials: bad broker {cfg.get('broker')!r}")
    accs = cfg.get("accounts") or []
    if not accs:
        raise OrdernetError("credentials: no accounts listed")

    _placeholder = (None, "", "FILL_ME")
    for a in accs:
        # Normalize: Spark uses the account number as the username, so
        # the file may omit `username` entirely OR leave it as the
        # FILL_ME placeholder. Either way, fall back to account_number.
        if a.get("username") in _placeholder:
            num = a.get("account_number")
            if num and num not in _placeholder:
                a["username"] = num
        if a.get("username") in _placeholder:
            raise OrdernetError(
                f"credentials: account {a.get('label')!r} not filled in "
                f"(no username and no usable account_number)"
            )
        if a.get("password") in _placeholder:
            raise OrdernetError(
                f"credentials: account {a.get('label')!r} not filled in (password)"
            )
    return cfg


def pull_all_balances(cfg: dict | None = None) -> list[dict]:
    """For each account in the credentials file, log in and return its
    securities totals. Useful for batch jobs; the MCP server exposes
    finer-grained tools."""
    cfg = cfg or load_credentials()
    broker = cfg["broker"]
    out: list[dict] = []
    for acc_cfg in cfg["accounts"]:
        client = OrdernetReadOnlyClient(broker)
        try:
            client.authenticate(acc_cfg["username"], acc_cfg["password"])
            spark_accounts = client.get_accounts()
        except Exception as e:
            log.exception("ordernet auth/list failed for %s", acc_cfg.get("label"))
            out.append({
                "label": acc_cfg.get("label"),
                "configured_account_number": acc_cfg.get("account_number"),
                "error": str(e),
                "balance_ils": None,
                "matched_account": None,
                "all_accounts": [],
            })
            continue

        configured = str(acc_cfg.get("account_number", ""))
        match = next((a for a in spark_accounts if a.number == configured), None)
        if match is None and len(spark_accounts) == 1:
            match = spark_accounts[0]

        balance: float | None = None
        err: str | None = None
        if match is not None:
            try:
                balance = client.get_balance(match)
            except Exception as e:
                log.exception("get_balance failed for %s/%s", acc_cfg.get("label"), match.key)
                err = str(e)
        else:
            err = (
                f"no spark account matches configured account_number={configured!r}; "
                f"available: {[a.number for a in spark_accounts]}"
            )

        out.append({
            "label": acc_cfg.get("label"),
            "configured_account_number": configured,
            "matched_account": (
                {"key": match.key, "number": match.number, "name": match.name}
                if match else None
            ),
            "all_accounts": [
                {"key": a.key, "number": a.number, "name": a.name} for a in spark_accounts
            ],
            "balance_ils": balance,
            "error": err,
        })
    return out


# Back-compat alias - some call sites still import the old name.
OrdernetClient = OrdernetReadOnlyClient
