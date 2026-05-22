# ordernet-mcp

Read-only [Model Context Protocol](https://modelcontextprotocol.io/) server for **Ordernet / Spark** — the online trading platform used by Israeli brokers such as Meitav, Psagot, and IBI.

**Author:** [Uri Harduf](https://github.com/urh)

## What is it good for?

Connect an AI assistant (Cursor, Claude Desktop, etc.) to your brokerage accounts so it can answer questions like:

- *"What is my total portfolio value across all accounts?"*
- *"How much cash do I have available to invest?"*
- *"List my current stock and ETF holdings with market values."*

The server is **read-only by construction**: it can authenticate and fetch balances and holdings, but cannot place trades, cancel orders, or move money. Every HTTP path goes through an explicit allowlist enforced in code and tests.

## Tools

| Tool | Description |
|------|-------------|
| `get_total_amount` | Total portfolio equity (cash + securities) in ILS |
| `get_total_cash` | NIS cash balance only |
| `get_current_stocks` | Per-instrument holdings with quantities and values |

All tools accept an optional `account_label` to filter to one account from your credentials file.

## Setup

### 1. Credentials

Copy the example and fill in your broker details:

```bash
mkdir -p ~/.config/ordernet
cp credentials.example.json ~/.config/ordernet/credentials.json
chmod 600 ~/.config/ordernet/credentials.json
```

For Spark/Meitav, the login username is usually your account number, so you can omit `username` and set only `account_number`.

Supported brokers: `meitav`, `psagot`, `ibi`.

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. MCP configuration

Add to your MCP client config (e.g. Cursor `mcp.json`):

```json
{
  "mcpServers": {
    "ordernet": {
      "command": "/path/to/ordernet-mcp/.venv/bin/python",
      "args": ["-m", "ordernet_mcp"],
      "env": {
        "ORDERNET_CREDENTIALS_PATH": "/Users/you/.config/ordernet/credentials.json"
      }
    }
  }
}
```

## CLI

Dump all configured account balances as JSON:

```bash
python scripts/pull.py
python scripts/pull.py --discover   # list Spark accounts visible to each login
```

## Security notes

- Store credentials outside the repo with restrictive permissions (`chmod 600`).
- Spark's API uses the same bearer token for reads and writes; this project limits itself to read endpoints only.
- Review the allowlist in `src/ordernet_mcp/client.py` before trusting the server with live credentials.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
