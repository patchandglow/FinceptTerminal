"""
mcp_server.py — Fincept MCP HTTP server for RD-Agent tool use.

Provides financial data tools that rdagent loops can call via
MCPServerStreamableHTTP. Runs as a standalone FastMCP server.

Tools exposed:
  - market_data        : fetch OHLCV + quote data for a symbol
  - financial_news     : fetch recent news headlines
  - economics_data     : fetch macro indicators (GDP, CPI, rates)
  - factor_backtest    : quick IC/Sharpe estimate for a factor expression
  - symbol_search      : search for ticker symbols

Usage (standalone):
  python mcp_server.py --port 18765

rdagent connects via:
  MCPServerStreamableHTTP("http://localhost:18765/mcp")
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ── Factor-expression sandbox ────────────────────────────────────────────────
# factor_backtest evaluates a caller-supplied expression. eval() with
# {"__builtins__": {}} is NOT a sandbox: exposing the pandas/numpy module
# objects hands over pd.read_pickle (deserialises arbitrary objects from a URL)
# and np.load, and even without them `().__class__.__mro__[1].__subclasses__()`
# walks back to the whole class hierarchy. Since factor_expr arrives from an
# LLM that reads untrusted market/news text, that is a prompt-injection → RCE
# path.
#
# So the expression is parsed and checked against an allowlist before it ever
# reaches eval(). Attribute access is the escape hatch, so it is gated on a
# name allowlist: `close.rolling(20).mean()` and `np.log(close)` (the syntax
# the docstring advertises) still work, while `.read_pickle` / `.__class__` /
# any dunder is rejected at parse time.

_FACTOR_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Attribute, ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List,
    ast.keyword, ast.Slice, ast.Subscript, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

_FACTOR_ALLOWED_ATTRS = frozenset({
    # numpy ufuncs
    "log", "log1p", "log10", "exp", "sqrt", "abs", "sign", "tanh", "clip",
    "maximum", "minimum", "where", "power", "square", "floor", "ceil",
    "nan_to_num",
    # pandas Series / rolling / ewm / expanding
    "rolling", "ewm", "expanding", "shift", "diff", "pct_change", "rank",
    "mean", "median", "std", "var", "sum", "min", "max", "corr", "cov",
    "skew", "kurt", "quantile", "cumsum", "cumprod", "fillna", "dropna",
    "round",
})


def _compile_factor_expr(expr: str):
    """Parse+validate a factor expression, returning a compiled code object.

    Raises ValueError with a caller-facing reason if the expression uses
    anything outside the allowlist.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid syntax: {e}") from None

    for node in ast.walk(tree):
        if not isinstance(node, _FACTOR_ALLOWED_NODES):
            raise ValueError(f"disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in _FACTOR_ALLOWED_ATTRS:
                raise ValueError(f"disallowed attribute: .{node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"disallowed name: {node.id}")
        # Only method calls (allowlisted above) and calls on plain names are
        # permitted; a call on anything else is a construction we have not
        # reasoned about.
        if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Attribute, ast.Name)):
            raise ValueError("disallowed call target")

    return compile(tree, "<factor_expr>", "eval")

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

MCP_SERVER_AVAILABLE = False
# Which FastMCP is in use. The two packages export the same class name and the
# same decorators, but they do NOT agree on how a server is started, so the
# serving code has to know which one it got. See _serve().
#   "mcp"     — the official SDK (`pip install mcp`), FastMCP under mcp.server
#   "fastmcp" — the standalone package (`pip install fastmcp`)
MCP_IMPL = ""
try:
    from mcp.server.fastmcp import FastMCP
    MCP_SERVER_AVAILABLE = True
    MCP_IMPL = "mcp"
except ImportError:
    try:
        from fastmcp import FastMCP  # type: ignore
        MCP_SERVER_AVAILABLE = True
        MCP_IMPL = "fastmcp"
    except ImportError:
        FastMCP = None  # type: ignore

YFINANCE_AVAILABLE = False
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    pass

REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Macro indicator catalogue (economics_data)
# ---------------------------------------------------------------------------
# Three kinds of indicator, and the difference is the whole point: a caller
# cannot interpret a number without knowing which kind produced it.
#
#   market — the ticker *is* the indicator.   vix -> ^VIX really is the VIX.
#   proxy  — the ticker is a tradeable stand-in whose price is NOT the
#            statistic. ^VXX is a volatility ETN; its ~5200 has no reading as
#            an unemployment rate. Presenting it unlabelled next to a market
#            value invites exactly that misreading.
#   static — a constant baked in here (or via env override), not fetched.
#
# These live at module scope, not inside build_mcp_server(), so the valid-name
# list has exactly one home. It previously had three — the two dicts, a
# hardcoded default list, and the docstring — and the latter two had already
# drifted: `nasdaq` and `dow` were fetchable but undocumented, so a caller had
# no way to learn they existed. Derive, don't restate.

INDICATOR_TICKERS: dict[str, str] = {
    "treasury_10y": "^TNX",
    "treasury_2y":  "^IRX",
    "vix":          "^VIX",
    "dxy":          "DX-Y.NYB",
    "oil_wti":      "CL=F",
    "gold":         "GC=F",
    "sp500":        "^GSPC",
    "nasdaq":       "^IXIC",
    "dow":          "^DJI",
}

# Market proxies for series with no yfinance ticker. Each MUST carry a note
# saying what the number actually is — that is the entire reason this table is
# separate from INDICATOR_TICKERS rather than merged into it.
FRED_PROXIES: dict[str, str] = {
    "cpi":          "RINF",
    "unemployment": "^VXX",
}

PROXY_NOTES: dict[str, str] = {
    "cpi": (
        "PROXY, not the CPI. This is the market price of RINF (ProShares "
        "Inflation Expectations ETF), a tradeable read on expected inflation. "
        "It is not a CPI index level or an inflation rate. For the actual "
        "series use FRED CPIAUCSL."
    ),
    "unemployment": (
        "PROXY, and a poor one. This is the market price of ^VXX (a "
        "short-term VIX futures ETN): a risk-sentiment gauge with no "
        "unemployment content whatsoever. Do not read it as an unemployment "
        "rate. For the actual series use FRED UNRATE."
    ),
}

# Constants, with an env override. `note` is surfaced verbatim to the caller.
STATIC_INDICATORS: dict[str, tuple[str, str]] = {
    "fed_rate":   ("FED_RATE_OVERRIDE",   "5.25"),
    "gdp_growth": ("GDP_GROWTH_OVERRIDE", "2.8"),
}


def valid_indicators() -> list[str]:
    """Every name economics_data accepts, in a stable documented order."""
    return (
        list(INDICATOR_TICKERS)
        + list(FRED_PROXIES)
        + list(STATIC_INDICATORS)
    )


def resolve_indicator(name: str) -> tuple[str, str] | None:
    """Map a caller-supplied indicator name to ``(kind, key)``.

    Returns None when the name matches nothing — which is the caller's cue to
    report it rather than drop it. Lookup is case-insensitive and strips
    surrounding whitespace: an LLM writing the acronym as "CPI" is asking for
    the same series as "cpi", and silently failing that is the single most
    likely way to trigger the empty-result confusion this catalogue exists to
    prevent.
    """
    key = (name or "").strip().lower()
    if key in INDICATOR_TICKERS:
        return ("market", key)
    if key in FRED_PROXIES:
        return ("proxy", key)
    if key in STATIC_INDICATORS:
        return ("static", key)
    return None


# ---------------------------------------------------------------------------
# Price-frame hygiene
# ---------------------------------------------------------------------------
#
# yfinance emits a row for the session currently in progress — and sometimes for
# a halted or untraded one — whose OHLC are all NaN while Volume is a real
# number. `hist.empty` is False, `len(hist)` counts it, and `iloc[-1]` returns
# it, so every "just take the last bar" path silently reports NaN as the price.
#
# It is not a regional quirk. It is whichever market has an unsettled bar at the
# moment you ask: reported for SAP.DE/CPG.L on one day, reproduced for AAPL on
# another.
#
# The damage is worse than one wrong field, because `json.dumps` writes NaN as a
# bare `NaN` literal and RFC 8259 defines no such token. A strict parser — Qt's
# QJsonDocument on the C++ side included — rejects the WHOLE document, so a
# single hollow bar destroys the entire tool result rather than one number in it.


def _drop_hollow_bars(hist: Any) -> Any:
    """Drop rows with no usable Close from a yfinance history frame.

    Close is the anchor: every consumer here prices off it, and a row without
    one carries nothing worth keeping. Rows that merely lack an Open or a High
    survive — dropping those would discard a perfectly good close — and
    `_finite` keeps their gaps out of the JSON.
    """
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist:
        return hist
    return hist.dropna(subset=["Close"])


def _finite(value: Any, ndigits: int | None = None) -> float | None:
    """`float(value)`, or None if it is NaN/inf or not a number.

    None serialises to `null`, which is valid JSON and honestly says "no value".
    NaN serialises to a bare `NaN`, which does not parse. Never emit a raw float
    from market data into a response without passing it through here.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits) if ndigits is not None else f


# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------

def build_mcp_server() -> Any:
    """Build and return the FastMCP server instance."""
    if not MCP_SERVER_AVAILABLE:
        raise RuntimeError(
            "FastMCP not installed. Run: pip install mcp[cli] or pip install fastmcp"
        )

    mcp = FastMCP(
        name="fincept-tools",
        instructions=(
            "Financial data tools for quantitative research. "
            "Use market_data to get price history, financial_news for recent headlines, "
            "economics_data for macro indicators, and factor_backtest to evaluate factor expressions."
        ),
    )

    # ── Tool: market_data ────────────────────────────────────────────────────
    @mcp.tool()
    def market_data(
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        include_fundamentals: bool = False,
    ) -> dict[str, Any]:
        """
        Fetch OHLCV price history and current quote for a symbol.

        Args:
            symbol:               Ticker symbol (e.g. AAPL, 000001.SS, BTC-USD)
            period:               Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval:             Bar interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
            include_fundamentals: Include P/E, P/B, market cap, etc.

        Returns:
            dict with keys: symbol, period, bars (list of OHLCV dicts),
            latest_price, change_pct, volume, fundamentals (if requested)
        """
        if not YFINANCE_AVAILABLE:
            return {"error": "yfinance not installed. Run: pip install yfinance"}

        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return {"error": f"No data found for symbol {symbol!r}"}

            # Distinguished from "no data at all" on purpose: a window in which
            # every bar is hollow is a different problem from an unknown symbol,
            # and collapsing the two sends the caller hunting for a typo.
            hist = _drop_hollow_bars(hist)
            if hist.empty:
                return {"error": f"No usable bars for {symbol!r} (every bar in the window is empty)"}

            bars = []
            for ts, row in hist.iterrows():
                close = _finite(row["Close"], 4)
                if close is None:
                    continue  # belt and braces: _drop_hollow_bars already went first
                volume = _finite(row["Volume"])
                bars.append({
                    "date":   str(ts.date()) if hasattr(ts, "date") else str(ts),
                    "open":   _finite(row["Open"], 4),
                    "high":   _finite(row["High"], 4),
                    "low":    _finite(row["Low"], 4),
                    "close":  close,
                    # int(NaN) raises, so an unguarded cast turned a hollow
                    # volume into a failed call for the whole symbol.
                    "volume": int(volume) if volume is not None else None,
                })

            latest = bars[-1] if bars else {}
            prev   = bars[-2] if len(bars) > 1 else latest
            prev_close = prev.get("close")
            # `if prev_close` alone is not a NaN guard — bool(nan) is True, so the
            # old test waved NaN through and produced a NaN change_pct. Every
            # close reaching this point is finite by construction; the test now
            # only has to exclude a zero divisor.
            change_pct = (
                (latest["close"] - prev_close) / prev_close * 100
                if prev_close else 0.0
            )

            result: dict[str, Any] = {
                "symbol":       symbol.upper(),
                "period":       period,
                "interval":     interval,
                "bar_count":    len(bars),
                "latest_price": latest.get("close"),
                "change_pct":   round(change_pct, 3),
                "volume":       latest.get("volume"),
                "bars":         bars[-252:],  # cap at ~1 year of daily bars
            }

            if include_fundamentals:
                info = ticker.info
                result["fundamentals"] = {
                    "market_cap":      info.get("marketCap"),
                    "pe_ratio":        info.get("trailingPE"),
                    "pb_ratio":        info.get("priceToBook"),
                    "dividend_yield":  info.get("dividendYield"),
                    "52w_high":        info.get("fiftyTwoWeekHigh"),
                    "52w_low":         info.get("fiftyTwoWeekLow"),
                    "avg_volume":      info.get("averageVolume"),
                    "sector":          info.get("sector"),
                    "industry":        info.get("industry"),
                    "description":     (info.get("longBusinessSummary") or "")[:500],
                }

            return result

        except Exception as e:
            logger.exception("market_data(%s) failed", symbol)
            return {"error": str(e), "symbol": symbol}

    # ── Tool: financial_news ─────────────────────────────────────────────────
    @mcp.tool()
    def financial_news(
        query: str = "",
        symbol: str = "",
        limit: int = 20,
        days_back: int = 7,
    ) -> dict[str, Any]:
        """
        Fetch recent financial news headlines.

        Args:
            query:     Search keywords (e.g. "Fed interest rates", "NVDA earnings")
            symbol:    Ticker symbol to get news for (e.g. AAPL). Used if query is empty.
            limit:     Max number of articles to return (1-50)
            days_back: How many days back to search (1-30)

        Returns:
            dict with keys: articles (list), total, query_used
        """
        limit = max(1, min(50, limit))
        days_back = max(1, min(30, days_back))
        articles = []

        # Try yfinance news for symbol-specific queries
        if symbol and YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol.upper())
                news = ticker.news or []
                cutoff = datetime.now() - timedelta(days=days_back)
                for item in news[:limit]:
                    pub_ts = item.get("providerPublishTime", 0)
                    pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else None
                    if pub_dt and pub_dt < cutoff:
                        continue
                    articles.append({
                        "title":     item.get("title", ""),
                        "source":    item.get("publisher", ""),
                        "published": pub_dt.isoformat() if pub_dt else "",
                        "url":       item.get("link", ""),
                        "summary":   item.get("summary", ""),
                        "symbol":    symbol.upper(),
                    })
            except Exception as e:
                logger.warning("yfinance news for %s failed: %s", symbol, e)

        # Fallback: NewsAPI if key is configured
        if not articles and REQUESTS_AVAILABLE:
            api_key = os.environ.get("NEWS_API_KEY", "")
            if api_key:
                try:
                    params: dict[str, Any] = {
                        "apiKey":   api_key,
                        "pageSize": limit,
                        "language": "en",
                        "sortBy":   "publishedAt",
                        "from":     (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                    }
                    if query:
                        params["q"] = query
                    elif symbol:
                        params["q"] = symbol
                    else:
                        params["q"] = "stock market finance"

                    resp = requests.get(
                        "https://newsapi.org/v2/everything", params=params, timeout=10
                    )
                    if resp.ok:
                        for item in resp.json().get("articles", []):
                            articles.append({
                                "title":     item.get("title", ""),
                                "source":    item.get("source", {}).get("name", ""),
                                "published": item.get("publishedAt", ""),
                                "url":       item.get("url", ""),
                                "summary":   item.get("description", ""),
                            })
                except Exception as e:
                    logger.warning("NewsAPI failed: %s", e)

        return {
            "articles":   articles[:limit],
            "total":      len(articles),
            "query_used": query or symbol or "general financial news",
            "days_back":  days_back,
        }

    # ── Tool: economics_data ─────────────────────────────────────────────────
    @mcp.tool()
    def economics_data(
        indicators: list[str] | None = None,
        country: str = "US",
    ) -> dict[str, Any]:
        """
        Fetch macroeconomic indicators using yfinance proxies.

        Every value carries a "source" saying what the number actually is:
          market - the ticker IS the indicator (vix -> ^VIX).
          proxy  - a tradeable stand-in whose price is NOT the statistic. Also
                   flagged proxy=true with a note. Do not report it as the
                   series it stands in for.
          static - a constant, not fetched.

        Args:
            indicators: Names to fetch, case-insensitive. Valid names:
                        treasury_10y, treasury_2y, vix, dxy, oil_wti, gold,
                        sp500, nasdaq, dow (market); cpi, unemployment
                        (proxy); fed_rate, gdp_growth (static).
                        Defaults to all.
            country:    Country code. Only US has data; anything else still
                        returns US figures and says so in the response.

        Returns:
            dict with indicators -> {value, change_pct, date, ticker, source}.
            Names matching nothing are listed in unknown_indicators (with
            valid_indicators) rather than silently dropped.
        """
        if not YFINANCE_AVAILABLE:
            return {"error": "yfinance not installed. Run: pip install yfinance"}

        if indicators is None:
            indicators = valid_indicators()

        result: dict[str, Any] = {
            "country":   country,
            "timestamp": datetime.now().isoformat(),
        }
        # The country argument is accepted but every series here is US. Echoing
        # the request back unqualified would label US numbers as the caller's
        # country, so say plainly which one the data is.
        if (country or "").strip().upper() not in ("", "US", "USA"):
            result["data_country"] = "US"
            result["country_note"] = (
                f"No {country} data available; all values below are US. "
                "The country argument is not yet honoured."
            )

        data: dict[str, Any] = {}
        unknown: list[str] = []

        for ind in indicators:
            resolved = resolve_indicator(ind)
            if resolved is None:
                # Do NOT drop it. An unrecognised name and a genuinely empty
                # result used to be the same response ({"indicators": {}}),
                # which made a typo indistinguishable from unavailable data.
                unknown.append(ind)
                continue

            kind, key = resolved

            if kind == "static":
                env_var, default = STATIC_INDICATORS[key]
                data[key] = {
                    "value":  float(os.environ.get(env_var, default)),
                    "note":   f"Set {env_var} env var to override, or use FRED API",
                    "source": "static",
                }
                continue

            ticker_sym = (
                INDICATOR_TICKERS[key] if kind == "market" else FRED_PROXIES[key]
            )
            try:
                ticker = yf.Ticker(ticker_sym)
                hist = ticker.history(period="5d", interval="1d")
                if hist.empty:
                    data[key] = {"error": f"No data for {ticker_sym}", "ticker": ticker_sym}
                    continue

                # Before this, a hollow last bar made the guard below report the
                # indicator as unavailable while a perfectly good close sat one
                # row up. Dropping first turns that error into the right number.
                hist = _drop_hollow_bars(hist)
                if hist.empty:
                    data[key] = {
                        "error":  f"No usable close for {ticker_sym} (every bar is empty)",
                        "ticker": ticker_sym,
                    }
                    continue

                latest_close = float(hist["Close"].iloc[-1])
                # A non-empty frame can still carry a hollow last bar, and the
                # NaN that produces is not merely wrong — json.dumps writes it
                # as bare `NaN`, which is not JSON (RFC 8259 has no such
                # literal). A strict client — Qt's QJsonDocument on the C++
                # side included — then fails to parse the WHOLE tool result,
                # so one stale ETF takes down every other indicator with it.
                if latest_close != latest_close:  # NaN
                    data[key] = {
                        "error":  f"No usable close for {ticker_sym} (latest bar is empty)",
                        "ticker": ticker_sym,
                    }
                    continue

                prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest_close
                if prev_close != prev_close:      # NaN
                    prev_close = latest_close
                change_pct = (latest_close - prev_close) / prev_close * 100 if prev_close else 0

                entry: dict[str, Any] = {
                    "value":      round(latest_close, 4),
                    "change_pct": round(change_pct, 3),
                    "date":       str(hist.index[-1].date()),
                    "ticker":     ticker_sym,
                    "source":     "market" if kind == "market" else "proxy",
                }
                if kind == "proxy":
                    # The value is a share price, not the statistic. Say so on
                    # the value itself — a note filed elsewhere in the response
                    # is a note the reader of this number will not see.
                    entry["proxy"] = True
                    entry["note"] = PROXY_NOTES[key]
                data[key] = entry
            except Exception as e:
                data[key] = {"error": str(e), "ticker": ticker_sym}

        result["indicators"] = data
        if unknown:
            result["unknown_indicators"] = unknown
            result["valid_indicators"] = valid_indicators()
            result["error"] = (
                "Unrecognised indicator(s): "
                + ", ".join(repr(u) for u in unknown)
                + ". Valid ids: "
                + ", ".join(valid_indicators())
            )
        return result

    # ── Tool: factor_backtest ────────────────────────────────────────────────
    @mcp.tool()
    def factor_backtest(
        symbol: str,
        factor_expr: str,
        period: str = "2y",
        top_pct: float = 0.2,
    ) -> dict[str, Any]:
        """
        Quick IC/Sharpe estimate for a factor expression on a single symbol.

        Computes the factor value for each bar using the expression, then
        calculates next-period return rank correlation (IC) and a simple
        long-top-decile strategy Sharpe.

        Args:
            symbol:      Ticker symbol to test on
            factor_expr: Python expression using columns: open, high, low, close,
                         volume, returns. E.g. "close / close.rolling(20).mean() - 1"
            period:      Data period (1y, 2y, 5y)
            top_pct:     Top percentile to go long (0.1 = top 10%)

        Returns:
            dict with ic, ic_ir, sharpe, win_rate, max_drawdown
        """
        if not YFINANCE_AVAILABLE:
            return {"error": "yfinance not installed"}
        try:
            import pandas as pd
            import numpy as np

            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period, interval="1d")
            # Before the guard, not after: a hollow bar was counted toward the
            # 60-bar minimum, and it also entered `returns`, where pct_change
            # spread its NaN to the neighbouring row and cost the most recent
            # real observation its forward return. A hollow bar in the interior
            # additionally poisons a full window of any rolling factor.
            hist = _drop_hollow_bars(hist)
            if len(hist) < 60:
                return {"error": f"Insufficient data for {symbol} ({len(hist)} usable bars)"}

            df = pd.DataFrame({
                "open":   hist["Open"],
                "high":   hist["High"],
                "low":    hist["Low"],
                "close":  hist["Close"],
                "volume": hist["Volume"],
            })
            df["returns"] = df["close"].pct_change()

            # Evaluate factor expression — allowlist-checked at parse time,
            # see _compile_factor_expr above.
            try:
                code = _compile_factor_expr(factor_expr)
            except ValueError as e:
                return {"error": f"Factor expression rejected: {e}"}
            try:
                factor_vals = eval(  # noqa: S307
                    code,
                    {"__builtins__": {}},
                    {**{col: df[col] for col in df.columns},
                     "pd": pd, "np": np},
                )
            except Exception as e:
                return {"error": f"Factor expression error: {e}"}

            df["factor"] = factor_vals
            df["fwd_ret"] = df["returns"].shift(-1)
            df = df.dropna()

            if len(df) < 20:
                return {"error": "Too few valid bars after factor computation"}

            # IC = rank correlation between factor and forward return
            ic_series = df["factor"].rolling(20).corr(df["fwd_ret"])
            ic = float(ic_series.mean())
            ic_ir = float(ic / ic_series.std()) if ic_series.std() > 0 else 0.0

            # Simple long-top strategy
            threshold = df["factor"].quantile(1 - top_pct)
            long_mask = df["factor"] >= threshold
            strategy_ret = df["fwd_ret"].where(long_mask, 0.0)
            ann = 252 ** 0.5
            sharpe = float(strategy_ret.mean() / strategy_ret.std() * ann) if strategy_ret.std() > 0 else 0.0
            win_rate = float((strategy_ret[long_mask] > 0).mean()) if long_mask.sum() > 0 else 0.0

            cum = (1 + strategy_ret).cumprod()
            rolling_max = cum.cummax()
            drawdown = (cum - rolling_max) / rolling_max
            max_dd = float(drawdown.min())

            return {
                "symbol":       symbol.upper(),
                "factor_expr":  factor_expr,
                "period":       period,
                "bar_count":    len(df),
                "ic":           round(ic, 4),
                "ic_ir":        round(ic_ir, 4),
                "sharpe":       round(sharpe, 3),
                "win_rate":     round(win_rate, 3),
                "max_drawdown": round(max_dd, 3),
                "long_bars":    int(long_mask.sum()),
            }
        except Exception as e:
            logger.exception("factor_backtest(%s) failed", symbol)
            return {"error": str(e)}

    # ── Tool: symbol_search ──────────────────────────────────────────────────
    @mcp.tool()
    def symbol_search(query: str, limit: int = 10) -> dict[str, Any]:
        """
        Search for ticker symbols matching a company name or keyword.

        Args:
            query: Company name or keyword (e.g. "Apple", "semiconductor ETF")
            limit: Max results to return (1-20)

        Returns:
            dict with results list of {symbol, name, exchange, type}
        """
        if not YFINANCE_AVAILABLE:
            return {"error": "yfinance not installed"}
        try:
            results = yf.Search(query, max_results=min(20, limit))
            quotes = results.quotes if hasattr(results, "quotes") else []
            return {
                "query":   query,
                "results": [
                    {
                        "symbol":   q.get("symbol", ""),
                        "name":     q.get("shortname") or q.get("longname", ""),
                        "exchange": q.get("exchange", ""),
                        "type":     q.get("quoteType", ""),
                    }
                    for q in quotes[:limit]
                ],
                "total": len(quotes),
            }
        except Exception as e:
            return {"error": str(e), "query": query}

    return mcp


# ---------------------------------------------------------------------------
# Entry point — run as standalone HTTP server
# ---------------------------------------------------------------------------

def _serve(server: Any, host: str, port: int, path: str) -> None:
    """Start `server` on streamable-HTTP, whichever FastMCP we ended up with.

    There are three live shapes for this and they are mutually incompatible:

      1. standalone ``fastmcp``
             run(transport, host=..., port=..., path=...)
      2. official ``mcp`` SDK, FastMCP era (>=1.26, incl. 1.29)
             run(transport, mount_path=None)   <- host/port/path NOT accepted
             host, port and streamable_http_path are Settings fields
      3. official ``mcp`` SDK, MCPServer era
             run(transport, **kwargs) -> forwarded to run_streamable_http_async,
             where the path keyword is ``streamable_http_path``, not ``path``

    This used to hardcode shape 1 while the import block preferred shape 2, so
    `pip install mcp` — the install this repo's own error messages tell you to
    do — produced a server that raised ``TypeError: FastMCP.run() got an
    unexpected keyword argument 'host'`` on every start and never listened.
    It only worked if ``fastmcp`` was installed *and* ``mcp`` was not.

    Rather than pick a package and hope, set the settings when the object has
    them and pass the keywords when ``run`` actually accepts them. Shapes 2 and
    3 each ignore the half that does not apply to them.
    """
    import inspect

    # Shape 2 reads these at run() time. hasattr-guarded because the standalone
    # package's settings object does not carry the same field names.
    settings = getattr(server, "settings", None)
    if settings is not None:
        for attr, value in (("host", host), ("port", port), ("streamable_http_path", path)):
            if hasattr(settings, attr):
                setattr(settings, attr, value)

    params = inspect.signature(server.run).parameters
    takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    if "host" in params or takes_kwargs:
        # Shapes 1 and 3. They disagree on the path keyword, and passing the
        # wrong one is a TypeError, so it is chosen by which package we imported
        # rather than guessed.
        path_kw = "path" if MCP_IMPL == "fastmcp" else "streamable_http_path"
        server.run(transport="streamable-http", host=host, port=port, **{path_kw: path})
        return

    # Shape 2: everything travelled via settings above.
    server.run(transport="streamable-http")


def _selftest() -> int:
    """Offline checks for the tool contracts. No network, no API key.

    Guards the two regressions from issue #382 — an unknown indicator name
    silently producing the same response as an empty result, and a proxy value
    presented as if it were the statistic — plus the NaN-is-not-JSON hazard
    that one stale ETF bar introduces into every other indicator's result.

    Section [4] guards issue #380: yfinance emits a bar for the unsettled
    session with all-NaN OHLC, and every "take the last bar" path reported that
    NaN as the price. Live data cannot be relied on to contain a hollow bar on
    any given day, so those frames are synthetic and pin the exact shape
    yfinance produces — all-NaN OHLC beside a real Volume.
    """
    failures: list[str] = []

    def check(label: str, ok: bool) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    print("[1] indicator catalogue")
    names = valid_indicators()
    check("no duplicate names", len(names) == len(set(names)))
    check("every proxy has a note", set(FRED_PROXIES) <= set(PROXY_NOTES))
    check("proxy and market tables are disjoint",
          not (set(FRED_PROXIES) & set(INDICATOR_TICKERS)))
    check("catalogue covers all three kinds",
          {resolve_indicator(n)[0] for n in names} == {"market", "proxy", "static"})

    print("[2] name resolution")
    check("known name resolves", resolve_indicator("vix") == ("market", "vix"))
    check("case/space insensitive", resolve_indicator("  CPI ") == ("proxy", "cpi"))
    check("unknown name returns None", resolve_indicator("banana") is None)
    check("near-miss 'gdp' is NOT silently accepted", resolve_indicator("gdp") is None)

    if not MCP_SERVER_AVAILABLE or not YFINANCE_AVAILABLE:
        print("[3] tool contract: SKIPPED (needs mcp + yfinance installed)")
        print(f"\n{len(failures)} failure(s)")
        return 1 if failures else 0

    print("[3] tool contract (offline paths only)")
    fn = build_mcp_server()._tool_manager._tools["economics_data"].fn
    typo, nonsense = fn(indicators=["gdp"]), fn(indicators=["banana"])
    check("a typo is reported, not dropped", typo.get("unknown_indicators") == ["gdp"])
    check("typo and nonsense are distinguishable",
          typo.get("unknown_indicators") != nonsense.get("unknown_indicators"))
    check("the correct name is discoverable from the response",
          "gdp_growth" in typo.get("valid_indicators", []))
    static_only = fn(indicators=["fed_rate", "gdp_growth"])
    check("static values still labelled source=static",
          all(v.get("source") == "static" for v in static_only["indicators"].values()))
    try:
        json.dumps(typo, allow_nan=False)
        json.dumps(static_only, allow_nan=False)
        check("responses are RFC-8259 valid JSON (no bare NaN)", True)
    except ValueError as e:
        check(f"responses are RFC-8259 valid JSON (no bare NaN): {e}", False)

    print("[4] hollow bars (issue #380)")
    import pandas as pd

    nan = float("nan")

    def frame(closes: list, *, hollow_volume: bool = False):
        """A yfinance-shaped daily frame. A NaN close marks a hollow bar."""
        return pd.DataFrame(
            {
                "Open": closes, "High": closes, "Low": closes, "Close": closes,
                # Volume is real even on a hollow bar — that is precisely why
                # the NaN slipped through instead of raising.
                "Volume": [nan if (hollow_volume and c != c) else 1_000_000.0 for c in closes],
            },
            index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
        )

    check("hollow trailing bar dropped", len(_drop_hollow_bars(frame([1.0, 2.0, nan]))) == 2)
    check("interior hollow bar dropped", len(_drop_hollow_bars(frame([1.0, nan, 3.0]))) == 2)
    check("clean frame untouched", len(_drop_hollow_bars(frame([1.0, 2.0]))) == 2)
    check("_finite rejects NaN and inf, keeps and rounds reals",
          _finite(nan) is None and _finite(float("inf")) is None and _finite(1.23456, 2) == 1.23)

    tools = build_mcp_server()._tool_manager._tools
    market_fn, factor_fn = tools["market_data"].fn, tools["factor_backtest"].fn

    def fake_yf(df):
        class _Ticker:
            info: dict = {}

            def history(self, **_kw):
                return df.copy()

        return type("yf", (), {"Ticker": staticmethod(lambda _sym: _Ticker())})

    global yf
    real_yf = yf
    try:
        yf = fake_yf(frame([100.0, 189.88, nan]))
        got = market_fn(symbol="SAP.DE", period="5d", interval="1d")
        check("latest_price is the last VALID close, not NaN", got.get("latest_price") == 189.88)
        check("the hollow bar is not counted", got.get("bar_count") == 2)
        try:
            json.dumps(got, allow_nan=False)
            check("market_data is RFC-8259 valid JSON", True)
        except ValueError as e:
            check(f"market_data is RFC-8259 valid JSON: {e}", False)

        # int(NaN) raises, so an unguarded volume cast used to fail the call.
        yf = fake_yf(frame([100.0, 101.0, nan], hollow_volume=True))
        got = market_fn(symbol="X", period="5d", interval="1d")
        check("a fully hollow bar does not error the whole call",
              "error" not in got and got.get("latest_price") == 101.0)

        yf = fake_yf(frame([nan, nan]))
        got = market_fn(symbol="Y", period="5d", interval="1d")
        check("an all-hollow window says so, rather than 'symbol not found'",
              "every bar in the window is empty" in str(got.get("error")))

        # 59 real bars plus a hollow one totalled 60 and slipped past the
        # sufficiency guard, which then returned confident statistics computed
        # on 58 rows.
        yf = fake_yf(frame([100.0 + i * 0.5 for i in range(59)] + [nan]))
        got = factor_fn(symbol="S", factor_expr="close", period="1y")
        check("59 real + 1 hollow is refused, and counted honestly",
              "(59 usable bars)" in str(got.get("error")))
        yf = fake_yf(frame([100.0 + i * 0.5 for i in range(60)] + [nan]))
        got = factor_fn(symbol="S", factor_expr="close", period="1y")
        check("60 real + 1 hollow still runs", "error" not in got)
    finally:
        yf = real_yf

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fincept MCP server for RD-Agent")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--selftest", action="store_true",
                        help="Run offline contract checks and exit")
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    if not MCP_SERVER_AVAILABLE:
        print(json.dumps({"error": "mcp[cli] not installed. Run: pip install mcp[cli]"}))
        raise SystemExit(1)

    server = build_mcp_server()
    # stderr, not stdout: stdout is the JSON-RPC channel for MCP's stdio
    # transport, so a banner printed there corrupts the stream for any client
    # that speaks it. Harmless over HTTP, fatal over stdio, and free to get
    # right. mcp_tools.start_mcp_server_process pipes both, so the line is
    # still captured either way.
    print(
        f"Fincept MCP server starting on http://{args.host}:{args.port}/mcp",
        file=sys.stderr,
        flush=True,
    )
    _serve(server, args.host, args.port, "/mcp")


if __name__ == "__main__":
    main()
