"""Financial domain tools for Duxx AI agents.

Provides tools for stock prices, market analysis, portfolio metrics,
currency conversion, and financial ratio calculations. Uses yfinance
(lazy-imported) for market data.

Required dependencies:
    pip install yfinance
"""

from __future__ import annotations

import json
from typing import Any

from duxx_ai.core.tool import Tool, tool

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="stock_price",
    description="Get current or historical stock price data for a ticker symbol.",
    tags=["financial", "market"],
)
def stock_price(symbol: str, period: str = "1d") -> str:
    """Fetch stock price data using yfinance.

    Args:
        symbol: Ticker symbol (e.g. AAPL, GOOGL, MSFT).
        period: Time period -- 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.

    Returns:
        JSON-formatted price data or error message.
    """
    try:
        import yfinance as yf
    except ImportError:
        return (
            "Error: yfinance is not installed. "
            "Install it with: pip install yfinance"
        )

    if not symbol or not symbol.strip():
        return "Error: ticker symbol is required."

    valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in valid_periods:
        return f"Error: invalid period '{period}'. Use one of: {', '.join(sorted(valid_periods))}"

    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)

        if hist.empty:
            return f"No data found for symbol '{symbol}'. Verify the ticker is correct."

        # Get current info
        info = ticker.info or {}
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        # Format history
        records = []
        for date, row in hist.tail(10).iterrows():
            records.append({
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "open": round(row.get("Open", 0), 2),
                "high": round(row.get("High", 0), 2),
                "low": round(row.get("Low", 0), 2),
                "close": round(row.get("Close", 0), 2),
                "volume": int(row.get("Volume", 0)),
            })

        result = {
            "symbol": symbol.upper(),
            "name": info.get("shortName", symbol),
            "currency": info.get("currency", "USD"),
            "current_price": current_price,
            "period": period,
            "data_points": len(hist),
            "recent_history": records,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error fetching stock data: {type(e).__name__}: {e}"


@tool(
    name="market_analysis",
    description="Get comparative market data and analysis for multiple symbols.",
    tags=["financial", "market", "analysis"],
)
def market_analysis(symbols: str, metrics: str = "price,volume") -> str:
    """Analyse multiple stocks/assets and compare key metrics.

    Args:
        symbols: Comma-separated ticker symbols (e.g. AAPL,GOOGL,MSFT).
        metrics: Comma-separated metrics to include --
            price, volume, pe_ratio, market_cap, dividend_yield, beta.

    Returns:
        JSON-formatted comparative analysis.
    """
    try:
        import yfinance as yf
    except ImportError:
        return (
            "Error: yfinance is not installed. "
            "Install it with: pip install yfinance"
        )

    if not symbols or not symbols.strip():
        return "Error: at least one symbol is required."

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    metric_list = [m.strip().lower() for m in metrics.split(",") if m.strip()]

    results: list[dict[str, Any]] = []

    for sym in symbol_list:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            _hist = ticker.history(period="5d")  # reserved for trailing-window metrics

            entry: dict[str, Any] = {"symbol": sym, "name": info.get("shortName", sym)}

            if "price" in metric_list:
                entry["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
                entry["previous_close"] = info.get("previousClose")
                if entry["current_price"] and entry["previous_close"]:
                    change = entry["current_price"] - entry["previous_close"]
                    pct = (change / entry["previous_close"]) * 100
                    entry["price_change"] = round(change, 2)
                    entry["price_change_pct"] = round(pct, 2)

            if "volume" in metric_list:
                entry["volume"] = info.get("volume") or info.get("regularMarketVolume")
                entry["avg_volume"] = info.get("averageVolume")

            if "pe_ratio" in metric_list:
                entry["pe_trailing"] = info.get("trailingPE")
                entry["pe_forward"] = info.get("forwardPE")

            if "market_cap" in metric_list:
                entry["market_cap"] = info.get("marketCap")

            if "dividend_yield" in metric_list:
                entry["dividend_yield"] = info.get("dividendYield")
                if entry["dividend_yield"]:
                    entry["dividend_yield"] = round(entry["dividend_yield"] * 100, 2)

            if "beta" in metric_list:
                entry["beta"] = info.get("beta")

            results.append(entry)
        except Exception as e:
            results.append({"symbol": sym, "error": f"{type(e).__name__}: {e}"})

    return json.dumps({"analysis": results, "metrics_requested": metric_list}, indent=2)


@tool(
    name="portfolio_metrics",
    description="Calculate portfolio performance metrics given holdings.",
    tags=["financial", "portfolio"],
)
def portfolio_metrics(holdings: str, benchmark: str = "SPY") -> str:
    """Calculate portfolio metrics from a JSON holdings specification.

    Args:
        holdings: JSON-encoded list of holdings, each with 'symbol', 'shares',
            and optionally 'cost_basis'. Example:
            [{"symbol":"AAPL","shares":10,"cost_basis":150.00}]
        benchmark: Benchmark ticker for comparison (default: SPY).

    Returns:
        JSON-formatted portfolio summary and metrics.
    """
    try:
        import yfinance as yf
    except ImportError:
        return (
            "Error: yfinance is not installed. "
            "Install it with: pip install yfinance"
        )

    try:
        holding_list = json.loads(holdings)
    except json.JSONDecodeError as e:
        return f"Error: invalid holdings JSON -- {e}"

    if not isinstance(holding_list, list) or not holding_list:
        return "Error: holdings must be a non-empty JSON array."

    portfolio_data: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0

    for h in holding_list:
        sym = h.get("symbol", "").upper()
        shares = h.get("shares", 0)
        cost_basis = h.get("cost_basis")

        if not sym or shares <= 0:
            continue

        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

            value = price * shares
            total_value += value

            entry: dict[str, Any] = {
                "symbol": sym,
                "shares": shares,
                "current_price": price,
                "market_value": round(value, 2),
            }

            if cost_basis is not None:
                cost = cost_basis * shares
                total_cost += cost
                gain = value - cost
                gain_pct = (gain / cost) * 100 if cost > 0 else 0
                entry["cost_basis"] = cost_basis
                entry["total_cost"] = round(cost, 2)
                entry["unrealized_gain"] = round(gain, 2)
                entry["unrealized_gain_pct"] = round(gain_pct, 2)

            portfolio_data.append(entry)
        except Exception as e:
            portfolio_data.append({"symbol": sym, "error": str(e)})

    # Add allocation weights
    for entry in portfolio_data:
        if "market_value" in entry and total_value > 0:
            entry["weight_pct"] = round((entry["market_value"] / total_value) * 100, 2)

    result: dict[str, Any] = {
        "total_market_value": round(total_value, 2),
        "holdings": portfolio_data,
        "benchmark": benchmark,
    }

    if total_cost > 0:
        result["total_cost"] = round(total_cost, 2)
        result["total_gain"] = round(total_value - total_cost, 2)
        result["total_return_pct"] = round(((total_value - total_cost) / total_cost) * 100, 2)

    return json.dumps(result, indent=2)


@tool(
    name="currency_convert",
    description="Convert between currencies using live exchange rates.",
    tags=["financial", "currency"],
)
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another.

    Uses yfinance forex pairs for live rates. Common currency codes:
    USD, EUR, GBP, JPY, CAD, AUD, CHF, CNY, INR, etc.

    Args:
        amount: Amount to convert.
        from_currency: Source currency code (e.g. USD).
        to_currency: Target currency code (e.g. EUR).

    Returns:
        Conversion result with rate.
    """
    try:
        import yfinance as yf
    except ImportError:
        return (
            "Error: yfinance is not installed. "
            "Install it with: pip install yfinance"
        )

    from_c = from_currency.strip().upper()
    to_c = to_currency.strip().upper()

    if from_c == to_c:
        return json.dumps({
            "amount": amount,
            "from": from_c,
            "to": to_c,
            "rate": 1.0,
            "converted": amount,
        }, indent=2)

    # yfinance forex pair format: EURUSD=X
    pair = f"{from_c}{to_c}=X"

    try:
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="1d")

        if hist.empty:
            return (
                f"Error: could not find exchange rate for {from_c}/{to_c}. "
                f"Verify the currency codes are correct."
            )

        rate = float(hist["Close"].iloc[-1])
        converted = round(amount * rate, 4)

        return json.dumps({
            "amount": amount,
            "from": from_c,
            "to": to_c,
            "rate": round(rate, 6),
            "converted": converted,
            "pair": pair,
        }, indent=2)

    except Exception as e:
        return f"Error converting currency: {type(e).__name__}: {e}"


@tool(
    name="financial_ratios",
    description="Calculate common financial ratios from provided figures.",
    tags=["financial", "analysis"],
)
def financial_ratios(
    revenue: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
    total_debt: float,
) -> str:
    """Calculate key financial ratios from input financial figures.

    All values should be in the same currency and units (e.g. millions USD).

    Args:
        revenue: Total revenue / sales.
        net_income: Net income / profit.
        total_assets: Total assets.
        total_equity: Total shareholders' equity.
        total_debt: Total debt / liabilities.

    Returns:
        JSON with calculated financial ratios.
    """
    ratios: dict[str, Any] = {}

    # Profitability
    if revenue != 0:
        ratios["net_profit_margin"] = round((net_income / revenue) * 100, 2)
    else:
        ratios["net_profit_margin"] = None

    if total_assets != 0:
        ratios["return_on_assets"] = round((net_income / total_assets) * 100, 2)
    else:
        ratios["return_on_assets"] = None

    if total_equity != 0:
        ratios["return_on_equity"] = round((net_income / total_equity) * 100, 2)
    else:
        ratios["return_on_equity"] = None

    # Leverage
    if total_equity != 0:
        ratios["debt_to_equity"] = round(total_debt / total_equity, 4)
    else:
        ratios["debt_to_equity"] = None

    if total_assets != 0:
        ratios["debt_to_assets"] = round(total_debt / total_assets, 4)
    else:
        ratios["debt_to_assets"] = None

    # Efficiency
    if total_assets != 0:
        ratios["asset_turnover"] = round(revenue / total_assets, 4)
    else:
        ratios["asset_turnover"] = None

    if total_equity != 0:
        ratios["equity_multiplier"] = round(total_assets / total_equity, 4)
    else:
        ratios["equity_multiplier"] = None

    result = {
        "inputs": {
            "revenue": revenue,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_equity": total_equity,
            "total_debt": total_debt,
        },
        "ratios": ratios,
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_TOOLS: dict[str, Tool] = {
    "stock_price": stock_price,
    "market_analysis": market_analysis,
    "portfolio_metrics": portfolio_metrics,
    "currency_convert": currency_convert,
    "financial_ratios": financial_ratios,
}


def get_financial_tools(names: list[str] | None = None) -> list[Tool]:
    """Get financial tools by name. If names is None, return all."""
    if names is None:
        return list(MODULE_TOOLS.values())
    return [MODULE_TOOLS[n] for n in names if n in MODULE_TOOLS]


try:
    from duxx_ai.tools.registry import register_domain

    register_domain("financial", MODULE_TOOLS)
except ImportError:
    pass
