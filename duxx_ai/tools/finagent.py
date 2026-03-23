"""Financial Financial Data Tools for Duxx AI Agents.

Provides agent-ready tools for financial data access powered by Financial's
Open Data Platform. Covers stocks, crypto, forex, options, economy, and more.

Requires: pip install finagent

Usage:
    from duxx_ai.tools.finagent import get_financial_tools, FinancialAgent

    # Get all financial tools
    tools = get_financial_tools()

    # Or create a ready-made financial agent
    agent = FinancialAgent.create(provider="yfinance")
    result = await agent.run("Analyze AAPL stock performance")
"""

from __future__ import annotations

import logging
from typing import Any

from duxx_ai.core.tool import Tool, ToolParameter

logger = logging.getLogger(__name__)


def _safe_finagent_call(fn_path: str, **kwargs: Any) -> str:
    """Safely call an Financial function and return formatted result."""
    try:
        from finagent import obb
        # Navigate to the function
        parts = fn_path.split(".")
        obj = obb
        for part in parts:
            obj = getattr(obj, part)
        result = obj(**kwargs)
        if hasattr(result, "to_dataframe"):
            df = result.to_dataframe()
            return df.to_string(max_rows=20)
        return str(result.results) if hasattr(result, "results") else str(result)
    except ImportError:
        return f"[Financial not installed. Run: pip install finagent]\nWould call: obb.{fn_path}({kwargs})"
    except Exception as e:
        return f"Error calling obb.{fn_path}: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Individual Financial Tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _create_stock_price_tool() -> Tool:
    tool = Tool(
        name="stock_price",
        description="Get historical stock price data (OHLCV) for any ticker symbol. Returns date, open, high, low, close, volume.",
        parameters=[
            ToolParameter(name="symbol", type="string", description="Stock ticker (e.g. AAPL, TSLA, MSFT)", required=True),
            ToolParameter(name="period", type="string", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max", required=False, default="3mo"),
        ],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.price.historical", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_company_profile_tool() -> Tool:
    tool = Tool(
        name="company_profile",
        description="Get company profile information — sector, industry, employees, description, market cap, CEO, website.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.profile", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_financial_metrics_tool() -> Tool:
    tool = Tool(
        name="financial_metrics",
        description="Get key financial metrics — P/E ratio, EPS, market cap, revenue, profit margins, ROE, debt ratios.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.fundamental.metrics", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_income_statement_tool() -> Tool:
    tool = Tool(
        name="income_statement",
        description="Get income statement data — revenue, cost of goods, gross profit, operating income, net income, EPS.",
        parameters=[
            ToolParameter(name="symbol", type="string", description="Stock ticker", required=True),
            ToolParameter(name="period", type="string", description="annual or quarter", required=False, default="annual"),
        ],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.fundamental.income", symbol=args.get("symbol", "AAPL"), period=args.get("period", "annual"))
    tool.bind(_execute)
    return tool


def _create_balance_sheet_tool() -> Tool:
    tool = Tool(
        name="balance_sheet",
        description="Get balance sheet data — total assets, liabilities, equity, cash, debt, inventory.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.fundamental.balance", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_cash_flow_tool() -> Tool:
    tool = Tool(
        name="cash_flow",
        description="Get cash flow statement — operating, investing, financing cash flows, free cash flow, capex.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.fundamental.cash", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_stock_screener_tool() -> Tool:
    tool = Tool(
        name="stock_screener",
        description="Screen stocks by criteria — market cap, sector, P/E ratio, dividend yield, etc. Returns matching tickers.",
        parameters=[
            ToolParameter(name="sector", type="string", description="Sector filter (e.g. Technology, Healthcare)", required=False),
            ToolParameter(name="market_cap_min", type="number", description="Minimum market cap in billions", required=False),
        ],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        return _safe_finagent_call("equity.screener")
    tool.bind(_execute)
    return tool


def _create_market_news_tool() -> Tool:
    tool = Tool(
        name="market_news",
        description="Get latest financial news — market news, company-specific news, analyst reports.",
        parameters=[
            ToolParameter(name="symbol", type="string", description="Stock ticker for company news (optional)", required=False),
            ToolParameter(name="limit", type="integer", description="Number of articles", required=False, default=10),
        ],
        tags=["finance", "finagent", "news"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        symbol = args.get("symbol")
        if symbol:
            return _safe_finagent_call("news.company", symbol=symbol, limit=args.get("limit", 10))
        return _safe_finagent_call("news.world", limit=args.get("limit", 10))
    tool.bind(_execute)
    return tool


def _create_crypto_price_tool() -> Tool:
    tool = Tool(
        name="crypto_price",
        description="Get cryptocurrency price data — BTC, ETH, and 1000+ altcoins with OHLCV history.",
        parameters=[ToolParameter(name="symbol", type="string", description="Crypto pair (e.g. BTC-USD, ETH-USD)", required=True)],
        tags=["finance", "finagent", "crypto"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("crypto.price.historical", symbol=args.get("symbol", "BTC-USD"))
    tool.bind(_execute)
    return tool


def _create_forex_rate_tool() -> Tool:
    tool = Tool(
        name="forex_rate",
        description="Get foreign exchange rates — currency pairs like EUR/USD, GBP/JPY, etc.",
        parameters=[ToolParameter(name="symbol", type="string", description="Currency pair (e.g. EUR/USD)", required=True)],
        tags=["finance", "finagent", "forex"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("currency.price.historical", symbol=args.get("symbol", "EUR/USD"))
    tool.bind(_execute)
    return tool


def _create_economic_indicator_tool() -> Tool:
    tool = Tool(
        name="economic_indicator",
        description="Get macroeconomic data — GDP, inflation (CPI), unemployment, interest rates, consumer sentiment.",
        parameters=[
            ToolParameter(name="indicator", type="string", description="Indicator: gdp, cpi, unemployment, interest_rate, pmi", required=True),
            ToolParameter(name="country", type="string", description="Country code (e.g. US, GB, JP)", required=False, default="US"),
        ],
        tags=["finance", "finagent", "economy"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        indicator = args.get("indicator", "gdp")
        routes = {"gdp": "economy.gdp.nominal", "cpi": "economy.cpi", "unemployment": "economy.unemployment", "interest_rate": "economy.interest_rate"}
        route = routes.get(indicator, f"economy.{indicator}")
        return _safe_finagent_call(route, country=args.get("country", "US"))
    tool.bind(_execute)
    return tool


def _create_options_chain_tool() -> Tool:
    tool = Tool(
        name="options_chain",
        description="Get options chain data — calls and puts with strike prices, premiums, Greeks, open interest.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "options"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("derivatives.options.chains", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_earnings_calendar_tool() -> Tool:
    tool = Tool(
        name="earnings_calendar",
        description="Get upcoming earnings reports — dates, estimated EPS, actual EPS, surprise percentage.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker (optional for all upcoming)", required=False)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        symbol = args.get("symbol")
        if symbol:
            return _safe_finagent_call("equity.calendar.earnings", symbol=symbol)
        return _safe_finagent_call("equity.calendar.earnings")
    tool.bind(_execute)
    return tool


def _create_analyst_estimates_tool() -> Tool:
    tool = Tool(
        name="analyst_estimates",
        description="Get analyst consensus estimates — price targets, buy/sell ratings, revenue forecasts.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.estimates.consensus", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_market_index_tool() -> Tool:
    tool = Tool(
        name="market_index",
        description="Get market index data — S&P 500, NASDAQ, Dow Jones, Russell 2000, FTSE, Nikkei, etc.",
        parameters=[ToolParameter(name="symbol", type="string", description="Index symbol (e.g. ^GSPC, ^IXIC, ^DJI)", required=True)],
        tags=["finance", "finagent", "index"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("index.price.historical", symbol=args.get("symbol", "^GSPC"))
    tool.bind(_execute)
    return tool


def _create_commodity_price_tool() -> Tool:
    tool = Tool(
        name="commodity_price",
        description="Get commodity prices — gold, silver, oil (WTI/Brent), natural gas, copper, wheat, corn.",
        parameters=[ToolParameter(name="symbol", type="string", description="Commodity symbol (e.g. GC=F for gold, CL=F for oil)", required=True)],
        tags=["finance", "finagent", "commodities"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.price.historical", symbol=args.get("symbol", "GC=F"))
    tool.bind(_execute)
    return tool


def _create_sec_filings_tool() -> Tool:
    tool = Tool(
        name="sec_filings",
        description="Get SEC filings — 10-K, 10-Q, 8-K annual/quarterly reports and disclosures.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "regulatory"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.fundamental.filings", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_insider_trading_tool() -> Tool:
    tool = Tool(
        name="insider_trading",
        description="Get insider trading activity — executive buys/sells, transaction amounts, filing dates.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.ownership.insider_trading", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


def _create_dividend_tool() -> Tool:
    tool = Tool(
        name="dividends",
        description="Get dividend history and upcoming dividends — payment dates, amounts, yield, ex-dates.",
        parameters=[ToolParameter(name="symbol", type="string", description="Stock ticker", required=True)],
        tags=["finance", "finagent", "equity"],
    )
    async def _execute(call: Any) -> str:
        args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        return _safe_finagent_call("equity.fundamental.dividends", symbol=args.get("symbol", "AAPL"))
    tool.bind(_execute)
    return tool


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_FINANCIAL_TOOLS = {
    "stock_price": _create_stock_price_tool,
    "company_profile": _create_company_profile_tool,
    "financial_metrics": _create_financial_metrics_tool,
    "income_statement": _create_income_statement_tool,
    "balance_sheet": _create_balance_sheet_tool,
    "cash_flow": _create_cash_flow_tool,
    "stock_screener": _create_stock_screener_tool,
    "market_news": _create_market_news_tool,
    "crypto_price": _create_crypto_price_tool,
    "forex_rate": _create_forex_rate_tool,
    "economic_indicator": _create_economic_indicator_tool,
    "options_chain": _create_options_chain_tool,
    "earnings_calendar": _create_earnings_calendar_tool,
    "analyst_estimates": _create_analyst_estimates_tool,
    "market_index": _create_market_index_tool,
    "commodity_price": _create_commodity_price_tool,
    "sec_filings": _create_sec_filings_tool,
    "insider_trading": _create_insider_trading_tool,
    "dividends": _create_dividend_tool,
}


def get_financial_tools(names: list[str] | None = None) -> list[Tool]:
    """Get Financial financial tools.

    Args:
        names: Specific tool names, or None for all tools.

    Returns:
        List of Tool objects ready for use with any Duxx AI Agent.

    Usage:
        tools = get_financial_tools()  # All 19 tools
        tools = get_financial_tools(["stock_price", "financial_metrics", "market_news"])
    """
    if names:
        return [ALL_FINANCIAL_TOOLS[n]() for n in names if n in ALL_FINANCIAL_TOOLS]
    return [fn() for fn in ALL_FINANCIAL_TOOLS.values()]


def list_financial_tools() -> list[dict[str, str]]:
    """List all available Financial tools with descriptions."""
    tools = get_financial_tools()
    return [{"name": t.name, "description": t.description, "tags": t.tags} for t in tools]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pre-built Financial Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FinancialAgent:
    """Pre-built financial analysis agent powered by Financial data.

    Usage:
        from duxx_ai.tools.finagent import FinancialAgent

        agent = FinancialAgent.create()
        result = await agent.run("Analyze AAPL stock — fundamentals, recent performance, and outlook")
        result = await agent.run("Compare MSFT vs GOOGL P/E ratios")
        result = await agent.run("What are the top gainers today?")
        result = await agent.run("Get the latest crypto prices for BTC and ETH")
    """

    SYSTEM_PROMPT = """You are a senior financial analyst with access to real-time market data via Financial.

Your capabilities:
- Stock analysis (price history, fundamentals, metrics, filings)
- Company research (profiles, earnings, dividends, insider trading)
- Market overview (indices, news, screeners, calendars)
- Crypto analysis (prices, market data)
- Forex rates and currency analysis
- Economic indicators (GDP, CPI, unemployment, interest rates)
- Options analysis (chains, Greeks, implied volatility)
- Commodity prices (gold, oil, natural gas)

When analyzing stocks:
1. Start with the company profile and current price
2. Review key financial metrics (P/E, EPS, margins)
3. Check income statement trends
4. Note any recent news or insider activity
5. Provide a balanced assessment with data-backed insights

Always cite specific numbers and dates. Be precise with financial data.
Format responses clearly with sections and bullet points."""

    @classmethod
    def create(
        cls,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        tools: list[str] | None = None,
    ) -> Any:
        """Create a financial analysis agent.

        Args:
            llm_provider: LLM provider name
            llm_model: Model name
            tools: Specific Financial tools to include (None = all)

        Returns:
            Duxx AI Agent configured for financial analysis
        """
        from duxx_ai.core.agent import Agent, AgentConfig
        from duxx_ai.core.llm import LLMConfig

        finagent_tools = get_financial_tools(tools)

        agent = Agent(
            config=AgentConfig(
                name="financial-analyst",
                system_prompt=cls.SYSTEM_PROMPT,
                llm=LLMConfig(provider=llm_provider, model=llm_model),
                max_iterations=15,
            ),
            tools=finagent_tools,
        )
        return agent
