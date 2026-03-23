"""Financial Agent Tools for Duxx AI — Direct data access, zero external framework dependency.

Connects directly to free + paid financial data APIs:
- Yahoo Finance (yfinance) — stocks, crypto, forex, indices, commodities
- FRED (Federal Reserve) — economic indicators (free API key)
- SEC EDGAR — company filings (free, no key)
- CoinGecko — crypto (free, no key)
- ExchangeRate API — forex (free, no key)
- Alpha Vantage — market data (optional paid key)
- FMP (Financial Modeling Prep) — fundamentals (optional paid key)
- Polygon.io — real-time market data (optional paid key)

No OpenBB, no LangChain, no external framework. Pure Duxx AI.

Usage:
    from duxx_ai.tools.finagent import get_financial_tools, FinancialAgent

    tools = get_financial_tools()
    agent = FinancialAgent.create()
    result = await agent.run("Analyze AAPL stock performance")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from duxx_ai.core.tool import Tool, ToolParameter

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Direct Data Fetchers (no external framework)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _yf_fetch(symbol: str, period: str = "3mo") -> str:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty: return f"No data found for {symbol}"
        return hist.tail(20).to_string()
    except ImportError: return _yahoo_api_fallback(symbol)
    except Exception as e: return f"Error fetching {symbol}: {e}"


def _yf_info(symbol: str) -> str:
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        keys = ["shortName","sector","industry","fullTimeEmployees","marketCap","trailingPE","forwardPE","trailingEps","dividendYield","fiftyTwoWeekHigh","fiftyTwoWeekLow","averageVolume","currency","website","longBusinessSummary"]
        result = {k: info.get(k, "N/A") for k in keys if k in info}
        if "longBusinessSummary" in result: result["longBusinessSummary"] = result["longBusinessSummary"][:300] + "..."
        return json.dumps(result, indent=2, default=str)
    except ImportError: return "[yfinance not installed. Run: pip install yfinance]"
    except Exception as e: return f"Error: {e}"


def _yf_financials(symbol: str, statement: str = "income") -> str:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        df = {"income": t.income_stmt, "balance": t.balance_sheet, "cash": t.cashflow}.get(statement, t.income_stmt)
        return df.to_string() if df is not None and not df.empty else f"No {statement} data"
    except ImportError: return "[pip install yfinance]"
    except Exception as e: return f"Error: {e}"


def _yahoo_api_fallback(symbol: str) -> str:
    try:
        import httpx
        resp = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=3mo&interval=1d", headers={"User-Agent": "DuxxAI/1.0"}, timeout=10)
        if resp.status_code == 200:
            meta = resp.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
            return json.dumps({k: meta.get(k) for k in ["symbol","currency","regularMarketPrice","previousClose","fiftyTwoWeekHigh","fiftyTwoWeekLow"]}, indent=2)
        return f"Yahoo API: {resp.status_code}"
    except Exception as e: return f"Error: {e}"


def _fred_fetch(series_id: str) -> str:
    import os; key = os.environ.get("FRED_API_KEY", "")
    if not key: return "[Set FRED_API_KEY. Free at: https://fred.stlouisfed.org/docs/api/api_key.html]"
    try:
        import httpx
        resp = httpx.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": series_id, "api_key": key, "file_type": "json", "sort_order": "desc", "limit": 15}, timeout=10)
        return "\n".join(f"{o['date']}: {o['value']}" for o in resp.json().get("observations", []))
    except Exception as e: return f"FRED error: {e}"


def _coingecko_fetch(coin_id: str) -> str:
    try:
        import httpx
        data = httpx.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}", params={"localization":"false","tickers":"false","community_data":"false","developer_data":"false"}, timeout=10).json()
        m = data.get("market_data", {})
        return json.dumps({"name":data.get("name"),"symbol":data.get("symbol"),"price_usd":m.get("current_price",{}).get("usd"),"market_cap":m.get("market_cap",{}).get("usd"),"24h_change":m.get("price_change_percentage_24h"),"7d_change":m.get("price_change_percentage_7d"),"ath":m.get("ath",{}).get("usd"),"volume":m.get("total_volume",{}).get("usd")}, indent=2, default=str)
    except Exception as e: return f"CoinGecko error: {e}"


def _exchangerate_fetch(base: str = "USD") -> str:
    try:
        import httpx
        rates = httpx.get(f"https://open.er-api.com/v6/latest/{base}", timeout=10).json().get("rates", {})
        return json.dumps({"base": base, "rates": {k: rates[k] for k in ["EUR","GBP","JPY","CHF","AUD","CAD","CNY","INR","KRW","SGD"] if k in rates}}, indent=2)
    except Exception as e: return f"Forex error: {e}"


def _sec_filings(symbol: str) -> str:
    try:
        import httpx
        resp = httpx.get(f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&forms=10-K,10-Q,8-K", headers={"User-Agent": "DuxxAI research@duxx.ai"}, timeout=10)
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])[:5]
            return "\n".join(f"{h['_source'].get('file_date','N/A')} | {h['_source'].get('form_type','N/A')} | {h['_source'].get('entity_name','N/A')}" for h in hits) or f"No filings for {symbol}"
        return f"SEC search: {resp.status_code}"
    except Exception as e: return f"SEC error: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _t(name, desc, params, fn, tags=None):
    tool = Tool(name=name, description=desc, parameters=params, tags=tags or ["finance"])
    tool.bind(fn)
    return tool

def _a(call): return call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})


def _create_tools() -> dict[str, Tool]:
    tools = {}

    async def _f1(c): return _yf_fetch(_a(c).get("symbol","AAPL"), _a(c).get("period","3mo"))
    tools["stock_price"] = _t("stock_price", "Get historical stock price (OHLCV). Source: Yahoo Finance.", [ToolParameter(name="symbol",type="string",description="Ticker e.g. AAPL",required=True), ToolParameter(name="period",type="string",description="1d,5d,1mo,3mo,6mo,1y,5y,max",required=False,default="3mo")], _f1, ["finance","equity"])

    async def _f2(c): return _yf_info(_a(c).get("symbol","AAPL"))
    tools["company_profile"] = _t("company_profile", "Get company info — sector, industry, market cap, P/E, EPS, 52wk range.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f2, ["finance","equity"])

    async def _f3(c): return _yf_financials(_a(c).get("symbol","AAPL"), "income")
    tools["income_statement"] = _t("income_statement", "Get income statement — revenue, gross profit, net income, EPS.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f3, ["finance","equity"])

    async def _f4(c): return _yf_financials(_a(c).get("symbol","AAPL"), "balance")
    tools["balance_sheet"] = _t("balance_sheet", "Get balance sheet — assets, liabilities, equity, cash, debt.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f4, ["finance","equity"])

    async def _f5(c): return _yf_financials(_a(c).get("symbol","AAPL"), "cash")
    tools["cash_flow"] = _t("cash_flow", "Get cash flow — operating, investing, financing, free cash flow.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f5, ["finance","equity"])

    async def _f6(c):
        try:
            import yfinance as yf; news = yf.Ticker(_a(c).get("symbol","AAPL")).news[:5]
            return "\n".join(f"- {n.get('title','')} ({n.get('publisher','')})" for n in news) or "No news"
        except: return "[pip install yfinance for news]"
    tools["market_news"] = _t("market_news", "Get latest financial news for a stock.", [ToolParameter(name="symbol",type="string",description="Ticker",required=False)], _f6, ["finance","news"])

    async def _f7(c): return _coingecko_fetch(_a(c).get("coin_id","bitcoin"))
    tools["crypto_price"] = _t("crypto_price", "Get crypto data — price, market cap, changes, ATH. Free via CoinGecko.", [ToolParameter(name="coin_id",type="string",description="bitcoin, ethereum, solana, cardano",required=True)], _f7, ["finance","crypto"])

    async def _f8(c): return _exchangerate_fetch(_a(c).get("base","USD"))
    tools["forex_rates"] = _t("forex_rates", "Get forex rates for major currencies. Free, no API key.", [ToolParameter(name="base",type="string",description="USD, EUR, GBP",required=False,default="USD")], _f8, ["finance","forex"])

    async def _f9(c):
        ind = _a(c).get("indicator","GDP").upper()
        m = {"GDP":"GDP","CPI":"CPIAUCSL","UNEMPLOYMENT":"UNRATE","FED_RATE":"FEDFUNDS","INFLATION":"T10YIE","M2":"M2SL","RETAIL_SALES":"RSXFS","HOUSING":"HOUST","CONSUMER_SENTIMENT":"UMCSENT"}
        return _fred_fetch(m.get(ind, ind))
    tools["economic_indicator"] = _t("economic_indicator", "Get US economic data from FRED — GDP, CPI, unemployment, fed rate, inflation.", [ToolParameter(name="indicator",type="string",description="GDP,CPI,UNEMPLOYMENT,FED_RATE,INFLATION,M2,RETAIL_SALES,HOUSING,CONSUMER_SENTIMENT",required=True)], _f9, ["finance","economy"])

    async def _f10(c): return _sec_filings(_a(c).get("symbol","AAPL"))
    tools["sec_filings"] = _t("sec_filings", "Get SEC filings (10-K, 10-Q, 8-K) from EDGAR. Free.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f10, ["finance","regulatory"])

    async def _f11(c):
        try:
            import yfinance as yf; t = yf.Ticker(_a(c).get("symbol","AAPL")); d = t.options[:3]; chain = t.option_chain(d[0])
            return f"Expiry: {d[0]}\nCALLS:\n{chain.calls.head(10).to_string()}\nPUTS:\n{chain.puts.head(10).to_string()}"
        except: return "[pip install yfinance for options]"
    tools["options_chain"] = _t("options_chain", "Get options chain — calls/puts, strike, premium, IV, volume.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f11, ["finance","options"])

    async def _f12(c):
        try:
            import yfinance as yf; d = yf.Ticker(_a(c).get("symbol","AAPL")).dividends
            return d.tail(20).to_string() if d is not None and not d.empty else "No dividends"
        except: return "[pip install yfinance]"
    tools["dividends"] = _t("dividends", "Get dividend history — dates, amounts, yield.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f12, ["finance","equity"])

    async def _f13(c):
        try:
            import yfinance as yf; d = yf.Ticker(_a(c).get("symbol","AAPL")).insider_transactions
            return d.head(15).to_string() if d is not None and not d.empty else "No insider data"
        except: return "[pip install yfinance]"
    tools["insider_trading"] = _t("insider_trading", "Get insider trading — executive buys/sells, amounts.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f13, ["finance","equity"])

    async def _f14(c):
        try:
            import yfinance as yf; d = yf.Ticker(_a(c).get("symbol","AAPL")).recommendations
            return d.tail(10).to_string() if d is not None and not d.empty else "No recommendations"
        except: return "[pip install yfinance]"
    tools["analyst_recommendations"] = _t("analyst_recommendations", "Get analyst buy/sell/hold recommendations.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f14, ["finance","equity"])

    async def _f15(c): return _yf_fetch(_a(c).get("symbol","^GSPC"), "1mo")
    tools["market_index"] = _t("market_index", "Get index data — S&P 500 (^GSPC), NASDAQ (^IXIC), Dow (^DJI).", [ToolParameter(name="symbol",type="string",description="^GSPC, ^IXIC, ^DJI, ^RUT",required=True)], _f15, ["finance","index"])

    async def _f16(c): return _yf_fetch(_a(c).get("symbol","GC=F"), "1mo")
    tools["commodity_price"] = _t("commodity_price", "Get commodity prices — Gold (GC=F), Oil (CL=F), Silver (SI=F).", [ToolParameter(name="symbol",type="string",description="GC=F, CL=F, SI=F, NG=F",required=True)], _f16, ["finance","commodities"])

    async def _f17(c):
        syms = _a(c).get("symbols","AAPL,MSFT").split(",")
        return "\n\n".join(f"=== {s.strip()} ===\n{_yf_info(s.strip())}" for s in syms[:5])
    tools["compare_stocks"] = _t("compare_stocks", "Compare multiple stocks side by side.", [ToolParameter(name="symbols",type="string",description="Comma-separated: AAPL,MSFT,GOOGL",required=True)], _f17, ["finance","equity"])

    async def _f18(c):
        try:
            import yfinance as yf; return str(yf.Ticker(_a(c).get("symbol","AAPL")).calendar)
        except: return "[pip install yfinance]"
    tools["earnings_calendar"] = _t("earnings_calendar", "Get upcoming earnings date and estimates.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f18, ["finance","equity"])

    async def _f19(c):
        try:
            import yfinance as yf; t = yf.Ticker(_a(c).get("symbol","AAPL")); h = t.institutional_holders
            return h.head(15).to_string() if h is not None and not h.empty else "No institutional data"
        except: return "[pip install yfinance]"
    tools["institutional_holders"] = _t("institutional_holders", "Get top institutional holders — Vanguard, BlackRock, etc.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _f19, ["finance","equity"])

    return tools


ALL_FINANCIAL_TOOLS = _create_tools()


def get_financial_tools(names: list[str] | None = None) -> list[Tool]:
    """Get financial data tools. 19 tools covering stocks, crypto, forex, economy, options.

    Data sources (all free, no external framework):
        Yahoo Finance (yfinance), CoinGecko, ExchangeRate API, FRED, SEC EDGAR
    """
    if names: return [ALL_FINANCIAL_TOOLS[n] for n in names if n in ALL_FINANCIAL_TOOLS]
    return list(ALL_FINANCIAL_TOOLS.values())


def list_financial_tools() -> list[dict[str, str]]:
    return [{"name": t.name, "description": t.description, "tags": t.tags} for t in ALL_FINANCIAL_TOOLS.values()]


class FinancialAgent:
    """Pre-built financial analyst agent. Direct data sources, no external framework."""

    SYSTEM_PROMPT = """You are a senior financial analyst with real-time market data access.

Tools: stock prices, company profiles, financial statements (income/balance/cash flow),
analyst recommendations, insider trading, dividends, options chains, crypto (CoinGecko),
forex rates, economic indicators (FRED), SEC filings, earnings calendar, institutional holders.

Analysis approach:
1. Current price and key metrics (P/E, EPS, market cap)
2. Financial statement trends
3. Analyst sentiment and insider activity
4. Relevant news and upcoming events
5. Data-backed assessment

Always cite specific numbers. Use tables for comparisons."""

    @classmethod
    def create(cls, llm_provider: str = "openai", llm_model: str = "gpt-4o", tools: list[str] | None = None) -> Any:
        from duxx_ai.core.agent import Agent, AgentConfig
        from duxx_ai.core.llm import LLMConfig
        return Agent(
            config=AgentConfig(name="financial-analyst", system_prompt=cls.SYSTEM_PROMPT, llm=LLMConfig(provider=llm_provider, model=llm_model), max_iterations=15),
            tools=get_financial_tools(tools),
        )
