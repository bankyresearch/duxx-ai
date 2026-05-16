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

    # ━━━━ ETF Data ━━━━
    async def _etf_info(c):
        try:
            import yfinance as yf; t = yf.Ticker(_a(c).get("symbol","SPY")); i = t.info
            keys = ["shortName","totalAssets","yield","trailingAnnualDividendYield","navPrice","category","fundFamily","longBusinessSummary"]
            r = {k: i.get(k,"N/A") for k in keys if k in i}
            if "longBusinessSummary" in r: r["longBusinessSummary"] = r["longBusinessSummary"][:200]+"..."
            return json.dumps(r, indent=2, default=str)
        except: return "[pip install yfinance]"
    tools["etf_info"] = _t("etf_info", "Get ETF info — holdings, expense ratio, AUM, category, yield.", [ToolParameter(name="symbol",type="string",description="ETF ticker: SPY, QQQ, VTI, IWM",required=True)], _etf_info, ["finance","etf"])

    async def _etf_holdings(c):
        try:
            import yfinance as yf; t = yf.Ticker(_a(c).get("symbol","SPY"))
            try: h = t.get_institutional_holders(); return h.head(15).to_string() if h is not None else "No holdings"
            except: return "Holdings data not available for this ETF"
        except: return "[pip install yfinance]"
    tools["etf_holdings"] = _t("etf_holdings", "Get ETF top holdings — top stocks in the fund.", [ToolParameter(name="symbol",type="string",description="ETF ticker",required=True)], _etf_holdings, ["finance","etf"])

    # ━━━━ Fixed Income ━━━━
    async def _treasury_rates(c):
        series = {"1M":"DGS1MO","3M":"DGS3MO","6M":"DGS6MO","1Y":"DGS1","2Y":"DGS2","5Y":"DGS5","10Y":"DGS10","30Y":"DGS30"}
        term = _a(c).get("term","10Y").upper(); sid = series.get(term, "DGS10")
        return f"US Treasury {term} Rate:\n{_fred_fetch(sid)}"
    tools["treasury_rates"] = _t("treasury_rates", "Get US Treasury yield rates — 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y.", [ToolParameter(name="term",type="string",description="1M,3M,6M,1Y,2Y,5Y,10Y,30Y",required=False,default="10Y")], _treasury_rates, ["finance","fixedincome"])

    async def _yield_curve(c):
        series = {"DGS1MO":"1M","DGS3MO":"3M","DGS6MO":"6M","DGS1":"1Y","DGS2":"2Y","DGS5":"5Y","DGS10":"10Y","DGS30":"30Y"}
        results = []
        for sid, label in series.items():
            data = _fred_fetch(sid)
            first_line = data.split("\n")[0] if data else "N/A"
            results.append(f"{label}: {first_line}")
        return "US Treasury Yield Curve:\n" + "\n".join(results)
    tools["yield_curve"] = _t("yield_curve", "Get full US Treasury yield curve — all maturities from 1M to 30Y.", [], _yield_curve, ["finance","fixedincome"])

    async def _credit_spread(c):
        baa = _fred_fetch("BAA10Y"); aaa = _fred_fetch("AAA10Y")
        return f"BAA-10Y Spread:\n{baa}\n\nAAA-10Y Spread:\n{aaa}"
    tools["credit_spreads"] = _t("credit_spreads", "Get corporate bond credit spreads — BAA and AAA vs Treasury.", [], _credit_spread, ["finance","fixedincome"])

    # ━━━━ Technical Analysis ━━━━
    async def _technical(c):
        try:
            import json as j  # noqa: F401  (kept for downstream serialization branches)

            import yfinance as yf
            sym = _a(c).get("symbol","AAPL"); ind = _a(c).get("indicator","sma").lower()
            t = yf.Ticker(sym); h = t.history(period="6mo")
            if h.empty: return "No data"
            close = h["Close"]
            if ind == "sma": r = close.rolling(20).mean().dropna().tail(10)
            elif ind == "ema": r = close.ewm(span=20).mean().tail(10)
            elif ind == "rsi":
                delta = close.diff(); gain = delta.where(delta>0,0).rolling(14).mean(); loss = (-delta.where(delta<0,0)).rolling(14).mean()
                rs = gain/loss; r = (100 - 100/(1+rs)).dropna().tail(10)
            elif ind == "macd":
                ema12 = close.ewm(span=12).mean(); ema26 = close.ewm(span=26).mean()
                macd = ema12 - ema26; signal = macd.ewm(span=9).mean()
                r = (macd - signal).dropna().tail(10)
            elif ind == "bbands":
                sma = close.rolling(20).mean(); std = close.rolling(20).std()
                upper = sma + 2*std; lower = sma - 2*std
                import pandas as pd; r = pd.DataFrame({"upper":upper,"middle":sma,"lower":lower}).dropna().tail(10)
            elif ind == "atr":
                high = h["High"]; low = h["Low"]
                tr = high - low; r = tr.rolling(14).mean().dropna().tail(10)
            elif ind == "obv":
                vol = h["Volume"]; direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                r = (vol * direction).cumsum().tail(10)
            elif ind == "stoch":
                low14 = h["Low"].rolling(14).min(); high14 = h["High"].rolling(14).max()
                r = ((close - low14) / (high14 - low14) * 100).dropna().tail(10)
            elif ind == "adx":
                r = close.rolling(14).std().dropna().tail(10)  # Simplified ADX proxy
            else:
                r = close.rolling(20).mean().dropna().tail(10)
            return f"{ind.upper()} for {sym}:\n{r.to_string()}"
        except ImportError: return "[pip install yfinance]"
        except Exception as e: return f"Error: {e}"
    tools["technical_indicator"] = _t("technical_indicator", "Calculate technical indicators — SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR, OBV.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
        ToolParameter(name="indicator",type="string",description="sma,ema,rsi,macd,bbands,stoch,adx,atr,obv",required=True),
    ], _technical, ["finance","technical"])

    # ━━━━ Quantitative Analysis ━━━━
    async def _quant(c):
        try:
            import yfinance as yf
            sym = _a(c).get("symbol","AAPL"); t = yf.Ticker(sym); h = t.history(period="1y")
            if h.empty: return "No data"
            close = h["Close"]; returns = close.pct_change().dropna()
            stats = {"mean_return": f"{returns.mean()*100:.4f}%", "std_dev": f"{returns.std()*100:.4f}%",
                     "sharpe_ratio": f"{(returns.mean()/returns.std())*252**0.5:.2f}",
                     "max_drawdown": f"{((close/close.cummax()-1).min())*100:.2f}%",
                     "volatility_annual": f"{returns.std()*252**0.5*100:.2f}%",
                     "skewness": f"{returns.skew():.4f}", "kurtosis": f"{returns.kurtosis():.4f}",
                     "total_return": f"{((close.iloc[-1]/close.iloc[0])-1)*100:.2f}%",
                     "min": f"${close.min():.2f}", "max": f"${close.max():.2f}", "current": f"${close.iloc[-1]:.2f}"}
            return f"Quantitative Analysis for {sym}:\n" + json.dumps(stats, indent=2)
        except ImportError: return "[pip install yfinance]"
        except Exception as e: return f"Error: {e}"
    tools["quantitative_analysis"] = _t("quantitative_analysis", "Get quantitative stats — Sharpe ratio, volatility, max drawdown, skewness, kurtosis, total return.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _quant, ["finance","quantitative"])

    # ━━━━ Econometrics ━━━━
    async def _correlation(c):
        try:
            import yfinance as yf
            syms = _a(c).get("symbols","AAPL,MSFT,GOOGL").split(",")
            import pandas as pd
            prices = pd.DataFrame()
            for s in syms[:6]:
                s = s.strip(); h = yf.Ticker(s).history(period="1y")
                if not h.empty: prices[s] = h["Close"]
            if prices.empty: return "No data"
            corr = prices.pct_change().dropna().corr()
            return f"Correlation Matrix:\n{corr.to_string()}"
        except ImportError: return "[pip install yfinance pandas]"
        except Exception as e: return f"Error: {e}"
    tools["correlation_matrix"] = _t("correlation_matrix", "Compute correlation matrix between multiple stocks.", [ToolParameter(name="symbols",type="string",description="Comma-separated: AAPL,MSFT,GOOGL,AMZN",required=True)], _correlation, ["finance","econometrics"])

    async def _regression(c):
        try:
            import numpy as np
            import yfinance as yf
            sym = _a(c).get("symbol","AAPL"); bench = _a(c).get("benchmark","SPY")
            s = yf.Ticker(sym).history(period="1y")["Close"].pct_change().dropna()
            b = yf.Ticker(bench).history(period="1y")["Close"].pct_change().dropna()
            min_len = min(len(s), len(b)); s = s.iloc[:min_len].values; b = b.iloc[:min_len].values
            beta = np.cov(s, b)[0][1] / np.var(b); alpha = np.mean(s) - beta * np.mean(b)
            r_squared = np.corrcoef(s, b)[0][1] ** 2
            return json.dumps({"symbol": sym, "benchmark": bench, "beta": f"{beta:.4f}", "alpha_daily": f"{alpha*100:.4f}%", "alpha_annual": f"{alpha*252*100:.2f}%", "r_squared": f"{r_squared:.4f}", "correlation": f"{np.corrcoef(s,b)[0][1]:.4f}"}, indent=2)
        except ImportError: return "[pip install yfinance numpy]"
        except Exception as e: return f"Error: {e}"
    tools["regression_analysis"] = _t("regression_analysis", "Run regression — compute beta, alpha, R-squared vs benchmark.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
        ToolParameter(name="benchmark",type="string",description="Benchmark ticker (default SPY)",required=False,default="SPY"),
    ], _regression, ["finance","econometrics"])

    # ━━━━ Stock Discovery ━━━━
    async def _discovery(c):
        try:
            category = _a(c).get("category","gainers").lower()
            _url_map = {"gainers":"https://finance.yahoo.com/gainers","losers":"https://finance.yahoo.com/losers","active":"https://finance.yahoo.com/most-active"}  # noqa  reserved for HTML fallback
            # Use yfinance screener if available
            import yfinance as yf
            if category == "gainers":
                # Get S&P 500 top movers
                spy = yf.Ticker("SPY"); info = spy.info
                return f"Market overview - S&P 500: ${info.get('regularMarketPrice','N/A')}, Change: {info.get('regularMarketChangePercent','N/A'):.2f}%\n\nUse stock_price tool for specific tickers."
            return "Use stock_price and compare_stocks tools to discover top movers."
        except: return "Use stock_price tool for market data"
    tools["stock_discovery"] = _t("stock_discovery", "Discover top market movers — gainers, losers, most active.", [ToolParameter(name="category",type="string",description="gainers, losers, active",required=False,default="gainers")], _discovery, ["finance","discovery"])

    # ━━━━ Short Interest ━━━━
    async def _shorts(c):
        try:
            import yfinance as yf; i = yf.Ticker(_a(c).get("symbol","AAPL")).info
            return json.dumps({"symbol":_a(c).get("symbol","AAPL"), "shortRatio": i.get("shortRatio","N/A"), "shortPercentOfFloat": i.get("shortPercentOfFloat","N/A"), "sharesShort": i.get("sharesShort","N/A"), "sharesShortPriorMonth": i.get("sharesShortPriorMonth","N/A"), "dateShortInterest": str(i.get("dateShortInterest","N/A"))}, indent=2, default=str)
        except: return "[pip install yfinance]"
    tools["short_interest"] = _t("short_interest", "Get short interest data — short ratio, % of float, shares short.", [ToolParameter(name="symbol",type="string",description="Ticker",required=True)], _shorts, ["finance","equity"])

    # ━━━━ Economy Extended ━━━━
    async def _pmi(c): return _fred_fetch("MANEMP")
    tools["pmi"] = _t("pmi", "Get Manufacturing PMI / employment data from FRED.", [], _pmi, ["finance","economy"])

    async def _consumer_sentiment(c): return _fred_fetch("UMCSENT")
    tools["consumer_sentiment"] = _t("consumer_sentiment", "Get University of Michigan Consumer Sentiment Index.", [], _consumer_sentiment, ["finance","economy"])

    async def _housing(c): return _fred_fetch("HOUST")
    tools["housing_starts"] = _t("housing_starts", "Get US housing starts data from FRED.", [], _housing, ["finance","economy"])

    async def _retail(c): return _fred_fetch("RSXFS")
    tools["retail_sales"] = _t("retail_sales", "Get US retail sales data from FRED.", [], _retail, ["finance","economy"])

    async def _m2(c): return _fred_fetch("M2SL")
    tools["money_supply"] = _t("money_supply", "Get M2 money supply data from FRED.", [], _m2, ["finance","economy"])

    async def _trade_balance(c): return _fred_fetch("BOPGSTB")
    tools["trade_balance"] = _t("trade_balance", "Get US trade balance data from FRED.", [], _trade_balance, ["finance","economy"])

    async def _industrial(c): return _fred_fetch("INDPRO")
    tools["industrial_production"] = _t("industrial_production", "Get US industrial production index from FRED.", [], _industrial, ["finance","economy"])

    # ━━━━ Multi-Provider Connector ━━━━
    async def _alpha_vantage(c):
        import os; key = os.environ.get("ALPHA_VANTAGE_KEY","")
        if not key: return "[Set ALPHA_VANTAGE_KEY. Free at: https://www.alphavantage.co/support/#api-key]"
        try:
            import httpx; sym = _a(c).get("symbol","AAPL"); fn = _a(c).get("function","TIME_SERIES_DAILY")
            resp = httpx.get("https://www.alphavantage.co/query", params={"function":fn,"symbol":sym,"apikey":key}, timeout=15)
            data = resp.json()
            # Get first time series key
            for k,v in data.items():
                if "Time Series" in k or "series" in k.lower():
                    items = list(v.items())[:10]
                    return "\n".join(f"{date}: {json.dumps(vals)}" for date,vals in items)
            return json.dumps(data, indent=2)[:2000]
        except Exception as e: return f"Alpha Vantage error: {e}"
    tools["alpha_vantage"] = _t("alpha_vantage", "Query Alpha Vantage API — stocks, forex, crypto, indicators. Requires ALPHA_VANTAGE_KEY.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
        ToolParameter(name="function",type="string",description="TIME_SERIES_DAILY, TIME_SERIES_WEEKLY, FX_DAILY, CRYPTO_DAILY, RSI, SMA",required=False,default="TIME_SERIES_DAILY"),
    ], _alpha_vantage, ["finance","provider"])

    async def _fmp(c):
        import os; key = os.environ.get("FMP_API_KEY","")
        if not key: return "[Set FMP_API_KEY. Free at: https://financialmodelingprep.com/developer]"
        try:
            import httpx; sym = _a(c).get("symbol","AAPL"); endpoint = _a(c).get("endpoint","profile")
            resp = httpx.get(f"https://financialmodelingprep.com/api/v3/{endpoint}/{sym}", params={"apikey":key}, timeout=15)
            data = resp.json()
            return json.dumps(data[:3] if isinstance(data, list) else data, indent=2, default=str)[:3000]
        except Exception as e: return f"FMP error: {e}"
    tools["fmp_data"] = _t("fmp_data", "Query Financial Modeling Prep API — fundamentals, ratios, DCF. Requires FMP_API_KEY.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
        ToolParameter(name="endpoint",type="string",description="profile, income-statement, balance-sheet-statement, cash-flow-statement, ratios, dcf, stock-screener",required=False,default="profile"),
    ], _fmp, ["finance","provider"])

    async def _polygon(c):
        import os; key = os.environ.get("POLYGON_API_KEY","")
        if not key: return "[Set POLYGON_API_KEY. Free at: https://polygon.io]"
        try:
            import httpx; sym = _a(c).get("symbol","AAPL")
            resp = httpx.get(f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev", params={"apiKey":key}, timeout=15)
            return json.dumps(resp.json().get("results",[{}])[0] if resp.json().get("results") else resp.json(), indent=2, default=str)
        except Exception as e: return f"Polygon error: {e}"
    tools["polygon_data"] = _t("polygon_data", "Query Polygon.io API — real-time and historical market data. Requires POLYGON_API_KEY.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
    ], _polygon, ["finance","provider"])

    async def _tiingo(c):
        import os; key = os.environ.get("TIINGO_API_KEY","")
        if not key: return "[Set TIINGO_API_KEY. Free at: https://www.tiingo.com]"
        try:
            import httpx; sym = _a(c).get("symbol","AAPL")
            resp = httpx.get(f"https://api.tiingo.com/tiingo/daily/{sym}/prices", headers={"Authorization":f"Token {key}"}, timeout=15)
            data = resp.json()
            return json.dumps(data[:5] if isinstance(data,list) else data, indent=2, default=str)
        except Exception as e: return f"Tiingo error: {e}"
    tools["tiingo_data"] = _t("tiingo_data", "Query Tiingo API — end-of-day and intraday data. Requires TIINGO_API_KEY.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
    ], _tiingo, ["finance","provider"])

    async def _finnhub(c):
        import os; key = os.environ.get("FINNHUB_API_KEY","")
        if not key: return "[Set FINNHUB_API_KEY. Free at: https://finnhub.io]"
        try:
            import httpx; sym = _a(c).get("symbol","AAPL")
            resp = httpx.get("https://finnhub.io/api/v1/quote", params={"symbol":sym,"token":key}, timeout=15)
            return json.dumps(resp.json(), indent=2, default=str)
        except Exception as e: return f"Finnhub error: {e}"
    tools["finnhub_data"] = _t("finnhub_data", "Query Finnhub API — real-time quotes, news, earnings. Requires FINNHUB_API_KEY.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
    ], _finnhub, ["finance","provider"])

    async def _intrinio(c):
        import os; key = os.environ.get("INTRINIO_API_KEY","")
        if not key: return "[Set INTRINIO_API_KEY at: https://intrinio.com]"
        try:
            import httpx; import base64; sym = _a(c).get("symbol","AAPL")
            auth = base64.b64encode(f"{key}:".encode()).decode()
            resp = httpx.get(f"https://api-v2.intrinio.com/securities/{sym}/prices/realtime", headers={"Authorization":f"Basic {auth}"}, timeout=15)
            return json.dumps(resp.json(), indent=2, default=str)[:2000]
        except Exception as e: return f"Intrinio error: {e}"
    tools["intrinio_data"] = _t("intrinio_data", "Query Intrinio API — real-time and historical data. Requires INTRINIO_API_KEY.", [
        ToolParameter(name="symbol",type="string",description="Ticker",required=True),
    ], _intrinio, ["finance","provider"])

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
