"""
Simple Financial Tools for LangChain Agent
Following LangChain best practices - no complex input parsing needed!
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Literal
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import config

# Simple Pydantic models for structured output
class StockPriceData(BaseModel):
    """Structured stock price information."""
    symbol: str = Field(description="Stock ticker symbol")
    current_price: Optional[float] = Field(description="Current stock price")
    market_cap: Optional[int] = Field(description="Market capitalization")
    pe_ratio: Optional[float] = Field(description="Price-to-earnings ratio")
    week_52_high: Optional[float] = Field(description="52-week high price")
    week_52_low: Optional[float] = Field(description="52-week low price")
    formatted_summary: str = Field(description="Human-readable summary")
    error: Optional[str] = None

class CompanyInfo(BaseModel):
    """Structured company information."""
    symbol: str = Field(description="Stock ticker symbol")
    name: Optional[str] = Field(description="Company name")
    sector: Optional[str] = Field(description="Business sector")
    industry: Optional[str] = Field(description="Industry classification")
    country: Optional[str] = Field(description="Country of incorporation")
    employees: Optional[int] = Field(description="Number of employees")
    business_summary: Optional[str] = Field(description="Business description")
    error: Optional[str] = None

class FinancialHistoryResult(BaseModel):
    symbol: str
    period: str
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    total_return_percent: Optional[float] = None
    cagr_percent: Optional[float] = None
    volatility_percent: Optional[float] = None
    max_drawdown_percent: Optional[float] = None
    trading_days: Optional[int] = None
    formatted_summary: Optional[str] = None
    error: Optional[str] = None

class CompoundGrowthResult(BaseModel):
    principal: float
    annual_rate: float
    years: float
    future_value: float
    total_growth: float
    total_return_percent: float
    formatted_summary: str
    error: Optional[str] = None

class FinancialRatioResult(BaseModel):
    numerator: float
    denominator: float
    ratio_type: str
    ratio_value: Optional[float] = None
    description: Optional[str] = None
    interpretation: Optional[str] = None
    context: Optional[str] = None
    formatted_summary: Optional[str] = None
    error: Optional[str] = None

# Simple tools - LangChain handles input parsing automatically
@tool
def get_stock_price(symbol: str) -> StockPriceData:
    """
    Get current stock price and key metrics for a publicly traded company.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA')

    Returns:
        Structured stock price data including current price, market cap, and ratios
    """
    try:
        symbol = symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')

        # Get market cap (convert to readable format)
        market_cap = info.get('marketCap')
        if market_cap:
            market_cap = int(market_cap)

        # Get P/E ratio
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')

        # Get 52-week range
        week_52_high = info.get('fiftyTwoWeekHigh')
        week_52_low = info.get('fiftyTwoWeekLow')

        # Create formatted summary
        summary_parts = [f"Stock: {symbol}"]
        if current_price:
            summary_parts.append(f"Price: ${current_price:.2f}")
        if market_cap:
            summary_parts.append(f"Market Cap: ${market_cap:,}")
        if pe_ratio:
            summary_parts.append(f"P/E: {pe_ratio:.2f}")
        if week_52_high and week_52_low:
            summary_parts.append(f"52W Range: ${week_52_low:.2f} - ${week_52_high:.2f}")

        formatted_summary = " | ".join(summary_parts)

        return StockPriceData(
            symbol=symbol,
            current_price=current_price,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            week_52_high=week_52_high,
            week_52_low=week_52_low,
            formatted_summary=formatted_summary,
            error=None
        )

    except Exception as e:
        return StockPriceData(
            symbol=symbol,
            current_price=None,
            market_cap=None,
            pe_ratio=None,
            week_52_high=None,
            week_52_low=None,
            formatted_summary=f"Error retrieving stock data for {symbol}: {str(e)}",
            error=str(e)
        )

@tool
def get_company_info(symbol: str) -> CompanyInfo:
    """
    Get detailed company information for a publicly traded company.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        Structured company information including name, sector, industry details
    """
    try:
        symbol = symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return CompanyInfo(
            symbol=symbol,
            name=info.get('longName') or info.get('shortName'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            country=info.get('country'),
            employees=info.get('fullTimeEmployees'),
            business_summary=info.get('longBusinessSummary', 'No business summary available'),
            error=None
        )

    except Exception as e:
        return CompanyInfo(
            symbol=symbol,
            name=None,
            sector=None,
            industry=None,
            country=None,
            employees=None,
            business_summary=f"Error retrieving company info for {symbol}: {str(e)}",
            error=str(e)
        )

@tool
def get_financial_history(query: str) -> FinancialHistoryResult:
    """
    Get historical financial performance and calculate key metrics.

    Args:
        query: Query in format "SYMBOL PERIOD" (e.g., "AAPL 5y", "TSLA 2y")

    Returns:
        FinancialHistoryResult: Historical performance analysis
    """
    try:
        # Parse the query to extract symbol and period
        parts = query.strip().split()
        if len(parts) >= 2:
            symbol = parts[0].upper().strip()
            period = parts[1].lower().strip()
        elif len(parts) == 1:
            symbol = parts[0].upper().strip()
            period = "1y"
        else:
            symbol = query.upper().strip()
            period = "1y"

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return FinancialHistoryResult(
                symbol=symbol,
                period=period,
                formatted_summary=f"No historical data available for {symbol}",
                error="No data available"
            )

        # Calculate metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100

        # Calculate CAGR (Compound Annual Growth Rate)
        years = len(hist) / 252  # Approximate trading days per year
        if years > 0:
            cagr = ((end_price / start_price) ** (1/years) - 1) * 100
        else:
            cagr = 0

        # Calculate volatility (annualized)
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # Calculate max drawdown
        rolling_max = hist['Close'].expanding().max()
        drawdown = (hist['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        formatted_summary = f"""
{symbol} Performance ({period}):
• Total Return: {total_return:.2f}%
• CAGR: {cagr:.2f}%
• Volatility: {volatility:.2f}%
• Max Drawdown: {max_drawdown:.2f}%
• Trading Days: {len(hist)}
        """.strip()

        return FinancialHistoryResult(
            symbol=symbol,
            period=period,
            start_price=round(start_price, 2),
            end_price=round(end_price, 2),
            total_return_percent=round(total_return, 2),
            cagr_percent=round(cagr, 2),
            volatility_percent=round(volatility, 2),
            max_drawdown_percent=round(max_drawdown, 2),
            trading_days=len(hist),
            formatted_summary=formatted_summary,
            error=None
        )

    except Exception as e:
        return FinancialHistoryResult(
            symbol=symbol,
            period=period,
            formatted_summary=f"Error retrieving financial history for {symbol}: {str(e)}",
            error=str(e)
        )

@tool
def calculate_compound_growth(query: str) -> CompoundGrowthResult:
    """
    Calculate compound growth and future value of an investment.

    Args:
        query: Query in format "PRINCIPAL RATE YEARS" (e.g., "10000 0.07 10")

    Returns:
        CompoundGrowthResult: Future value, growth, and return calculations
    """
    try:
        # Parse the query to extract principal, annual_rate, years
        parts = query.strip().split()
        if len(parts) >= 3:
            principal = float(parts[0])
            annual_rate = float(parts[1])
            years = float(parts[2])
        else:
            raise ValueError("Query must contain principal, annual_rate, and years separated by spaces")

        if years <= 0 or principal <= 0:
            return CompoundGrowthResult(
                principal=principal,
                annual_rate=annual_rate,
                years=years,
                future_value=0.0,
                total_growth=0.0,
                total_return_percent=0.0,
                formatted_summary="",
                error="Principal and years must be positive numbers"
            )

        future_value = principal * (1 + annual_rate) ** years
        total_growth = future_value - principal
        total_return_pct = (future_value / principal - 1) * 100

        formatted_summary = f"""
Compound Growth Calculation:
• Initial Investment: ${principal:,.2f}
• Annual Return Rate: {annual_rate*100:.2f}%
• Time Period: {years} years
• Future Value: ${future_value:,.2f}
• Total Growth: ${total_growth:,.2f}
• Total Return: {total_return_pct:.2f}%
        """.strip()

        return CompoundGrowthResult(
            principal=principal,
            annual_rate=annual_rate,
            years=years,
            future_value=round(future_value, 2),
            total_growth=round(total_growth, 2),
            total_return_percent=round(total_return_pct, 2),
            formatted_summary=formatted_summary,
            error=None
        )

    except Exception as e:
        return CompoundGrowthResult(
            principal=principal,
            annual_rate=annual_rate,
            years=years,
            future_value=0.0,
            total_growth=0.0,
            total_return_percent=0.0,
            formatted_summary="",
            error=f"Calculation error: {str(e)}"
        )

@tool
def calculate_financial_ratio(query: str) -> FinancialRatioResult:
    """
    Calculate and interpret financial ratios.

    Args:
        query: Query in format "NUMERATOR DENOMINATOR TYPE" (e.g., "82.50 5.50 pe")

    Returns:
        FinancialRatioResult: Ratio value, interpretation, and formatted summary
    """
    try:
        # Parse the query to extract numerator, denominator, ratio_type
        parts = query.strip().split()
        if len(parts) >= 3:
            numerator = float(parts[0])
            denominator = float(parts[1])
            ratio_type = parts[2].lower()
        elif len(parts) >= 2:
            numerator = float(parts[0])
            denominator = float(parts[1])
            ratio_type = "generic"
        else:
            raise ValueError("Query must contain at least numerator and denominator separated by spaces")

        if denominator == 0:
            return FinancialRatioResult(
                numerator=numerator,
                denominator=denominator,
                ratio_type=ratio_type,
                error="Denominator cannot be zero"
            )

        ratio_value = numerator / denominator

        # Contextual interpretations
        interpretations = {
            'pe': {
                "description": "Price-to-Earnings Ratio",
                "interpretation": f"P/E ratio of {ratio_value:.1f}",
                "context": "High (potentially overvalued)" if ratio_value > 25 else "Moderate" if ratio_value > 15 else "Low (potentially undervalued)"
            },
            'debt_to_equity': {
                "description": "Debt-to-Equity Ratio",
                "interpretation": f"D/E ratio of {ratio_value:.2f}",
                "context": "High leverage" if ratio_value > 1 else "Conservative leverage"
            },
            'current': {
                "description": "Current Ratio",
                "interpretation": f"Current ratio of {ratio_value:.2f}",
                "context": "Good liquidity" if ratio_value > 1.5 else "Potential liquidity concern"
            },
            'roe': {
                "description": "Return on Equity",
                "interpretation": f"ROE of {ratio_value*100:.1f}%",
                "context": "Strong" if ratio_value > 0.15 else "Average" if ratio_value > 0.10 else "Weak"
            },
            'generic': {
                "description": "Financial Ratio",
                "interpretation": f"Ratio value: {ratio_value:.3f}",
                "context": "Custom calculation"
            }
        }

        info = interpretations.get(ratio_type, interpretations['generic'])

        formatted_summary = f"{info['description']}: {ratio_value:.2f} - {info['context']}"

        return FinancialRatioResult(
            numerator=numerator,
            denominator=denominator,
            ratio_type=ratio_type,
            ratio_value=round(ratio_value, 4),
            description=info["description"],
            interpretation=info["interpretation"],
            context=info["context"],
            formatted_summary=formatted_summary,
            error=None
        )

    except Exception as e:
        return FinancialRatioResult(
            numerator=numerator,
            denominator=denominator,
            ratio_type=ratio_type,
            error=f"Calculation error: {str(e)}"
        )

# Initialize Tavily search tool
try:
    tavily_search = TavilySearch(
        api_key=config.TAVILY_API_KEY,
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False
    )
except Exception as e:
    print(f"Warning: Tavily search not available: {e}")
    tavily_search = None

# List of all financial tools
FINANCIAL_TOOLS = [
    get_stock_price,
    get_company_info,
    get_financial_history,
    calculate_compound_growth,
    calculate_financial_ratio,
]

# Add Tavily search if available
if tavily_search:
    FINANCIAL_TOOLS.append(tavily_search)

print(f"Loaded {len(FINANCIAL_TOOLS)} financial tools")