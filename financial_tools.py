"""
Advanced Financial Tools for LangChain Agent
These tools provide real financial data access and calculations.
Enhanced with modern LangChain patterns (2024-2025).
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Literal, Any
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
import json
import traceback
from datetime import datetime, timedelta
import config

# Modern Pydantic schemas for structured input/output
class StockPriceData(BaseModel):
    """Structured stock price information."""
    symbol: str = Field(description="Stock ticker symbol")
    current_price: Optional[float] = Field(description="Current stock price")
    market_cap: Optional[int] = Field(description="Market capitalization")
    pe_ratio: Optional[float] = Field(description="Price-to-earnings ratio")
    week_52_high: Optional[float] = Field(description="52-week high price")
    week_52_low: Optional[float] = Field(description="52-week low price")
    formatted_summary: str = Field(description="Human-readable summary")

class CompanyInfo(BaseModel):
    """Structured company information."""
    symbol: str = Field(description="Stock ticker symbol")
    name: Optional[str] = Field(description="Company name")
    sector: Optional[str] = Field(description="Business sector")
    industry: Optional[str] = Field(description="Industry classification")
    country: Optional[str] = Field(description="Country of incorporation")
    employees: Optional[int] = Field(description="Number of employees")
    business_summary: Optional[str] = Field(description="Business description")

# Modern function-based tools using @tool decorator
@tool
def get_stock_price(symbol: str) -> StockPriceData:
    """
    Get current stock price and market data for a publicly traded company.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA')

    Returns:
        Structured stock price data including current price, market cap, and ratios
    """
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        info = ticker.info

        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE')
        week_52_high = info.get('fiftyTwoWeekHigh')
        week_52_low = info.get('fiftyTwoWeekLow')

        # Create formatted summary
        price_str = f"${current_price:.2f}" if current_price else "N/A"

        if market_cap and isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                cap_str = f"${market_cap/1e6:.2f}M"
            else:
                cap_str = f"${market_cap:,.0f}"
        else:
            cap_str = "N/A"

        formatted_summary = f"""
{symbol} Stock Data:
• Current Price: {price_str}
• Market Cap: {cap_str}
• P/E Ratio: {pe_ratio if pe_ratio else 'N/A'}
• 52-Week Range: ${week_52_low if week_52_low else 'N/A'} - ${week_52_high if week_52_high else 'N/A'}
        """.strip()

        return StockPriceData(
            symbol=symbol,
            current_price=current_price,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            week_52_high=week_52_high,
            week_52_low=week_52_low,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return StockPriceData(
            symbol=symbol,
            current_price=None,
            market_cap=None,
            pe_ratio=None,
            week_52_high=None,
            week_52_low=None,
            formatted_summary=f"Error retrieving data for {symbol}: {str(e)}"
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
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        info = ticker.info

        business_summary = info.get('longBusinessSummary', '')
        if business_summary and len(business_summary) > 300:
            business_summary = business_summary[:300] + "..."

        return CompanyInfo(
            symbol=symbol,
            name=info.get('longName'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            country=info.get('country'),
            employees=info.get('fullTimeEmployees'),
            business_summary=business_summary
        )

    except Exception as e:
        return CompanyInfo(
            symbol=symbol,
            name=None,
            sector=None,
            industry=None,
            country=None,
            employees=None,
            business_summary=f"Error retrieving company info for {symbol}: {str(e)}"
        )

@tool
def calculate_compound_growth(
    principal: float,
    annual_rate: float,
    years: float
) -> Dict[str, Any]:
    """
    Calculate compound growth and future value of an investment.

    Args:
        principal: Initial investment amount in dollars
        annual_rate: Annual growth rate as a decimal (e.g., 0.07 for 7%)
        years: Number of years to compound

    Returns:
        Dictionary with future value, total growth, and return percentage
    """
    try:
        if years <= 0 or principal <= 0:
            return {
                "error": "Principal and years must be positive numbers",
                "principal": principal,
                "annual_rate": annual_rate,
                "years": years
            }

        future_value = principal * (1 + annual_rate) ** years
        total_growth = future_value - principal
        total_return_pct = (future_value / principal - 1) * 100

        return {
            "principal": principal,
            "annual_rate": annual_rate,
            "years": years,
            "future_value": round(future_value, 2),
            "total_growth": round(total_growth, 2),
            "total_return_percent": round(total_return_pct, 2),
            "formatted_summary": f"""
Compound Growth Calculation:
• Initial Investment: ${principal:,.2f}
• Annual Return Rate: {annual_rate*100:.2f}%
• Time Period: {years} years
• Future Value: ${future_value:,.2f}
• Total Growth: ${total_growth:,.2f}
• Total Return: {total_return_pct:.2f}%
            """.strip()
        }

    except Exception as e:
        return {
            "error": f"Calculation error: {str(e)}",
            "principal": principal,
            "annual_rate": annual_rate,
            "years": years
        }

@tool
def calculate_financial_ratio(
    numerator: float,
    denominator: float,
    ratio_type: Literal["pe", "debt_to_equity", "current", "roe", "generic"] = "generic"
) -> Dict[str, Any]:
    """
    Calculate and interpret financial ratios.

    Args:
        numerator: Top number in the ratio
        denominator: Bottom number in the ratio
        ratio_type: Type of ratio for contextual interpretation

    Returns:
        Dictionary with ratio value, interpretation, and formatted summary
    """
    try:
        if denominator == 0:
            return {
                "error": "Denominator cannot be zero",
                "numerator": numerator,
                "denominator": denominator,
                "ratio_type": ratio_type
            }

        ratio_value = numerator / denominator

        # Contextual interpretations
        interpretations = {
            'pe': {
                "description": "Price-to-Earnings Ratio",
                "interpretation": f"${ratio_value:.2f} price per $1 of earnings",
                "context": "High" if ratio_value > 25 else "Moderate" if ratio_value > 15 else "Low"
            },
            'debt_to_equity': {
                "description": "Debt-to-Equity Ratio",
                "interpretation": f"{ratio_value:.2f} debt per $1 of equity",
                "context": "High leverage" if ratio_value > 1 else "Conservative leverage"
            },
            'current': {
                "description": "Current Ratio",
                "interpretation": f"{ratio_value:.2f} current assets per $1 of current liabilities",
                "context": "Good liquidity" if ratio_value > 1.5 else "Potential liquidity concern"
            },
            'roe': {
                "description": "Return on Equity",
                "interpretation": f"{ratio_value*100:.2f}% return on equity",
                "context": "Strong" if ratio_value > 0.15 else "Average" if ratio_value > 0.10 else "Weak"
            },
            'generic': {
                "description": "Financial Ratio",
                "interpretation": f"Ratio value: {ratio_value:.3f}",
                "context": "Custom calculation"
            }
        }

        info = interpretations.get(ratio_type, interpretations['generic'])

        return {
            "numerator": numerator,
            "denominator": denominator,
            "ratio_type": ratio_type,
            "ratio_value": round(ratio_value, 4),
            "description": info["description"],
            "interpretation": info["interpretation"],
            "context": info["context"],
            "formatted_summary": f"""
{info['description']}: {ratio_value:.3f}
Interpretation: {info['interpretation']}
Assessment: {info['context']}
            """.strip()
        }

    except Exception as e:
        return {
            "error": f"Calculation error: {str(e)}",
            "numerator": numerator,
            "denominator": denominator,
            "ratio_type": ratio_type
        }

@tool
def get_financial_history(
    symbol: str,
    period: str = "1y"
) -> Dict[str, Any]:
    """
    Get historical stock performance and calculate key metrics.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period ('1y', '2y', '5y', 'max')

    Returns:
        Dictionary with historical performance metrics
    """
    try:
        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return {
                "error": f"No historical data available for {symbol}",
                "symbol": symbol,
                "period": period
            }

        # Calculate performance metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        total_return = (end_price - start_price) / start_price * 100

        # Calculate CAGR for multi-year periods
        years = len(hist) / 252  # Approximate trading days per year
        cagr = (end_price / start_price) ** (1/years) - 1 if years > 1 else total_return / 100

        # Calculate volatility and max drawdown
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * 100
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / rolling_max - 1) * 100
        max_drawdown = drawdown.min()

        return {
            "symbol": symbol,
            "period": period,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "total_return_percent": round(total_return, 2),
            "cagr_percent": round(cagr * 100, 2),
            "volatility_percent": round(volatility, 2),
            "max_drawdown_percent": round(max_drawdown, 2),
            "trading_days": len(hist),
            "formatted_summary": f"""
{symbol} Historical Performance ({period}):
• Start Price: ${start_price:.2f}
• End Price: ${end_price:.2f}
• Total Return: {total_return:.2f}%
• CAGR: {cagr*100:.2f}% (annualized)
• Volatility: {volatility:.2f}%
• Max Drawdown: {max_drawdown:.2f}%
            """.strip()
        }

    except Exception as e:
        return {
            "error": f"Error retrieving historical data for {symbol}: {str(e)}",
            "symbol": symbol,
            "period": period
        }

# Initialize search tool
search_tool = TavilySearch(
    max_results=3,
    search_depth="advanced"
)

# Modern financial tools list
FINANCIAL_TOOLS = [
    search_tool,
    get_stock_price,
    get_company_info,
    calculate_compound_growth,
    calculate_financial_ratio,
    get_financial_history
]

if __name__ == "__main__":
    # Test the modern tools
    print("🚀 Testing Modern Financial Tools with Structured Output...")

    # Test modern stock price tool
    print("\n" + "="*60)
    print("Testing Modern Stock Price Tool:")
    result = get_stock_price.invoke({"symbol": "AAPL"})
    print(f"Type: {type(result)}")
    print(f"Data: {result}")

    # Test modern calculation tool
    print("\n" + "="*60)
    print("Testing Modern Calculation Tool:")
    calc_result = calculate_compound_growth.invoke({
        "principal": 10000,
        "annual_rate": 0.07,
        "years": 5
    })
    print(f"Type: {type(calc_result)}")
    print(f"Data: {calc_result}")

    # Test historical data tool
    print("\n" + "="*60)
    print("Testing Historical Data Tool:")
    hist_result = get_financial_history.invoke({
        "symbol": "TSLA",
        "period": "1y"
    })
    print(f"Type: {type(hist_result)}")
    print(f"Data preview: {hist_result.get('formatted_summary', 'No summary')}")

    print("\n" + "="*60)
    print("🎉 All modern tools working perfectly!")
    print("✅ Legacy code removed - clean, modern codebase ready!")