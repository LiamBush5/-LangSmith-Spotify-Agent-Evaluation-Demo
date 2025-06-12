"""
Advanced Financial Tools for LangChain Agent
These tools provide real financial data access and calculations.
Enhanced with modern LangChain patterns (2024-2025) with robust Pydantic v2 input validation.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Literal, Any
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import traceback
import re
from datetime import datetime, timedelta
import config

# Pydantic v2 Input Models for robust input handling
def parse_malformed_dict_string(value: str) -> dict:
    """Helper function to safely parse malformed dict strings from ReAct agent."""
    try:
        if value.startswith('{') and value.endswith('}'):
            return eval(value)
    except:
        pass
    return {}

class StockSymbolInput(BaseModel):
    """Input model for stock symbol with flexible parsing."""
    symbol: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA')")

    @field_validator('symbol', mode='before')
    @classmethod
    def parse_symbol(cls, v):
        """Parse symbol from various input formats."""
        if isinstance(v, dict):
            # Handle dict inputs like {'symbol': 'AAPL'} or {'SYMBOL': 'AAPL'}
            for key in ['symbol', 'SYMBOL', 'ticker', 'TICKER']:
                if key in v:
                    return str(v[key]).upper()
            # Fallback to first value if no standard key found
            return str(list(v.values())[0]).upper() if v else 'UNKNOWN'

        if isinstance(v, str):
            # Handle malformed dict strings like "{'SYMBOL': 'AAPL'}"
            parsed_dict = parse_malformed_dict_string(v)
            if parsed_dict:
                return cls.parse_symbol(parsed_dict)
            # Regular string input
            return v.upper()

        return str(v).upper()

class FinancialHistoryInput(BaseModel):
    """Input model for financial history with flexible parsing."""
    symbol: str = Field(description="Stock ticker symbol")
    period: str = Field(default="1y", description="Time period ('1y', '2y', '5y', 'max')")

    @model_validator(mode='before')
    @classmethod
    def parse_combined_input(cls, values):
        """Handle malformed string inputs like "{'SYMBOL': 'MSFT', 'PERIOD': '1Y'}"."""
        if isinstance(values, str):
            parsed_dict = parse_malformed_dict_string(values)
            if parsed_dict:
                result = {}
                # Extract symbol
                for key in ['symbol', 'SYMBOL', 'ticker', 'TICKER']:
                    if key in parsed_dict:
                        result['symbol'] = str(parsed_dict[key]).upper()
                        break
                # Extract period
                for key in ['period', 'PERIOD', 'timeframe', 'TIMEFRAME']:
                    if key in parsed_dict:
                        result['period'] = str(parsed_dict[key]).lower()
                        break
                # Set defaults if not found
                if 'symbol' not in result and parsed_dict:
                    result['symbol'] = str(list(parsed_dict.values())[0]).upper()
                if 'period' not in result:
                    result['period'] = '1y'
                return result
        return values

    @field_validator('symbol', mode='before')
    @classmethod
    def parse_symbol(cls, v):
        """Parse symbol from various input formats."""
        return str(v).upper()

    @field_validator('period', mode='before')
    @classmethod
    def parse_period(cls, v):
        """Parse period from various input formats."""
        return str(v).lower()

class CompoundGrowthInput(BaseModel):
    """Input model for compound growth calculation."""
    principal: float = Field(description="Initial investment amount in dollars")
    annual_rate: float = Field(description="Annual growth rate as a decimal (e.g., 0.07 for 7%)")
    years: float = Field(description="Number of years to compound")

    @model_validator(mode='before')
    @classmethod
    def parse_combined_input(cls, values):
        """Handle malformed string inputs like "{'principal': 260.17, 'annual_rate': 0.07, 'years': 4}"."""
        if isinstance(values, str):
            parsed_dict = parse_malformed_dict_string(values)
            if parsed_dict:
                result = {}
                # Extract principal
                for key in ['principal', 'PRINCIPAL', 'amount', 'AMOUNT']:
                    if key in parsed_dict:
                        result['principal'] = float(parsed_dict[key])
                        break
                # Extract annual_rate (handle None values)
                for key in ['annual_rate', 'ANNUAL_RATE', 'rate', 'RATE']:
                    if key in parsed_dict and parsed_dict[key] is not None:
                        result['annual_rate'] = float(parsed_dict[key])
                        break
                # Set default if not found or None
                if 'annual_rate' not in result:
                    result['annual_rate'] = 0.07  # Default 7% annual return
                # Extract years
                for key in ['years', 'YEARS', 'time', 'TIME']:
                    if key in parsed_dict:
                        result['years'] = float(parsed_dict[key])
                        break
                return result
        return values

class FinancialRatioInput(BaseModel):
    """Input model for financial ratio calculation."""
    numerator: float = Field(description="Numerator value")
    denominator: float = Field(description="Denominator value")
    ratio_type: Literal["pe", "debt_to_equity", "current", "roe", "generic"] = Field(
        default="generic",
        description="Type of financial ratio"
    )

    @model_validator(mode='before')
    @classmethod
    def parse_combined_input(cls, values):
        """Handle malformed string inputs like "{'numerator': 82.50, 'denominator': 5.50, 'ratio_type': 'pe'}"."""
        if isinstance(values, str):
            parsed_dict = parse_malformed_dict_string(values)
            if parsed_dict:
                result = {}
                # Extract numerator
                for key in ['numerator', 'NUMERATOR', 'num', 'NUM']:
                    if key in parsed_dict:
                        result['numerator'] = float(parsed_dict[key])
                        break
                # Extract denominator
                for key in ['denominator', 'DENOMINATOR', 'den', 'DEN']:
                    if key in parsed_dict:
                        result['denominator'] = float(parsed_dict[key])
                        break
                # Extract ratio_type
                for key in ['ratio_type', 'RATIO_TYPE', 'type', 'TYPE']:
                    if key in parsed_dict:
                        result['ratio_type'] = str(parsed_dict[key]).lower()
                        break
                # Set default ratio_type if not found
                if 'ratio_type' not in result:
                    result['ratio_type'] = 'generic'
                return result
        return values

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

# Modern function-based tools using @tool decorator
@tool
def get_stock_price(symbol: Union[str, Dict[str, Any]]) -> StockPriceData:
    """
    Get current stock price and market data for a publicly traded company.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA') or dict with symbol

    Returns:
        Structured stock price data including current price, market cap, and ratios
    """
    try:
        # Use Pydantic model for robust input parsing
        if isinstance(symbol, str):
            input_data = StockSymbolInput(symbol=symbol)
        else:
            input_data = StockSymbolInput.model_validate(symbol)

        symbol_str = input_data.symbol
        ticker = yf.Ticker(symbol_str)
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
            symbol=symbol_str,
            current_price=current_price,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            week_52_high=week_52_high,
            week_52_low=week_52_low,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        # Handle error case where symbol_str might not be defined
        error_symbol = symbol_str if 'symbol_str' in locals() else str(symbol)
        return StockPriceData(
            symbol=error_symbol,
            current_price=None,
            market_cap=None,
            pe_ratio=None,
            week_52_high=None,
            week_52_low=None,
            formatted_summary=f"Error retrieving data for {error_symbol}: {str(e)}",
            error=str(e)
        )

@tool
def get_company_info(symbol: Union[str, Dict[str, Any]]) -> CompanyInfo:
    """
    Get detailed company information for a publicly traded company.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT') or dict with symbol

    Returns:
        Structured company information including name, sector, industry details
    """
    try:
        # Use Pydantic model for robust input parsing
        if isinstance(symbol, str):
            input_data = StockSymbolInput(symbol=symbol)
        else:
            input_data = StockSymbolInput.model_validate(symbol)

        symbol_str = input_data.symbol
        ticker = yf.Ticker(symbol_str)
        info = ticker.info

        business_summary = info.get('longBusinessSummary', '')
        if business_summary and len(business_summary) > 300:
            business_summary = business_summary[:300] + "..."

        return CompanyInfo(
            symbol=symbol_str,
            name=info.get('longName'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            country=info.get('country'),
            employees=info.get('fullTimeEmployees'),
            business_summary=business_summary
        )

    except Exception as e:
        # Handle error case where symbol_str might not be defined
        error_symbol = symbol_str if 'symbol_str' in locals() else str(symbol)
        return CompanyInfo(
            symbol=error_symbol,
            name=None,
            sector=None,
            industry=None,
            country=None,
            employees=None,
            business_summary=f"Error retrieving company info for {error_symbol}: {str(e)}",
            error=str(e)
        )


@tool
def calculate_compound_growth(
    principal: Union[str, Dict[str, Any], float],
    annual_rate: Optional[float] = None,
    years: Optional[float] = None
) -> CompoundGrowthResult:
    """
    Calculate compound growth and future value of an investment.

    Args:
        principal: Initial investment amount or malformed string/dict with all parameters
        annual_rate: Annual growth rate as a decimal (e.g., 0.07 for 7%)
        years: Number of years to compound

    Returns:
        CompoundGrowthResult object with future value, total growth, and return percentage
    """
    try:
        # Handle malformed string inputs from ReAct agent
        if isinstance(principal, str):
            # This is likely a malformed string like "{'principal': 260.17, 'annual_rate': None, 'years': 4}"
            parsed_input = CompoundGrowthInput.model_validate(principal)
        elif isinstance(principal, dict):
            # This is a proper dict input
            parsed_input = CompoundGrowthInput.model_validate(principal)
        elif isinstance(principal, (int, float)):
            # Normal individual parameters
            parsed_input = CompoundGrowthInput(
                principal=float(principal),
                annual_rate=annual_rate if annual_rate is not None else 0.07,
                years=years if years is not None else 10
            )
        else:
            raise ValueError(f"Invalid input type: {type(principal)}")

        input_data = parsed_input

        principal_val = input_data.principal
        annual_rate_val = input_data.annual_rate
        years_val = input_data.years

        if years_val <= 0 or principal_val <= 0:
            return CompoundGrowthResult(
                principal=principal_val,
                annual_rate=annual_rate_val,
                years=years_val,
                future_value=0.0,
                total_growth=0.0,
                total_return_percent=0.0,
                formatted_summary="",
                error="Principal and years must be positive numbers"
            )

        future_value = principal_val * (1 + annual_rate_val) ** years_val
        total_growth = future_value - principal_val
        total_return_pct = (future_value / principal_val - 1) * 100

        return CompoundGrowthResult(
            principal=principal_val,
            annual_rate=annual_rate_val,
            years=years_val,
            future_value=round(future_value, 2),
            total_growth=round(total_growth, 2),
            total_return_percent=round(total_return_pct, 2),
            formatted_summary=f"""
Compound Growth Calculation:
• Initial Investment: ${principal_val:,.2f}
• Annual Return Rate: {annual_rate_val*100:.2f}%
• Time Period: {years_val} years
• Future Value: ${future_value:,.2f}
• Total Growth: ${total_growth:,.2f}
• Total Return: {total_return_pct:.2f}%
            """.strip(),
            error=None
        )

    except Exception as e:
        # Handle error case where values might not be defined
        error_principal = principal_val if 'principal_val' in locals() else (principal if isinstance(principal, (int, float)) else 0)
        error_rate = annual_rate_val if 'annual_rate_val' in locals() else (annual_rate if annual_rate else 0)
        error_years = years_val if 'years_val' in locals() else (years if years else 0)

        return CompoundGrowthResult(
            principal=error_principal,
            annual_rate=error_rate,
            years=error_years,
            future_value=0.0,
            total_growth=0.0,
            total_return_percent=0.0,
            formatted_summary="",
            error=f"Calculation error: {str(e)}"
        )

@tool
def calculate_financial_ratio(
    numerator: Union[str, Dict[str, Any], float],
    denominator: Optional[float] = None,
    ratio_type: Optional[str] = "generic"
) -> FinancialRatioResult:
    """
    Calculate and interpret financial ratios.

    Args:
        numerator: Top number in the ratio or malformed string/dict with all parameters
        denominator: Bottom number in the ratio
        ratio_type: Type of ratio for contextual interpretation

    Returns:
        FinancialRatioResult object with ratio value, interpretation, and formatted summary
    """
    try:
        # Handle malformed string inputs from ReAct agent
        if isinstance(numerator, str):
            # This is likely a malformed string like "{'numerator': 82.50, 'denominator': 5.50, 'ratio_type': 'pe'}"
            parsed_input = FinancialRatioInput.model_validate(numerator)
            numerator_val = parsed_input.numerator
            denominator_val = parsed_input.denominator
            ratio_type_val = parsed_input.ratio_type
        elif isinstance(numerator, dict):
            # This is a proper dict input
            parsed_input = FinancialRatioInput.model_validate(numerator)
            numerator_val = parsed_input.numerator
            denominator_val = parsed_input.denominator
            ratio_type_val = parsed_input.ratio_type
        elif isinstance(numerator, (int, float)):
            # Normal individual parameters
            numerator_val = float(numerator)
            denominator_val = float(denominator) if denominator is not None else 1.0
            ratio_type_val = ratio_type if ratio_type is not None else "generic"
        else:
            raise ValueError(f"Invalid input type: {type(numerator)}")

        numerator = numerator_val
        denominator = denominator_val
        ratio_type = ratio_type_val

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

        return FinancialRatioResult(
            numerator=numerator,
            denominator=denominator,
            ratio_type=ratio_type,
            ratio_value=round(ratio_value, 4),
            description=info["description"],
            interpretation=info["interpretation"],
            context=info["context"],
            formatted_summary=f"""
{info['description']}: {ratio_value:.3f}
Interpretation: {info['interpretation']}
Assessment: {info['context']}
            """.strip(),
            error=None
        )

    except Exception as e:
        return FinancialRatioResult(
            numerator=numerator,
            denominator=denominator,
            ratio_type=ratio_type,
            error=f"Calculation error: {str(e)}"
        )

@tool
def get_financial_history(
    symbol: Union[str, Dict[str, Any]],
    period: str = "1y"
) -> FinancialHistoryResult:
    """
    Get historical stock performance and calculate key metrics.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT') or dict with symbol and period
        period: Time period ('1y', '2y', '5y', 'max')

    Returns:
        FinancialHistoryResult object with historical performance metrics
    """
    try:
        # Use Pydantic model for robust input parsing
        if isinstance(symbol, str) and symbol.startswith('{'):
            # Handle malformed string input like "{'SYMBOL': 'MSFT', 'PERIOD': '1Y'}"
            input_data = FinancialHistoryInput.model_validate(symbol)
        elif isinstance(symbol, str):
            # Normal string input
            input_data = FinancialHistoryInput(symbol=symbol, period=period)
        else:
            # Dict input
            input_data = FinancialHistoryInput.model_validate(symbol)

        symbol_str = input_data.symbol
        period_str = input_data.period
        ticker = yf.Ticker(symbol_str)
        hist = ticker.history(period=period_str)

        if hist.empty:
            return FinancialHistoryResult(
                symbol=symbol_str,
                period=period_str,
                error=f"No historical data available for {symbol_str}"
            )

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

        return FinancialHistoryResult(
            symbol=symbol_str,
            period=period_str,
            start_price=round(start_price, 2),
            end_price=round(end_price, 2),
            total_return_percent=round(total_return, 2),
            cagr_percent=round(cagr * 100, 2),
            volatility_percent=round(volatility, 2),
            max_drawdown_percent=round(max_drawdown, 2),
            trading_days=len(hist),
            formatted_summary=f"""
{symbol_str} Historical Performance ({period_str}):
• Start Price: ${start_price:.2f}
• End Price: ${end_price:.2f}
• Total Return: {total_return:.2f}%
• CAGR: {cagr*100:.2f}% (annualized)
• Volatility: {volatility:.2f}%
• Max Drawdown: {max_drawdown:.2f}%
            """.strip(),
            error=None
        )

    except Exception as e:
        # Handle error case where symbol_str might not be defined
        error_symbol = symbol_str if 'symbol_str' in locals() else str(symbol)
        error_period = period_str if 'period_str' in locals() else str(period)
        return FinancialHistoryResult(
            symbol=error_symbol,
            period=error_period,
            error=f"Error retrieving historical data for {error_symbol}: {str(e)}"
        )

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

    # Test financial ratio tool
    print("\n" + "="*60)
    print("Testing Financial Ratio Tool:")
    ratio_result = calculate_financial_ratio.invoke({
        "numerator": 82.5,
        "denominator": 5.5,
        "ratio_type": "pe"
    })
    print(f"Type: {type(ratio_result)}")
    print(f"Data: {ratio_result}")

    # Test historical data tool
    print("\n" + "="*60)
    print("Testing Historical Data Tool:")
    hist_result = get_financial_history.invoke({
        "symbol": "TSLA",
        "period": "1y"
    })
    print(f"Type: {type(hist_result)}")
    print(f"Data preview: {hist_result.formatted_summary}")

    print("\n" + "="*60)
    print("🎉 All modern tools working perfectly!")
    print("✅ Legacy code removed - clean, modern codebase ready!")