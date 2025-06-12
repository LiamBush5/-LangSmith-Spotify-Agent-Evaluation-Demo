"""
Advanced Financial Tools for LangChain Agent
These tools provide real financial data access and calculations.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
import json
import traceback
from datetime import datetime, timedelta
import config

class FinancialDataTool(BaseTool):
    """Tool for retrieving financial data using yfinance API"""

    name: str = "financial_data_api"
    description: str = """
    Retrieves comprehensive financial data for publicly traded companies.
    Use this for: stock prices, financial statements, company info, historical data.

    Input should be a JSON with:
    - symbol: stock ticker (e.g., 'AAPL', 'TSLA')
    - data_type: 'price' | 'financials' | 'info' | 'history'
    - period: for history - '1y', '2y', '5y', 'max'
    - metric: for financials - 'revenue', 'net_income', 'total_debt', etc.

    Example: {"symbol": "AAPL", "data_type": "financials", "metric": "revenue"}
    """

    def _run(self, query: str) -> str:
        try:
            # Parse input
            if isinstance(query, str):
                params = json.loads(query)
            else:
                params = query

            symbol = params.get('symbol', '').upper()
            data_type = params.get('data_type', 'info')
            period = params.get('period', '1y')
            metric = params.get('metric', '')

            if not symbol:
                return "Error: Symbol is required"

            # Get ticker object
            ticker = yf.Ticker(symbol)

            if data_type == 'price':
                # Current price data
                info = ticker.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                market_cap = info.get('marketCap', 'N/A')
                pe_ratio = info.get('trailingPE', 'N/A')

                return f"""
Current data for {symbol}:
- Current Price: ${current_price}
- Market Cap: ${market_cap:,} if market_cap != 'N/A' else 'N/A'
- P/E Ratio: {pe_ratio}
- 52-week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
- 52-week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
                """.strip()

            elif data_type == 'financials':
                # Financial statements
                financials = ticker.financials
                if financials.empty:
                    return f"No financial data available for {symbol}"

                # Get the most recent year's data
                latest_year = financials.columns[0]

                if metric:
                    # Look for specific metric
                    metric_variations = [
                        metric,
                        metric.title(),
                        metric.replace('_', ' ').title(),
                        'Total Revenue' if 'revenue' in metric.lower() else metric,
                        'Net Income' if 'net_income' in metric.lower() else metric
                    ]

                    for var in metric_variations:
                        if var in financials.index:
                            value = financials.loc[var, latest_year]
                            return f"{symbol} {var} for {latest_year.strftime('%Y')}: ${value:,.0f}"

                    return f"Metric '{metric}' not found. Available metrics: {list(financials.index[:10])}"
                else:
                    # Return key financial metrics
                    key_metrics = {}
                    metric_map = {
                        'Total Revenue': 'Revenue',
                        'Net Income': 'Net Income',
                        'Gross Profit': 'Gross Profit',
                        'Operating Income': 'Operating Income'
                    }

                    for financial_name, display_name in metric_map.items():
                        if financial_name in financials.index:
                            value = financials.loc[financial_name, latest_year]
                            key_metrics[display_name] = f"${value:,.0f}"

                    result = f"{symbol} Key Financials for {latest_year.strftime('%Y')}:\n"
                    for metric, value in key_metrics.items():
                        result += f"- {metric}: {value}\n"

                    return result.strip()

            elif data_type == 'history':
                # Historical price data
                hist = ticker.history(period=period)
                if hist.empty:
                    return f"No historical data available for {symbol}"

                # Calculate returns and trends
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                total_return = (end_price - start_price) / start_price * 100

                # Calculate CAGR for multi-year periods
                years = len(hist) / 252  # Approximate trading days per year
                cagr = (end_price / start_price) ** (1/years) - 1 if years > 1 else total_return / 100

                return f"""
{symbol} Historical Performance ({period}):
- Start Price: ${start_price:.2f}
- End Price: ${end_price:.2f}
- Total Return: {total_return:.2f}%
- CAGR: {cagr*100:.2f}% (annualized)
- Volatility (std): {hist['Close'].pct_change().std()*100:.2f}%
- Max Drawdown: {((hist['Close'] / hist['Close'].cummax() - 1).min())*100:.2f}%
                """.strip()

            elif data_type == 'info':
                # Company information
                info = ticker.info
                return f"""
{symbol} Company Information:
- Name: {info.get('longName', 'N/A')}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Country: {info.get('country', 'N/A')}
- Employees: {info.get('fullTimeEmployees', 'N/A')}
- Business Summary: {info.get('longBusinessSummary', 'N/A')[:200]}...
                """.strip()

            else:
                return f"Unknown data_type: {data_type}. Use: price, financials, info, or history"

        except Exception as e:
            return f"Error retrieving financial data: {str(e)}\nTraceback: {traceback.format_exc()}"

class FinancialCalculatorTool(BaseTool):
    """Advanced financial calculations tool"""

    name: str = "financial_calculator"
    description: str = """
    Performs sophisticated financial calculations.

    Supported calculations:
    - compound_growth: CAGR, future value
    - ratios: P/E, debt-to-equity, ROE, etc.
    - valuation: DCF components, multiples
    - portfolio: returns, risk metrics

    Input JSON format:
    {"calc_type": "compound_growth", "principal": 10000, "rate": 0.07, "years": 5}
    {"calc_type": "cagr", "start_value": 100, "end_value": 150, "years": 3}
    {"calc_type": "ratio", "numerator": 50, "denominator": 2.5, "ratio_type": "pe"}
    """

    def _run(self, query: str) -> str:
        try:
            if isinstance(query, str):
                params = json.loads(query)
            else:
                params = query

            calc_type = params.get('calc_type', '').lower()

            if calc_type == 'compound_growth':
                principal = float(params.get('principal', 0))
                rate = float(params.get('rate', 0))
                years = float(params.get('years', 0))

                future_value = principal * (1 + rate) ** years
                total_growth = future_value - principal
                total_return_pct = (future_value / principal - 1) * 100

                return f"""
Compound Growth Calculation:
- Initial Investment: ${principal:,.2f}
- Annual Return Rate: {rate*100:.2f}%
- Time Period: {years} years
- Future Value: ${future_value:,.2f}
- Total Growth: ${total_growth:,.2f}
- Total Return: {total_return_pct:.2f}%
                """.strip()

            elif calc_type == 'cagr':
                start_value = float(params.get('start_value', 0))
                end_value = float(params.get('end_value', 0))
                years = float(params.get('years', 0))

                if years <= 0 or start_value <= 0:
                    return "Error: Years and start_value must be positive"

                cagr = (end_value / start_value) ** (1/years) - 1
                total_return = (end_value / start_value - 1) * 100

                return f"""
CAGR Calculation:
- Starting Value: ${start_value:,.2f}
- Ending Value: ${end_value:,.2f}
- Time Period: {years} years
- CAGR: {cagr*100:.2f}% per year
- Total Return: {total_return:.2f}%
                """.strip()

            elif calc_type == 'ratio':
                numerator = float(params.get('numerator', 0))
                denominator = float(params.get('denominator', 1))
                ratio_type = params.get('ratio_type', 'generic')

                if denominator == 0:
                    return "Error: Denominator cannot be zero"

                ratio_value = numerator / denominator

                ratio_interpretations = {
                    'pe': f"P/E Ratio: {ratio_value:.2f} (Price per $1 of earnings)",
                    'debt_to_equity': f"Debt-to-Equity: {ratio_value:.2f} ({'High leverage' if ratio_value > 1 else 'Conservative leverage'})",
                    'current': f"Current Ratio: {ratio_value:.2f} ({'Good liquidity' if ratio_value > 1.5 else 'Potential liquidity concern'})",
                    'roe': f"ROE: {ratio_value*100:.2f}% ({'Strong' if ratio_value > 0.15 else 'Average' if ratio_value > 0.10 else 'Weak'} returns)",
                    'generic': f"Ratio: {ratio_value:.2f}"
                }

                return ratio_interpretations.get(ratio_type, ratio_interpretations['generic'])

            elif calc_type == 'portfolio_metrics':
                returns = params.get('returns', [])  # List of returns
                if not returns:
                    return "Error: Returns list is required"

                returns_array = np.array(returns)

                mean_return = np.mean(returns_array)
                volatility = np.std(returns_array)
                sharpe_ratio = mean_return / volatility if volatility > 0 else 0
                max_drawdown = np.min(returns_array)

                return f"""
Portfolio Metrics:
- Average Return: {mean_return*100:.2f}%
- Volatility (Std Dev): {volatility*100:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Worst Period: {max_drawdown*100:.2f}%
                """.strip()

            else:
                return f"Unknown calculation type: {calc_type}. Supported: compound_growth, cagr, ratio, portfolio_metrics"

        except Exception as e:
            return f"Error in financial calculation: {str(e)}"

# Initialize search tool
search_tool = TavilySearch(
    max_results=3,
    search_depth="advanced"
)

# Create tool instances
financial_data_tool = FinancialDataTool()
financial_calculator = FinancialCalculatorTool()

# Tool list for agent
FINANCIAL_TOOLS = [
    search_tool,
    financial_data_tool,
    financial_calculator
]

if __name__ == "__main__":
    # Test the tools
    print("Testing Financial Tools...")

    # Test financial data tool
    print("\n" + "="*50)
    print("Testing Financial Data Tool:")
    test_query = '{"symbol": "AAPL", "data_type": "price"}'
    result = financial_data_tool.run(test_query)
    print(result)

    # Test calculator tool
    print("\n" + "="*50)
    print("Testing Financial Calculator:")
    calc_query = '{"calc_type": "compound_growth", "principal": 10000, "rate": 0.07, "years": 5}'
    calc_result = financial_calculator.run(calc_query)
    print(calc_result)