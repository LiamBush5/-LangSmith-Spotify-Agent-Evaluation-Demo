"""
Financial Research Agent with LangSmith Tracing

Advanced financial agent using ReAct pattern with comprehensive tool access.
All agent, tool, and LLM calls are captured in LangSmith traces for observability.
"""
import uuid
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langsmith import Client
from langsmith.run_helpers import traceable
from financial_tools import FINANCIAL_TOOLS
import config
import pandas as pd

# Initialize LangSmith client
client = Client()

class FinancialAgent:
    """
    Financial research agent with comprehensive tool access and reasoning.
    """

    def __init__(self):
        """Initialize the financial agent with tools and LLM."""
        self.tools = FINANCIAL_TOOLS
        self.llm = config.get_chat_model()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.AGENT_MAX_ITERATIONS,
            max_execution_time=config.AGENT_MAX_EXECUTION_TIME,
            return_intermediate_steps=True
        )

    def _create_agent(self):
        """Create ReAct agent with financial expertise."""
        prompt_template = """You are a sophisticated financial research analyst with access to real-time financial data and advanced calculation tools.

AVAILABLE TOOLS:
{tools}

TOOL INPUT FORMAT:
- get_stock_price: Use MSFT (symbol only)
- get_company_info: Use AAPL (symbol only)
- get_financial_history: Use AAPL 5y (symbol and period)
- calculate_compound_growth: Use 10000 0.07 10 (principal rate years)
- calculate_financial_ratio: Use 82.50 5.50 pe (numerator denominator type)
- tavily_search_results_json: Use Microsoft stock price (search query)

FINANCIAL TOOL USAGE GUIDELINES:
- For current stock prices: Use get_stock_price(symbol) - returns structured StockPriceData
- For company information: Use get_company_info(symbol) - returns structured CompanyInfo
- For historical data: Use get_financial_history("SYMBOL PERIOD") - period can be "1y", "2y", "5y", "max"
- For compound growth: Use calculate_compound_growth("PRINCIPAL RATE YEARS")
- For financial ratios: Use calculate_financial_ratio("NUMERATOR DENOMINATOR TYPE")
- For recent news/events: Use tavily_search_results_json
- STRUCTURED OUTPUT: Financial tools return validated Pydantic models for reliability
- MINIMIZE TOOL CALLS: Plan your tool usage efficiently before starting
- Combine multiple data points in single tool calls when possible
- Always verify financial figures with reliable sources
- Show your reasoning step-by-step

NEVER use JSON format - use space-separated values for multi-parameter tools!

ANALYSIS STANDARDS:
- Provide specific numerical data when available
- Calculate relevant metrics (CAGR, ratios, percentages)
- Include context about market conditions
- Show reasoning step-by-step
- Format financial figures clearly

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

        return create_react_agent(self.llm, self.tools, prompt)

    def _serialize_tool_output(self, output: Any) -> Any:
        """Convert Pydantic models to dictionaries for LangSmith compatibility."""
        from pydantic import BaseModel

        if isinstance(output, BaseModel):
            return output.model_dump()
        elif isinstance(output, list):
            return [self._serialize_tool_output(item) for item in output]
        elif isinstance(output, dict):
            return {k: self._serialize_tool_output(v) for k, v in output.items()}
        else:
            return output

    @traceable(
        run_type="chain",
        name="FinancialAgentAnalysis",
        tags=["financial_agent", "analysis"],
        metadata={"agent_version": "v2.1"}
    )
    def analyze_query(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a financial question and return structured results.

        Args:
            query: The financial question to analyze
            thread_id: (Deprecated) Previously used for thread grouping

        Returns:
            Dictionary with agent response, reasoning steps, and tool usage metadata
        """
        print(f"\n🔍 Analyzing Query: {query}")
        print("="*60)

        if thread_id is None:
            thread_id = str(uuid.uuid4())

        try:
            # Execute the agent
            result = self.agent_executor.invoke(
                {"input": query},
                config={
                    "metadata": {
                        "query": query,
                        "agent_type": "financial_research",
                    },
                    "tags": ["financial_agent"]
                }
            )

            # Extract information
            response = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Process intermediate steps
            tool_trajectory = []
            reasoning_steps = []

            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    tool_name = action.tool if hasattr(action, 'tool') else "unknown"
                    tool_input = action.tool_input if hasattr(action, 'tool_input') else ""

                    tool_trajectory.append(tool_name)
                    serialized_observation = self._serialize_tool_output(observation)

                    reasoning_steps.append({
                        "tool": tool_name,
                        "input": str(tool_input),
                        "output": str(serialized_observation)[:200] + "..." if len(str(serialized_observation)) > 200 else str(serialized_observation)
                    })

            # Compile results
            analysis_result = {
                "response": response,
                "tool_trajectory": tool_trajectory,
                "reasoning_steps": reasoning_steps,
                "total_tool_calls": len(tool_trajectory),
                "unique_tools_used": list(set(tool_trajectory)),
                "query": query,
                "thread_id": thread_id
            }

            print(f"\nAnalysis Complete!")
            print(f"Tools Used: {', '.join(analysis_result['unique_tools_used'])}")
            print(f"Total Tool Calls: {analysis_result['total_tool_calls']}")

            if analysis_result['total_tool_calls'] >= config.AGENT_MAX_ITERATIONS * 0.8:
                print(f"⚠️  Warning: High tool usage ({analysis_result['total_tool_calls']}/{config.AGENT_MAX_ITERATIONS})")

            return analysis_result

        except Exception as e:
            error_result = {
                "response": f"Error during analysis: {str(e)}",
                "tool_trajectory": [],
                "reasoning_steps": [],
                "total_tool_calls": 0,
                "unique_tools_used": [],
                "query": query,
                "thread_id": thread_id,
                "error": True
            }

            print(f"❌ Analysis failed: {str(e)}")
            return error_result


@traceable(
    run_type="chain",
    name="FinancialAgentEvaluation",
    tags=["financial_agent", "evaluation"],
    metadata={"evaluation_run": True, "agent_version": "v2.1"}
)
def run_financial_agent(inputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Main entry point for financial agent evaluation.

    Args:
        inputs: Dictionary containing the query and additional parameters

    Returns:
        Dictionary containing the agent's response and metadata
    """
    # Handle multiple input formats
    query = inputs.get("input", inputs.get("query", inputs.get("question", "")))

    if not query:
        return {
            "response": "No query provided",
            "error": True,
            "tool_trajectory": [],
            "reasoning_steps": [],
            "total_tool_calls": 0,
            "unique_tools_used": [],
            "thread_id": None
        }

    print(f"\n{'='*80}")
    print("🤖 FINANCIAL AGENT EVALUATION")
    print(f"Query: {query}")
    print(f"{'='*80}")

    agent = FinancialAgent()
    result = agent.analyze_query(query)

    # Add timestamp
    result.update({
        "timestamp": str(pd.Timestamp.now())
    })

    print("\n✅ Evaluation Complete")
    print(f"Response Length: {len(result.get('response', ''))}")
    print(f"Tools Used: {result.get('total_tool_calls', 0)}")

    return result

@traceable(
    run_type="chain",
    name="FinancialAgentEvaluationWrapper",
    project_name=config.LANGSMITH_PROJECT,
    tags=["financial_agent", "evaluation", "wrapper"]
)
def run_financial_agent_with_project_routing(inputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Wrapper function that ensures traces go to the correct project.
    This is the function that should be passed to evaluate().
    """
    return run_financial_agent(inputs)







###more code

"""
Financial Tools for LangChain Agent

Professional financial analysis tools with structured outputs using Pydantic models.
All tools follow LangChain best practices for reliable agent integration.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
import config

# Pydantic models for structured outputs
class StockPriceData(BaseModel):
    """Stock price and key metrics."""
    symbol: str = Field(description="Stock ticker symbol")
    current_price: Optional[float] = Field(description="Current stock price")
    market_cap: Optional[int] = Field(description="Market capitalization")
    pe_ratio: Optional[float] = Field(description="Price-to-earnings ratio")
    week_52_high: Optional[float] = Field(description="52-week high price")
    week_52_low: Optional[float] = Field(description="52-week low price")
    formatted_summary: str = Field(description="Human-readable summary")
    error: Optional[str] = None

class CompanyInfo(BaseModel):
    """Company information and fundamentals."""
    symbol: str = Field(description="Stock ticker symbol")
    name: Optional[str] = Field(description="Company name")
    sector: Optional[str] = Field(description="Business sector")
    industry: Optional[str] = Field(description="Industry classification")
    country: Optional[str] = Field(description="Country of incorporation")
    employees: Optional[int] = Field(description="Number of employees")
    business_summary: Optional[str] = Field(description="Business description")
    error: Optional[str] = None

class FinancialHistoryResult(BaseModel):
    """Historical performance analysis."""
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
    """Compound growth calculation results."""
    principal: float
    annual_rate: float
    years: float
    future_value: float
    total_growth: float
    total_return_percent: float
    formatted_summary: str
    error: Optional[str] = None

class FinancialRatioResult(BaseModel):
    """Financial ratio calculation and interpretation."""
    numerator: float
    denominator: float
    ratio_type: str
    ratio_value: Optional[float] = None
    description: Optional[str] = None
    interpretation: Optional[str] = None
    context: Optional[str] = None
    formatted_summary: Optional[str] = None
    error: Optional[str] = None

@tool
def get_stock_price(symbol: str) -> StockPriceData:
    """
    Get current stock price and key metrics.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA')

    Returns:
        Structured stock price data with current price, market cap, and ratios
    """
    try:
        symbol = symbol.upper().strip()
        ticker = yf.Ticker(symbol)
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        week_52_high = info.get('fiftyTwoWeekHigh')
        week_52_low = info.get('fiftyTwoWeekLow')

        if market_cap:
            market_cap = int(market_cap)

        # Create summary
        summary_parts = [f"Stock: {symbol}"]
        if current_price:
            summary_parts.append(f"Price: ${current_price:.2f}")
        if market_cap:
            summary_parts.append(f"Market Cap: ${market_cap:,}")
        if pe_ratio:
            summary_parts.append(f"P/E: {pe_ratio:.2f}")
        if week_52_high and week_52_low:
            summary_parts.append(f"52W Range: ${week_52_low:.2f} - ${week_52_high:.2f}")

        return StockPriceData(
            symbol=symbol,
            current_price=current_price,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            week_52_high=week_52_high,
            week_52_low=week_52_low,
            formatted_summary=" | ".join(summary_parts)
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
    Get detailed company information.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        Structured company information including sector, industry, and business details
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
            business_summary=info.get('longBusinessSummary', 'No business summary available')
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
    Get historical performance and calculate key metrics.

    Args:
        query: Format "SYMBOL PERIOD" (e.g., "AAPL 5y", "TSLA 2y")

    Returns:
        Historical performance analysis with returns, CAGR, volatility, and drawdown
    """
    try:
        # Parse query
        parts = query.strip().split()
        symbol = parts[0].upper() if parts else query.upper()
        period = parts[1].lower() if len(parts) > 1 else "1y"

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

        # CAGR calculation
        years = len(hist) / 252  # Trading days per year
        cagr = ((end_price / start_price) ** (1/years) - 1) * 100 if years > 0 else 0

        # Volatility (annualized)
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # Max drawdown
        rolling_max = hist['Close'].expanding().max()
        drawdown = (hist['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        formatted_summary = f"{symbol} Performance ({period}): Total Return: {total_return:.2f}%, CAGR: {cagr:.2f}%, Volatility: {volatility:.2f}%, Max Drawdown: {max_drawdown:.2f}%"

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
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return FinancialHistoryResult(
            symbol=symbol if 'symbol' in locals() else 'UNKNOWN',
            period=period if 'period' in locals() else 'UNKNOWN',
            formatted_summary=f"Error retrieving financial history: {str(e)}",
            error=str(e)
        )

@tool
def calculate_compound_growth(query: str) -> CompoundGrowthResult:
    """
    Calculate compound growth and future value.

    Args:
        query: Format "PRINCIPAL RATE YEARS" (e.g., "10000 0.07 10")

    Returns:
        Future value, growth, and return calculations
    """
    try:
        parts = query.strip().split()
        if len(parts) < 3:
            raise ValueError("Query must contain principal, annual_rate, and years")

        principal = float(parts[0])
        annual_rate = float(parts[1])
        years = float(parts[2])

        if years <= 0 or principal <= 0:
            raise ValueError("Principal and years must be positive")

        future_value = principal * (1 + annual_rate) ** years
        total_growth = future_value - principal
        total_return_pct = (future_value / principal - 1) * 100

        formatted_summary = f"Investment: ${principal:,.2f} at {annual_rate*100:.2f}% for {years} years → Future Value: ${future_value:,.2f} (Total Return: {total_return_pct:.2f}%)"

        return CompoundGrowthResult(
            principal=principal,
            annual_rate=annual_rate,
            years=years,
            future_value=round(future_value, 2),
            total_growth=round(total_growth, 2),
            total_return_percent=round(total_return_pct, 2),
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return CompoundGrowthResult(
            principal=0.0,
            annual_rate=0.0,
            years=0.0,
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
        query: Format "NUMERATOR DENOMINATOR TYPE" (e.g., "82.50 5.50 pe")

    Returns:
        Ratio value, interpretation, and context
    """
    try:
        parts = query.strip().split()
        if len(parts) < 2:
            raise ValueError("Query must contain at least numerator and denominator")

        numerator = float(parts[0])
        denominator = float(parts[1])
        ratio_type = parts[2].lower() if len(parts) > 2 else "generic"

        if denominator == 0:
            raise ValueError("Denominator cannot be zero")

        ratio_value = numerator / denominator

        # Ratio interpretations
        ratio_info = {
            'pe': ("Price-to-Earnings Ratio", "High" if ratio_value > 25 else "Moderate" if ratio_value > 15 else "Low"),
            'debt_to_equity': ("Debt-to-Equity Ratio", "High leverage" if ratio_value > 1 else "Conservative"),
            'current': ("Current Ratio", "Good liquidity" if ratio_value > 1.5 else "Potential concern"),
            'roe': ("Return on Equity", "Strong" if ratio_value > 0.15 else "Average" if ratio_value > 0.10 else "Weak"),
            'generic': ("Financial Ratio", "Custom calculation")
        }

        description, context = ratio_info.get(ratio_type, ratio_info['generic'])
        interpretation = f"{description}: {ratio_value:.2f}"
        formatted_summary = f"{description}: {ratio_value:.2f} - {context}"

        return FinancialRatioResult(
            numerator=numerator,
            denominator=denominator,
            ratio_type=ratio_type,
            ratio_value=round(ratio_value, 4),
            description=description,
            interpretation=interpretation,
            context=context,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return FinancialRatioResult(
            numerator=0.0,
            denominator=0.0,
            ratio_type="error",
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
    FINANCIAL_TOOLS = [
        get_stock_price,
        get_company_info,
        get_financial_history,
        calculate_compound_growth,
        calculate_financial_ratio,
        tavily_search
    ]
except Exception as e:
    print(f"Warning: Tavily search not available: {e}")
    FINANCIAL_TOOLS = [
        get_stock_price,
        get_company_info,
        get_financial_history,
        calculate_compound_growth,
        calculate_financial_ratio
    ]

print(f"Loaded {len(FINANCIAL_TOOLS)} financial tools")

