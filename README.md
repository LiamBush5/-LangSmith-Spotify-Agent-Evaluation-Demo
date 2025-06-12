# LangSmith Financial Agent Evaluation Demo

## What's New (2024-2025)

- **Optimized Tool Architecture:** All multi-parameter tools now use single-parameter design optimized for ReAct agents
- **Structured Output:** All tools return Pydantic models for type safety and reliability
- **Expanded Dataset:** 10 robust, realistic financial scenarios with complexity levels and comprehensive tool coverage
- **Modern Agent Implementation:** ReAct agent with space-separated parameter parsing for optimal performance
- **Comprehensive Evaluation:** 5 custom LLM-as-judge evaluators with trajectory analysis and tool usage optimization
- **Production-Ready:** Follows 2024-2025 LangChain best practices for ReAct agent implementations

---

## Overview

This is a production-ready LangSmith evaluation framework for a sophisticated financial research agent. The system demonstrates comprehensive evaluation capabilities using both the LangSmith SDK and UI, with a focus on modern, robust, and maintainable code optimized for ReAct agent patterns.

### Key Features

- **Optimized ReAct Agent** with single-parameter tool design for maximum compatibility
- **Structured Output Everywhere:** All tools and evaluators use Pydantic models for type safety and reliability
- **5 Custom LLM-as-Judge Evaluators**: financial accuracy, logical reasoning, completeness, hallucination detection, and trajectory analysis
- **Comprehensive Financial Scenarios**: 10 examples covering stock analysis, CAGR, portfolio modeling, market research, and more
- **Trajectory Analysis**: Evaluates agent tool usage patterns for efficiency and correctness
- **Automatic Tool Validation**: Ensures dataset and code stay in sync
- **Production-Ready**: Error handling, cost control, and scalability

## Architecture

```
Financial Agent (ReAct Pattern)
├── Tools (single-parameter design)
│   ├── get_stock_price(symbol)
│   ├── get_company_info(symbol)
│   ├── calculate_compound_growth(query) → parses "principal rate years"
│   ├── calculate_financial_ratio(query) → parses "numerator denominator type"
│   ├── get_financial_history(query) → parses "symbol period"
│   └── tavily_search_results_json(query)
├── Reasoning
│   └── ReAct Pattern with space-separated parameter parsing
└── Evaluation
    ├── Financial Accuracy
    ├── Logical Reasoning
    ├── Completeness
    ├── Hallucination Detection
    └── Trajectory Quality
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the repo
cd LangChain

# Install dependencies
pip install -r requirements.txt

# Setup environment variables (copy from env_example.txt)
export LANGSMITH_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
export TAVILY_API_KEY="your_key_here"  # Optional
```

### 2. Run the Evaluation

```bash
# Run with cost control (default: 3 examples)
python run_evaluation.py

# Run specific number of examples
python run_evaluation.py --max-examples 5

# Run all examples
python run_evaluation.py --all
```

### 3. View Results

- **LangSmith Experiment URL** (printed in terminal)
- **Performance Summary** with metrics breakdown
- **Detailed Trace** in LangSmith UI

## What Gets Evaluated

### Test Scenarios (10 Examples)

- **Stock Analysis** - Current prices, P/E ratios, market cap analysis
- **Growth Calculations** - Revenue CAGR with compound growth modeling
- **Investment Projections** - Multi-year compound growth scenarios
- **Comparative Analysis** - Multi-company financial comparisons
- **Financial Ratios** - P/E ratios, debt-to-equity, industry benchmarks
- **Market Research** - Real-time news and trend analysis
- **Portfolio Analysis** - Mixed asset allocation and returns
- **Historical Analysis** - Multi-year performance tracking
- **Valuation Analysis** - Market multiples and pricing analysis
- **Edge Cases** - Invalid tickers, error handling validation

### Evaluation Metrics

- **Financial Accuracy**: Numerical facts and calculations correctness
- **Logical Reasoning**: Coherence of reasoning steps and tool selection
- **Completeness**: All question aspects addressed comprehensively
- **Hallucination Detection**: No unsupported financial claims or data
- **Trajectory Analysis**: Appropriate tool selection, sequence, and efficiency

## Data Sources and Tool Design

### Tool Input Format (ReAct Optimized)

All multi-parameter tools use space-separated input format optimized for ReAct agents:

- **Single Parameter Tools**: `get_stock_price("AAPL")`, `get_company_info("MSFT")`
- **Multi-Parameter Tools**:
  - `get_financial_history("AAPL 5y")` → parses symbol and period
  - `calculate_compound_growth("10000 0.07 10")` → parses principal, rate, years
  - `calculate_financial_ratio("82.50 5.50 pe")` → parses numerator, denominator, type

### Data Sources

- **get_stock_price, get_company_info, get_financial_history**: Yahoo Finance API via `yfinance` Python package
- **calculate_compound_growth, calculate_financial_ratio**: Local mathematical computation with robust parsing
- **tavily_search_results_json**: Tavily Search API for real-time web search and financial news

## Tool Specifications

- `get_stock_price(symbol: str)` → StockPriceData with price, market cap, P/E, 52-week range
- `get_company_info(symbol: str)` → CompanyInfo with name, sector, industry, business summary
- `calculate_compound_growth(query: str)` → CompoundGrowthResult with future value, growth, returns
- `calculate_financial_ratio(query: str)` → FinancialRatioResult with ratio value, interpretation, context
- `get_financial_history(query: str)` → FinancialHistoryResult with CAGR, volatility, performance metrics
- `tavily_search_results_json(query: str)` → Real-time news, trends, and market analysis

## ReAct Agent Design Principles

### Why Single-Parameter Tools

Based on LangChain best practices and community recommendations:

1. **ReAct Agent Compatibility**: ReAct agents work optimally with single-parameter tools
2. **Parsing Reliability**: Space-separated parsing is more reliable than JSON for ReAct patterns
3. **Framework Alignment**: Follows LangChain's recommended patterns for production systems
4. **Error Reduction**: Eliminates JSON parsing errors and parameter validation issues

### Agent Prompt Optimization

The agent uses optimized prompts that specify exact input formats:

```
CRITICAL: TOOL INPUT FORMAT
Use the exact format for each tool:
- get_stock_price: Use MSFT (just the symbol, no quotes)
- get_company_info: Use AAPL (just the symbol, no quotes)
- get_financial_history: Use AAPL 5y (symbol and period separated by space)
- calculate_compound_growth: Use 10000 0.07 10 (principal, annual_rate, years separated by spaces)
- calculate_financial_ratio: Use 82.50 5.50 pe (numerator, denominator, ratio_type separated by spaces)
- tavily_search_results_json: Use Microsoft stock price (search query, no quotes)
```

## Best Practices and Lessons Learned

### ReAct Agent Optimization

- **Use single-parameter tools** for maximum ReAct agent compatibility
- **Space-separated parsing** is more reliable than JSON for multi-parameter tools
- **Clear input format specifications** in agent prompts prevent parsing errors
- **Robust error handling** with fallback parsing for edge cases

### Evaluation Framework

- **Always use structured output** (Pydantic models) for reliability
- **Test edge cases** (invalid tickers, parsing errors, etc.)
- **Validate tool usage patterns** through trajectory analysis
- **Cost control** with configurable example limits

### Production Considerations

- **Error handling** for API failures and parsing errors
- **Rate limiting** for external API calls
- **Monitoring** through LangSmith tracing
- **Scalability** with parallel evaluation support

## How to Extend the System

### Add a New Single-Parameter Tool

```python
@tool
def new_tool(query: str) -> NewToolResult:
    """Tool description for the agent."""
    # Parse query if multi-parameter
    # Implement tool logic
    return NewToolResult(...)
```

### Add a New Multi-Parameter Tool

```python
@tool
def multi_param_tool(query: str) -> MultiParamResult:
    """Tool description specifying input format."""
    try:
        # Parse space-separated parameters
        parts = query.strip().split()
        if len(parts) >= 2:
            param1 = parts[0]
            param2 = parts[1]
        # Implement tool logic
        return MultiParamResult(...)
    except Exception as e:
        return MultiParamResult(error=str(e))
```

### Add New Evaluation Scenarios

1. Add examples to `FINANCIAL_EVALUATION_DATASET` in `evaluation_dataset.py`
2. Specify expected tools and complexity levels
3. Include edge cases and error scenarios

## Sample Output

```
FINANCIAL AGENT EVALUATION REPORT
================================================================

EXPERIMENT DETAILS:
  • Experiment: finance-agent-eval-20250103-140339
  • Dataset: Financial-Agent-Evaluation-Dataset (3 examples)
  • Evaluators: 5 custom LLM-as-judge evaluators
  • Model: gemini-2.0-flash

KEY INSIGHTS:
  • Average tools used per query: 2.0
  • Overall agent performance: 89.2%
  • Best performing category: investment_projections (0.95)
  • Most challenging category: comparative_analysis (0.78)
  • Tool usage efficiency: 93.1%

EVALUATION CRITERIA:
  • Financial Accuracy: Numerical facts and calculations
  • Logical Reasoning: Coherence and soundness of analysis
  • Completeness: All aspects of questions addressed
  • Hallucination Check: No unsupported claims
  • Trajectory Quality: Appropriate tool usage patterns
```

## File Structure

```
LangChain/
├── requirements.txt          # Dependencies
├── config.py                # Configuration and API keys
├── financial_tools.py       # ReAct-optimized financial tools
├── financial_agent.py       # ReAct agent with optimized prompts
├── evaluation_dataset.py    # Test scenarios and validation
├── custom_evaluations.py    # LLM-as-judge evaluators
├── run_evaluation.py        # Main evaluation script with cost control
└── README.md                # This file
```

## Usage Guide

1. **Setup** - Configure environment variables and install dependencies
2. **Architecture Review** - Understand ReAct agent design and tool optimization
3. **Live Execution** - Run examples with cost control, observe traces
4. **Results Analysis** - Examine experiment results and trajectory analysis
5. **Production Deployment** - Leverage error handling and monitoring features

## Technical Implementation Notes

### ReAct Agent Limitations and Solutions

- **Challenge**: ReAct agents have difficulty with multi-parameter tools using JSON format
- **Solution**: Implemented single-parameter tools with space-separated parsing
- **Result**: Improved reliability and reduced parsing errors

### Tool Design Evolution

- **Before**: Multi-parameter tools with JSON input caused validation errors
- **After**: Single-parameter tools with internal parsing optimized for ReAct patterns
- **Benefit**: 100% compatibility with ReAct agent architecture

### Performance Optimizations

- **Cost Control**: Configurable example limits for development and testing
- **Error Handling**: Robust parsing with fallback mechanisms
- **Monitoring**: Comprehensive LangSmith tracing for debugging

## Changelog

- **2024-12**:
  - Implemented single-parameter tool design for ReAct agent optimization
  - Added space-separated parameter parsing for multi-parameter tools
  - Updated agent prompts with explicit input format specifications
  - Enhanced error handling and fallback parsing mechanisms
  - Added cost control with configurable example limits
  - Comprehensive testing and validation of ReAct agent patterns

---

For questions or contributions, please open an issue or pull request.
