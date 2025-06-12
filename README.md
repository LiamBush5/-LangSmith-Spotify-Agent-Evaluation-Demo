# LangSmith Financial Agent Evaluation Demo

## 🚀 What's New (2024-2025)

- **Modernized Tooling:** All tools now use structured output with Pydantic models. No manual JSON parsing, no legacy code.
- **Expanded Dataset:** 15 robust, realistic financial scenarios with complexity levels and better tool coverage.
- **Legacy Code Removed:** Only modern tools remain (`get_stock_price`, `get_company_info`, `calculate_compound_growth`, `calculate_financial_ratio`, `get_financial_history`, `tavily_search_results_json`).
- **Agent Architecture Simplified:** No more `use_modern_tools` parameter. Agent always uses the latest tools.
- **Robust Evaluation:** Automatic tool validation, improved error handling, and comprehensive trajectory analysis.
- **Best Practices:** Follows 2024-2025 LangChain and LangSmith recommendations for reliability and maintainability.

---

## Overview

This is a production-ready LangSmith evaluation framework for a sophisticated financial research agent. The system demonstrates comprehensive evaluation capabilities using both the LangSmith SDK and UI, with a focus on modern, robust, and maintainable code.

### Key Features

- **Modern Multi-Tool Agent** with ReAct reasoning pattern
- **Structured Output Everywhere:** All tools and evaluators use Pydantic models for type safety and reliability
- **5 Custom LLM-as-Judge Evaluators**: financial accuracy, logical reasoning, completeness, hallucination detection, and trajectory analysis
- **Expanded Financial Scenarios**: 15+ examples covering stock analysis, CAGR, portfolio modeling, market research, and more
- **Trajectory Analysis**: Evaluates agent tool usage patterns for efficiency and correctness
- **Automatic Tool Validation**: Ensures dataset and code stay in sync
- **Production-Ready**: CI integration, error handling, and scalability

## Architecture

```
Financial Agent
├── Tools (all structured output)
│   ├── get_stock_price (yfinance)
│   ├── get_company_info (yfinance)
│   ├── calculate_compound_growth (math)
│   ├── calculate_financial_ratio (math)
│   ├── get_financial_history (yfinance)
│   └── tavily_search_results_json (Tavily API)
├── Reasoning
│   └── ReAct Pattern with Gemini/GPT-4
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
python run_evaluation.py --max-examples 3
```

### 3. View Results

- **LangSmith Experiment URL** (printed in terminal)
- **Performance Summary** with metrics breakdown
- **Detailed Trace** in LangSmith UI

## What Gets Evaluated

### Test Scenarios (15 Examples)

- **Stock Analysis** - Current prices, P/E ratios, market cap
- **Growth Analysis** - Revenue CAGR calculations with factors
- **Investment Projections** - Compound growth modeling
- **Comparative Analysis** - Multi-company revenue comparisons
- **Financial Ratios** - Debt-to-equity, industry benchmarks
- **Market Analysis** - Stock volatility and driving factors
- **Portfolio Analysis** - Mixed asset allocation returns
- **Performance Analysis** - Year-over-year financials
- **Valuation Analysis** - P/E, market multiples
- **Market Research** - Industry trends and impact analysis
- **Edge Cases** - Invalid tickers, zero division, etc.

### Evaluation Metrics

- **Financial Accuracy**: Numerical facts and calculations correctness
- **Logical Reasoning**: Coherence of reasoning steps
- **Completeness**: All question aspects addressed
- **Hallucination Detection**: No unsupported financial claims
- **Trajectory Analysis**: Appropriate tool selection and sequence

## Data Sources for Each Tool

- **get_stock_price, get_company_info, get_financial_history**: [Yahoo Finance API](https://finance.yahoo.com/) via `yfinance` Python package
- **calculate_compound_growth, calculate_financial_ratio**: Local mathematical computation (no external API)
- **tavily_search_results_json**: [Tavily Search API](https://www.tavily.com/) for real-time web search and news

## Modern Tool List

- `get_stock_price(symbol: str)` → Stock price, market cap, P/E, 52-week range
- `get_company_info(symbol: str)` → Company name, sector, industry, employees, business summary
- `calculate_compound_growth(principal: float, annual_rate: float, years: float)` → Future value, total growth, return percent
- `calculate_financial_ratio(numerator: float, denominator: float, ratio_type: str)` → Ratio value, interpretation, context
- `get_financial_history(symbol: str, period: str)` → Historical price, CAGR, volatility, max drawdown
- `tavily_search_results_json(query: str)` → Real-time news, trends, and market analysis

## Best Practices & Lessons Learned

- **Always use structured output** (`@tool` + Pydantic models) for reliability
- **Remove legacy code** as soon as modern tools are available
- **Validate tool names** in both code and dataset to prevent evaluation drift
- **Test edge cases** (invalid tickers, zero division, etc.)
- **Keep evaluation scenarios up to date** with market trends and new tool capabilities
- **Use robust error handling** in all tools and evaluators

## How to Extend the System

- **Add a New Tool:**
  1. Write a new function with the `@tool` decorator and Pydantic output (if structured)
  2. Add it to `FINANCIAL_TOOLS` in `financial_tools.py`
  3. Add new scenarios to `evaluation_dataset.py` that use the new tool
  4. Update `AVAILABLE_MODERN_TOOLS` and validation logic if needed

- **Add a New Scenario:**
  1. Add a new example to `FINANCIAL_EVALUATION_DATASET` in `evaluation_dataset.py`
  2. Specify the expected tools and category
  3. Optionally add new complexity levels or edge cases

- **Add a New Evaluator:**
  1. Implement a new evaluator in `custom_evaluations.py` using structured output
  2. Add it to the `FINANCIAL_EVALUATORS` list
  3. It will be automatically included in the evaluation run

## Sample Output

```
FINANCIAL AGENT EVALUATION REPORT
================================================================

EXPERIMENT DETAILS:
  • Experiment: finance-agent-eval-20250612-140339
  • Dataset: Financial-Agent-Evaluation-Dataset (15 examples)
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
├── financial_tools.py       # Modern financial tools (all structured output)
├── financial_agent.py       # ReAct agent implementation (modern tools only)
├── evaluation_dataset.py    # Test scenarios and data (modernized)
├── custom_evaluations.py    # LLM-as-judge evaluators (structured output)
├── run_evaluation.py        # Main evaluation script
└── README.md                # This file
```

## Usage Guide

1. **Setup** - Configure environment variables and install dependencies
2. **Architecture Overview** - Review the agent tools and evaluator types in LangSmith UI
3. **Live Execution** - Run examples end-to-end, observe trace and evaluator outputs
4. **Results Analysis** - Examine experiment results, trajectory analysis, performance breakdown
5. **Production Features** - Explore CI integration, regression testing, monitoring capabilities

## Changelog

- **2024-06:**
  - All tools now use structured output (Pydantic models)
  - Legacy code and manual JSON parsing removed
  - Evaluation dataset expanded to 15 examples, with complexity levels
  - Modern agent architecture (no legacy tool fallback)
  - Tool validation and robust error handling added
  - README fully updated for modern best practices

---

For questions or contributions, please open an issue or pull request!
