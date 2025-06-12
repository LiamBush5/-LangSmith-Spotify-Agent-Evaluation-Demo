# LangSmith Financial Agent Evaluation Demo

## Overview

This is a production-ready LangSmith evaluation framework for a sophisticated financial research agent. The system demonstrates comprehensive evaluation capabilities using both the LangSmith SDK and UI.

### Key Features

- **Advanced Multi-Tool Agent** with ReAct reasoning pattern
- **5 Custom LLM-as-Judge Evaluators** including financial accuracy, logical reasoning, completeness, hallucination detection, and trajectory analysis
- **Realistic Financial Scenarios** covering stock analysis, CAGR calculations, portfolio modeling, and market research
- **Trajectory Analysis** to evaluate agent tool usage patterns
- **Comprehensive Reporting** with insights and performance breakdowns
- **Production-Ready** with CI integration capabilities

## Architecture

```
Financial Agent
├── Tools
│   ├── Web Search (Tavily)
│   ├── Financial Data API (yfinance)
│   └── Financial Calculator
├── Reasoning
│   └── ReAct Pattern with GPT-4
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
# Clone or download the demo files
cd financial-agent-demo

# Install dependencies
pip install -r requirements.txt

# Setup environment variables (copy from env_example.txt)
export LANGSMITH_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
export TAVILY_API_KEY="your_key_here"  # Optional
```

### 2. Run the Demo

```bash
# Quick demonstration
python demo_single_example.py

# Run the comprehensive evaluation
python run_evaluation.py
```

### 3. View Results

The script will output:
- **LangSmith Experiment URL** - Share this with your interviewer
- **Performance Summary** with metrics breakdown
- **Detailed Report** saved as markdown file

## What Gets Evaluated

### Test Scenarios (10 Examples)

1. **Stock Analysis** - Current prices, P/E ratios, market cap
2. **Growth Analysis** - Revenue CAGR calculations with factors
3. **Investment Projections** - Compound growth modeling
4. **Comparative Analysis** - Multi-company revenue comparisons
5. **Financial Ratios** - Debt-to-equity, industry benchmarks
6. **Market Analysis** - Stock volatility and driving factors
7. **Portfolio Analysis** - Mixed asset allocation returns
8. **Performance Analysis** - Year-over-year financials
9. **Risk Analysis** - Sharpe ratios and risk-adjusted returns
10. **Market Research** - Industry trends and impact analysis

### Evaluation Metrics

#### Financial Accuracy
- Numerical facts and calculations correctness
- Financial methodology soundness
- Data consistency validation

#### Logical Reasoning
- Coherence of reasoning steps
- Logical flow from question to answer
- Appropriate assumptions and conclusions

#### Completeness
- All question aspects addressed
- Sufficient detail and context provided
- No missing critical information

#### Hallucination Detection
- No unsupported financial claims
- Consistency with reasoning process
- Realistic market data validation

#### Trajectory Analysis
- Appropriate tool selection and sequence
- Efficiency of tool usage
- Expected vs actual tool patterns

## Technical Highlights

### 1. Technical Sophistication
- **ReAct Agent** with complex tool orchestration
- **LLM-as-Judge** evaluation using GPT-4
- **Trajectory Analysis** beyond simple accuracy metrics
- **Real-time Data** integration with financial APIs

### 2. Production Readiness
- **Deterministic Evaluation** (temperature=0)
- **Error Handling** and fallback parsing
- **Scalable Architecture** with concurrency control
- **CI Integration** ready with pytest compatibility

### 3. Business Value
- **Financial Domain** expertise and realistic scenarios
- **Multi-criteria Evaluation** framework
- **Regression Testing** capabilities
- **Continuous Monitoring** foundation

## Sample Output

```
FINANCIAL AGENT EVALUATION REPORT
================================================================

EXPERIMENT DETAILS:
  • Experiment: finance-agent-eval-20241215-143022
  • Dataset: Financial-Agent-Evaluation-Dataset (10 examples)
  • Evaluators: 5 custom LLM-as-judge evaluators
  • Model: gpt-4

KEY INSIGHTS:
  • Average tools used per query: 2.3
  • Overall agent performance: 87.4%
  • Best performing category: investment_projection (0.95)
  • Most challenging category: comparative_analysis (0.78)
  • Tool usage efficiency: 91.2%

EVALUATION CRITERIA:
  • Financial Accuracy: Numerical facts and calculations
  • Logical Reasoning: Coherence and soundness of analysis
  • Completeness: All aspects of questions addressed
  • Hallucination Check: No unsupported claims
  • Trajectory Quality: Appropriate tool usage patterns
```

## System Capabilities

### Core Functionality
1. **Advanced LangSmith Integration** - Custom evaluators, trajectory analysis, advanced metrics
2. **Sophisticated Agent Architecture** - Multi-tool orchestration, ReAct reasoning, error handling
3. **Financial Domain Implementation** - Realistic scenarios, proper calculations, industry knowledge
4. **Production-Grade Design** - Regression testing, monitoring, scalability considerations
5. **Innovative Evaluation Methods** - Beyond accuracy to reasoning quality and tool efficiency

### Technical Implementation
- **LLM-as-Judge Implementation** with robust fallback parsing
- **Trajectory Evaluation Algorithm** using longest common subsequence
- **Multi-criteria Scoring** with weighted combinations
- **Comprehensive Error Handling** for production reliability

## File Structure

```
financial-agent-demo/
├── requirements.txt          # Dependencies
├── config.py                # Configuration and API keys
├── financial_tools.py       # Advanced financial tools
├── financial_agent.py       # ReAct agent implementation
├── evaluation_dataset.py    # Test scenarios and data
├── custom_evaluations.py    # LLM-as-judge evaluators
├── run_evaluation.py        # Main demo script
├── demo_single_example.py   # Quick live demonstration
└── README.md               # This file
```

## Usage Guide

### Running the System

1. **Setup** - Configure environment variables and install dependencies

2. **Architecture Overview** - Review the agent tools and evaluator types in LangSmith UI

3. **Live Execution** - Run examples end-to-end, observe trace and evaluator outputs

4. **Results Analysis** - Examine experiment results, trajectory analysis, performance breakdown

5. **Production Features** - Explore CI integration, regression testing, monitoring capabilities

### Key System Features
- **Real-world Financial Scenarios** - Comprehensive test coverage
- **Multi-dimensional Evaluation** - Beyond accuracy to reasoning quality
- **Trajectory Analysis** - Tool usage optimization insights
- **LLM-as-Judge Reliability** - Robust evaluation with fallback parsing
- **Production Readiness** - Error handling, deterministic evaluation, scaling

## Troubleshooting

### Common Issues
1. **API Key Errors** - Ensure all keys are set in environment
2. **Model Access** - GPT-4 access required for best results
3. **Rate Limits** - Adjust `MAX_CONCURRENCY` in config.py
4. **Import Errors** - Run `pip install -r requirements.txt`

### Performance Optimization
- Use `MAX_CONCURRENCY=2` for rate-limited accounts
- Consider `gpt-3.5-turbo` for cost optimization (with some quality trade-off)
- Cache financial data for repeated evaluations

## Success Metrics

### What Great Results Look Like
- **Financial Accuracy**: >90% for calculation-based queries
- **Logical Reasoning**: >85% for complex analysis questions
- **Completeness**: >80% addressing all question aspects
- **No Hallucinations**: >95% for factual claims
- **Trajectory Quality**: >85% for appropriate tool usage

### Red Flags to Address
- Low trajectory scores (wrong tool selection)
- High hallucination rates (model reliability issues)
- Poor reasoning scores (prompt engineering needed)

## LangSmith Integration

### SDK Usage
The framework demonstrates comprehensive LangSmith SDK usage:
- Dataset creation and management
- Experiment execution with custom evaluators
- Trace analysis and performance monitoring
- Results aggregation and reporting

### UI Demonstration
Access your LangSmith UI to show:
- Dataset visualization with 10 financial scenarios
- Experiment results with evaluation scores
- Individual trace analysis showing agent reasoning
- Comparative analysis across multiple runs

## Project Summary

This comprehensive evaluation framework provides:

- **Technical Excellence** - Advanced agent architecture with sophisticated evaluation
- **Domain Expertise** - Real financial scenarios with proper calculations
- **Production Quality** - Scalable, monitorable, CI-ready evaluation pipeline
- **Innovation** - Trajectory analysis and multi-criteria LLM-as-judge evaluation
- **Business Value** - Framework for ensuring financial agent reliability at scale

**To get started, run the evaluation and explore the LangSmith experiment results.**