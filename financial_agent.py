"""
Advanced Financial Research Agent with LangSmith Tracing
Uses ReAct pattern with multiple financial tools for comprehensive analysis.
"""
import os
from typing import Dict, List, Any, Optional
import json
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langsmith import traceable, Client
from financial_tools import FINANCIAL_TOOLS
import config

# Initialize LangSmith client
client = Client()

# Configure LangSmith environment
os.environ["LANGSMITH_TRACING"] = config.LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT

class FinancialAgent:
    """
    Advanced financial research agent with comprehensive tool access and reasoning.
    """

    def __init__(self):
        """Initialize the financial agent with modern structured output tools."""
        self.tools = FINANCIAL_TOOLS
        self.tool_calls_log = []  # Track tool usage for evaluation

        # Initialize LLM with LangSmith tracing using factory function
        self.llm = config.get_chat_model()

        # Create agent with enhanced prompt
        self.agent = self._create_agent()

        # Create executor with verbose logging
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.AGENT_MAX_ITERATIONS,  # Configurable max iterations
            max_execution_time=config.AGENT_MAX_EXECUTION_TIME,  # Configurable timeout
            return_intermediate_steps=True
        )

    def _create_agent(self):
        """Create ReAct agent with financial expertise."""

        prompt_template = """You are a sophisticated financial research analyst with access to real-time financial data and advanced calculation tools. Your goal is to provide accurate, comprehensive, and insightful financial analysis.

AVAILABLE TOOLS:
{tools}

CRITICAL: TOOL INPUT FORMAT
Use the exact format for each tool:
- get_stock_price: Use MSFT (just the symbol, no quotes)
- get_company_info: Use AAPL (just the symbol, no quotes)
- get_financial_history: Use AAPL 5y (symbol and period separated by space)
- calculate_compound_growth: Use 10000 0.07 10 (principal, rate, years separated by spaces)
- calculate_financial_ratio: Use 82.50 5.50 pe (numerator, denominator, type separated by spaces)
- tavily_search_results_json: Use Microsoft stock price (search query, no quotes)

NEVER use JSON format like {{"symbol": "AAPL"}} - use space-separated values for multi-parameter tools!

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

ANALYSIS STANDARDS:
- Provide specific numerical data when available
- Calculate relevant metrics (CAGR, ratios, percentages)
- Include context about market conditions or company fundamentals
- Mention limitations or assumptions in your analysis
- Format financial figures clearly (e.g., $X.X billion, X.X%)

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (use proper parameter format as shown above)
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

    @traceable
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to analyze financial queries with full tracing.

        Args:
            query: Financial question to analyze

        Returns:
            Dict containing response, tool calls, and reasoning steps
        """
        print(f"\n🔍 Analyzing Query: {query}")
        print("="*60)

        # Reset tool calls log
        self.tool_calls_log = []

        try:
            # Execute the agent
            result = self.agent_executor.invoke({"input": query})

            # Extract information
            response = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Process intermediate steps to extract tool usage
            tool_trajectory = []
            reasoning_steps = []

            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    tool_name = action.tool if hasattr(action, 'tool') else "unknown"
                    tool_input = action.tool_input if hasattr(action, 'tool_input') else ""

                    tool_trajectory.append(tool_name)

                    # Serialize the observation for LangSmith compatibility
                    serialized_observation = self._serialize_tool_output(observation)

                    reasoning_steps.append({
                        "tool": tool_name,
                        "input": str(tool_input),
                        "output": str(serialized_observation)[:200] + "..." if len(str(serialized_observation)) > 200 else str(serialized_observation)
                    })

            # Compile comprehensive result - ensure all outputs are serializable
            analysis_result = {
                "response": response,
                "tool_trajectory": tool_trajectory,
                "reasoning_steps": reasoning_steps,
                "total_tool_calls": len(tool_trajectory),
                "unique_tools_used": list(set(tool_trajectory)),
                "query": query
            }

            print(f"\nAnalysis Complete!")
            print(f"Tools Used: {', '.join(analysis_result['unique_tools_used'])}")
            print(f"Total Tool Calls: {analysis_result['total_tool_calls']}")

            # Performance warning if approaching limits
            if analysis_result['total_tool_calls'] >= config.AGENT_MAX_ITERATIONS * 0.8:
                print(f"⚠️  Warning: High tool usage ({analysis_result['total_tool_calls']}/{config.AGENT_MAX_ITERATIONS}) - consider optimizing query")

            return analysis_result

        except Exception as e:
            error_result = {
                "response": f"Error during analysis: {str(e)}",
                "tool_trajectory": self.tool_calls_log,
                "reasoning_steps": [],
                "total_tool_calls": len(self.tool_calls_log),
                "unique_tools_used": [],
                "query": query,
                "error": True
            }
            print(f"Error: {str(e)}")
            return error_result

# Convenience function for evaluation
@traceable
def run_financial_agent(inputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Target function for LangSmith evaluation.
    Expected input format: {"question": "financial query"}
    Returns format: {"response": "answer", "tool_trajectory": [...]}
    """
    agent = FinancialAgent()  # Uses Financial structured tools by default
    query = inputs.get("question", "")

    if not query:
        return {
            "response": "Error: No question provided",
            "tool_trajectory": [],
            "reasoning_steps": [],
            "total_tool_calls": 0,
            "unique_tools_used": []
        }

    result = agent.analyze_query(query)

    # Ensure all outputs are serializable for LangSmith
    serialized_result = agent._serialize_tool_output(result)

    # Return in expected format for evaluation
    return {
        "response": serialized_result["response"],
        "tool_trajectory": serialized_result["tool_trajectory"],
        "reasoning_steps": serialized_result["reasoning_steps"],
        "total_tool_calls": serialized_result["total_tool_calls"],
        "unique_tools_used": serialized_result["unique_tools_used"],
        "query": serialized_result["query"]
    }

if __name__ == "__main__":
    """Test the financial agent with sample queries."""

    # Test queries
    test_queries = [
        "What is Apple's current stock price and how has it performed over the last year?",
        "Compare Tesla's revenue growth over the past 3 years and calculate the CAGR.",
        "If I invest $10,000 in the S&P 500 with a 7% annual return, what will it be worth in 10 years?"
    ]

    agent = FinancialAgent()

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)

        result = agent.analyze_query(query)

        print(f"\n Response:\n{result['response']}")
        print(f"\nTools Used: {result['unique_tools_used']}")
        print(f"Tool Trajectory: {' → '.join(result['tool_trajectory'])}")

        # Wait for user input to continue (comment out for automated testing)
        # input("\nPress Enter to continue to next test...")

    print(f"\nAll tests completed! Check LangSmith project '{config.LANGSMITH_PROJECT}' for detailed traces.")