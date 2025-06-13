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


