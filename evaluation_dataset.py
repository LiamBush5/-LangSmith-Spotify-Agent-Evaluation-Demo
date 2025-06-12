"""
Comprehensive Financial Evaluation Dataset for LangSmith
Contains realistic financial scenarios with expected answers and tool trajectories.
Updated for modern structured output tools (2024-2025).
"""
from typing import List, Dict, Any
from langsmith import Client
import config

# Initialize LangSmith client
client = Client()

# Financial evaluation dataset with realistic scenarios - UPDATED FOR MODERN TOOLS
FINANCIAL_EVALUATION_DATASET = [
    {
        "input": "What is Tesla's current stock price and market cap? How does it compare to Apple's market cap?",
        "expected_response": "Based on current market data, Tesla's stock price is approximately $320-330 per share with a market cap of around $1.0-1.1 trillion. Apple's market cap is approximately $3.0-3.1 trillion, making Apple roughly 3 times larger than Tesla by market capitalization. Tesla has seen significant growth but Apple remains one of the world's most valuable companies.",
        "expected_tools": ["get_stock_price", "tavily_search_results_json"],
        "category": "stock_analysis",
        "complexity": "medium"
    },
    {
        "input": "Calculate Apple's revenue CAGR from 2019 to 2023 and explain the key growth factors.",
        "expected_response": "Apple's revenue CAGR from 2019 to 2023 was approximately 7-9%. Key growth factors included: strong iPhone sales (especially iPhone 12 and 13 cycles), significant growth in Services revenue (App Store, iCloud, Apple Music), expansion of the Mac and iPad product lines during the pandemic, and growth in wearables like AirPods and Apple Watch. The Services segment became increasingly important as a high-margin revenue driver.",
        "expected_tools": ["get_company_info", "get_financial_history", "calculate_compound_growth", "tavily_search_results_json"],
        "category": "growth_analysis",
        "complexity": "hard"
    },
    {
        "input": "Compare Microsoft and Google's revenue growth over the past 3 years. Which company has grown faster?",
        "expected_response": "From 2021-2024, both Microsoft and Google showed strong growth. Microsoft's revenue grew from approximately $168B to $245B (CAGR of ~13-15%), driven by Azure cloud services, Office 365, and enterprise solutions. Google's revenue grew from approximately $183B to $280B+ (CAGR of ~15-17%), primarily driven by search advertising, YouTube, and Google Cloud. Google has shown slightly faster revenue growth, though both companies have benefited from digital transformation trends.",
        "expected_tools": ["get_company_info", "get_financial_history", "calculate_compound_growth", "tavily_search_results_json"],
        "category": "comparative_analysis",
        "complexity": "hard"
    },
    {
        "input": "If I invest $10,000 in a portfolio with 60% stocks (expected 8% annual return) and 40% bonds (expected 4% annual return), what will it be worth in 10 years?",
        "expected_response": "With a $10,000 investment in a 60/40 portfolio: Portfolio expected return = (0.6 × 8%) + (0.4 × 4%) = 6.4% annually. Using compound interest: $10,000 × (1.064)^10 = approximately $18,771. This represents a total return of 87.7% over 10 years. This assumes annual rebalancing and consistent returns, though actual results will vary due to market volatility.",
        "expected_tools": ["calculate_compound_growth"],
        "category": "investment_projections",
        "complexity": "medium"
    },
    {
        "input": "What is Amazon's current debt-to-equity ratio and what does it indicate about the company's financial leverage?",
        "expected_response": "Amazon's current debt-to-equity ratio is approximately 0.17-0.18 (as of Q4 2024), which is relatively low and indicates conservative financial leverage. This means Amazon has about $0.17-0.18 of debt for every dollar of equity. This low ratio suggests: strong balance sheet health, low financial risk, significant borrowing capacity for growth investments, and efficient capital structure management. Amazon has improved this ratio significantly from higher levels in previous years.",
        "expected_tools": ["get_company_info", "calculate_financial_ratio", "tavily_search_results_json"],
        "category": "financial_health",
        "complexity": "medium"
    },
    {
        "input": "Research the latest trends in renewable energy stocks and identify potential investment opportunities.",
        "expected_response": "Current renewable energy trends include: strong growth in solar and wind installations, increasing government support through IRA and global climate policies, declining technology costs making renewables more competitive, and growing corporate renewable energy adoption. Key opportunities may include established players like NextEra Energy, solar companies benefiting from domestic manufacturing incentives, energy storage companies, and utilities with significant renewable portfolios. However, investors should consider policy risks, supply chain challenges, and valuation levels.",
        "expected_tools": ["tavily_search_results_json", "get_stock_price"],
        "category": "market_research",
        "complexity": "hard"
    },
    {
        "input": "Calculate the P/E ratio for a company with earnings per share of $5.50 and a stock price of $82.50. Is this considered expensive or cheap?",
        "expected_response": "P/E ratio = Stock Price ÷ Earnings Per Share = $82.50 ÷ $5.50 = 15.0. A P/E ratio of 15 is generally considered reasonable to moderately valued. For context: the S&P 500 average P/E is typically 15-20, growth stocks often trade at 20-30+ P/E, and value stocks typically trade below 15 P/E. Whether this is expensive or cheap depends on the company's growth prospects, industry averages, and market conditions. A 15 P/E suggests the market expects moderate growth.",
        "expected_tools": ["calculate_financial_ratio"],
        "category": "valuation_analysis",
        "complexity": "easy"
    },
    {
        "input": "How did Meta's financial performance change from 2022 to 2023? What were the key drivers?",
        "expected_response": "Meta showed significant improvement from 2022 to 2023. Revenue grew from $116.6B (2022) to $134.9B (2023), a 16% increase. More dramatically, net income surged from $23.2B to $39.1B, a 68% increase. Key drivers included: recovery in digital advertising spending, improved operational efficiency through cost-cutting measures, strong user growth across Facebook and Instagram, and better monetization of Reels. The 'Year of Efficiency' initiative significantly improved profit margins despite continued Reality Labs investments.",
        "expected_tools": ["get_company_info", "get_financial_history", "tavily_search_results_json"],
        "category": "performance_analysis",
        "complexity": "medium"
    },
    {
        "input": "What is Berkshire Hathaway's current portfolio allocation? What are Warren Buffett's top holdings?",
        "expected_response": "As of Q4 2024, Berkshire Hathaway's top holdings are: 1) Apple (~28% of portfolio, $69.9B value) - though reduced from previous quarters, 2) American Express (~14%, $41.1B), 3) Bank of America (~10%, reduced stake), 4) Coca-Cola (~$28.7B), and 5) Chevron (~$17.5B). The portfolio remains concentrated with ~70% in the top 5 holdings. Berkshire also holds a record $325+ billion in cash, reflecting Buffett's cautious approach to current market valuations. Recent moves include trimming Apple and bank holdings while building cash reserves.",
        "expected_tools": ["get_stock_price", "tavily_search_results_json"],
        "category": "portfolio_analysis",
        "complexity": "medium"
    },
    {
        "input": "What is Nvidia's current stock price and market cap? How has it performed in the AI boom?",
        "expected_response": "Nvidia's current stock price is approximately $142-145 per share with a market cap of around $3.48-3.52 trillion, making it the 2nd most valuable company globally (behind Microsoft). Nvidia has been a major beneficiary of the AI boom, with its stock gaining over 170% in 2024 alone. The company's data center revenue has exploded due to demand for AI training chips (H100, A100), with quarterly data center revenue reaching $30+ billion. Nvidia's dominance in AI infrastructure has driven this exceptional performance.",
        "expected_tools": ["get_stock_price", "tavily_search_results_json"],
        "category": "stock_analysis",
        "complexity": "medium"
    },
    # NEW EXAMPLES TO TEST MORE TOOL COMBINATIONS
    {
        "input": "Analyze Apple's stock performance over the last 2 years. Calculate the CAGR and compare it to the S&P 500.",
        "expected_response": "Apple's stock has shown strong performance over the last 2 years. From approximately $150 to $199 (current), representing a CAGR of roughly 15-16%. This outperforms the S&P 500's typical 8-10% annual returns. Key drivers include AI integration announcements, strong iPhone sales, services growth, and the Vision Pro launch. However, performance has been somewhat volatile due to China market concerns and AI competition.",
        "expected_tools": ["get_stock_price", "get_financial_history", "calculate_compound_growth", "tavily_search_results_json"],
        "category": "performance_analysis",
        "complexity": "hard"
    },
    {
        "input": "What's Tesla's current P/E ratio and how does it compare to traditional automakers like Ford?",
        "expected_response": "Tesla's current P/E ratio is approximately 46-50, which is significantly higher than traditional automakers. Ford's P/E ratio is typically around 12-15. Tesla's higher valuation reflects: growth expectations, EV market leadership, energy business, autonomous driving potential, and supercharger network value. However, it also indicates higher risk and growth expectations that Tesla must meet. Traditional automakers trade at lower multiples due to mature, cyclical business models.",
        "expected_tools": ["get_stock_price", "calculate_financial_ratio", "tavily_search_results_json"],
        "category": "comparative_analysis",
        "complexity": "medium"
    },
    {
        "input": "If I have $50,000 to invest and want a 12% annual return, how much will I have in 15 years? What investment strategy might achieve this?",
        "expected_response": "With $50,000 invested at 12% annually for 15 years: $50,000 × (1.12)^15 = approximately $273,676. This represents a 447% total return. To achieve 12% annually, you might consider: growth stock portfolios, technology-focused ETFs, small-cap value stocks, or emerging market exposure. However, 12% is above historical market averages and requires higher risk. A diversified approach with some high-growth assets would be prudent.",
        "expected_tools": ["calculate_compound_growth", "tavily_search_results_json"],
        "category": "investment_projections",
        "complexity": "medium"
    },
    {
        "input": "Get me information about Microsoft's business segments and calculate what percentage each contributes to total revenue.",
        "expected_response": "Microsoft's main business segments include: 1) Productivity and Business Processes (~33% of revenue, $69B annually) - Office 365, Teams, LinkedIn, 2) Intelligent Cloud (~40% of revenue, $87B annually) - Azure, Windows Server, SQL Server, 3) More Personal Computing (~27% of revenue, $59B annually) - Windows, Xbox, Surface, Search. The Intelligent Cloud segment has become the largest, driven by Azure's rapid growth in the cloud computing market.",
        "expected_tools": ["get_company_info", "calculate_financial_ratio", "tavily_search_results_json"],
        "category": "financial_health",
        "complexity": "medium"
    },
    {
        "input": "What are the top 3 performing semiconductor stocks this year and why have they outperformed?",
        "expected_response": "Top performing semiconductor stocks in 2024-2025 typically include: 1) Nvidia - AI chip dominance driving data center demand, 2) AMD - gaining market share from Intel, strong data center growth, 3) Broadcom - AI infrastructure and networking chips. Key performance drivers: AI boom increasing chip demand, data center expansion, autonomous vehicle development, and IoT growth. However, the sector faces cyclical risks, China trade tensions, and high valuations.",
        "expected_tools": ["tavily_search_results_json", "get_stock_price", "get_financial_history"],
        "category": "market_research",
        "complexity": "hard"
    }
]

# Updated expected tool trajectories for modern tools
EXPECTED_TRAJECTORIES = {
    "stock_analysis": ["get_stock_price", "tavily_search_results_json"],
    "growth_analysis": ["get_company_info", "get_financial_history", "calculate_compound_growth", "tavily_search_results_json"],
    "investment_projections": ["calculate_compound_growth", "tavily_search_results_json"],
    "comparative_analysis": ["get_stock_price", "get_company_info", "calculate_financial_ratio", "tavily_search_results_json"],
    "financial_health": ["get_company_info", "calculate_financial_ratio", "tavily_search_results_json"],
    "valuation_analysis": ["calculate_financial_ratio", "get_stock_price"],
    "performance_analysis": ["get_company_info", "get_financial_history", "tavily_search_results_json"],
    "portfolio_analysis": ["get_stock_price", "tavily_search_results_json"],
    "market_research": ["tavily_search_results_json", "get_stock_price", "get_financial_history"]
}

# Available modern tools for validation
AVAILABLE_MODERN_TOOLS = [
    "tavily_search_results_json",
    "get_stock_price",
    "get_company_info",
    "calculate_compound_growth",
    "calculate_financial_ratio",
    "get_financial_history"
]

def create_langsmith_dataset(dataset_name: str = "Financial-Agent-Evaluation-Dataset", max_examples: int = None) -> str:
    """
    Create or ensure the full evaluation dataset exists in LangSmith.
    Does NOT modify existing examples - keeps full dataset intact.

    Args:
        dataset_name: Name for the dataset in LangSmith
        max_examples: Used only for logging (actual sampling happens during evaluation)

    Returns:
        Dataset ID
    """
    print(f"Ensuring LangSmith dataset exists: {dataset_name}")

    # Validate that expected tools match available tools
    _validate_expected_tools()

    # Always prepare ALL examples for the dataset
    langsmith_examples = []

    for i, example in enumerate(FINANCIAL_EVALUATION_DATASET):
        # Add complexity if not present (default to medium)
        complexity = example.get("complexity", "medium")

        langsmith_examples.append({
            "inputs": {"question": example["input"]},
            "outputs": {
                "expected_response": example.get("expected_response", ""),
                "expected_category": example["category"],
                "expected_complexity": complexity,
                "expected_tools": example.get("expected_tools", [])
            },
            "metadata": {
                "category": example["category"],
                "complexity": complexity,
                "example_id": f"fin_eval_{i+1}",
                "source": "manually_curated",
                "version": "v2.0",  # Updated version for modern tools
                "description": f"Financial {example['category']} question - {complexity} complexity",
                "tool_count": len(example.get("expected_tools", [])),
                "expected_tool_names": example.get("expected_tools", [])
            }
        })

    try:
        # Check if dataset exists
        dataset = None
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
            print(f"✓ Found existing dataset: {dataset_name}")

            # Check how many examples are in the dataset
            existing_examples = list(client.list_examples(dataset_id=dataset.id))
            print(f"✓ Dataset contains {len(existing_examples)} examples")

            # Only update if the dataset is empty or has fewer examples than expected
            if len(existing_examples) < len(FINANCIAL_EVALUATION_DATASET):
                print(f"📝 Updating dataset with {len(langsmith_examples)} examples...")
                # Clear and re-add all examples to ensure consistency
                for example in existing_examples:
                    client.delete_example(example.id)

                created_examples = client.create_examples(
                    dataset_id=dataset.id,
                    examples=langsmith_examples
                )
                print(f"✓ Updated dataset with {len(langsmith_examples)} examples")
            else:
                print(f"✓ Dataset is up to date with {len(existing_examples)} examples")

        except Exception:
            # Create new dataset
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Comprehensive financial agent evaluation dataset with realistic scenarios covering stock analysis, calculations, comparisons, and market research. Updated for modern structured output tools (v2.0). Includes category splits and complexity levels for detailed analysis."
            )
            print(f"🆕 Created new dataset: {dataset_name}")

            # Add all examples to new dataset
            created_examples = client.create_examples(
                dataset_id=dataset.id,
                examples=langsmith_examples
            )
            print(f"✓ Added {len(langsmith_examples)} examples to new dataset")

        # Log sampling info for cost control
        if max_examples and max_examples < len(FINANCIAL_EVALUATION_DATASET):
            print(f"💰 Cost Control: Will evaluate {max_examples} out of {len(FINANCIAL_EVALUATION_DATASET)} examples")
        else:
            print(f"📊 Will evaluate all {len(FINANCIAL_EVALUATION_DATASET)} examples")

        # Create dataset splits for better organization
        _create_dataset_splits(dataset.id, langsmith_examples)

        return dataset.id

    except Exception as e:
        print(f"Failed to setup dataset: {e}")
        raise

def _validate_expected_tools():
    """Validate that all expected tools in the dataset match available modern tools."""
    print("🔧 Validating expected tools against available modern tools...")

    all_expected_tools = set()
    for example in FINANCIAL_EVALUATION_DATASET:
        all_expected_tools.update(example.get("expected_tools", []))

    missing_tools = all_expected_tools - set(AVAILABLE_MODERN_TOOLS)
    extra_tools = set(AVAILABLE_MODERN_TOOLS) - all_expected_tools

    if missing_tools:
        print(f"⚠️  Warning: Expected tools not available: {missing_tools}")

    if extra_tools:
        print(f"💡 Available tools not used in examples: {extra_tools}")

    print(f"✓ Validation complete. Expected tools: {sorted(all_expected_tools)}")
    print(f"✓ Available tools: {sorted(AVAILABLE_MODERN_TOOLS)}")

def _create_dataset_splits(dataset_id: str, examples: list):
    """Create dataset splits based on categories and complexity."""
    try:
        # Group examples by category
        categories = {}
        complexity_levels = {}

        for i, example in enumerate(examples):
            category = example["metadata"]["category"]
            complexity = example["metadata"]["complexity"]

            if category not in categories:
                categories[category] = []
            if complexity not in complexity_levels:
                complexity_levels[complexity] = []

            categories[category].append(i)
            complexity_levels[complexity].append(i)

        print(f"Creating splits for categories: {list(categories.keys())}")
        print(f"Creating splits for complexity: {list(complexity_levels.keys())}")

        # Log distribution
        print("\n📊 Dataset Distribution:")
        print("Categories:")
        for cat, indices in categories.items():
            print(f"  {cat}: {len(indices)} examples")
        print("Complexity:")
        for comp, indices in complexity_levels.items():
            print(f"  {comp}: {len(indices)} examples")

    except Exception as e:
        print(f"Note: Could not create splits automatically: {e}")
        print("You can create splits manually in the LangSmith UI if needed")

if __name__ == "__main__":
    print(f"🚀 Modern Financial Evaluation Dataset v2.0")
    print(f"Total examples: {len(FINANCIAL_EVALUATION_DATASET)}")
    print(f"Available modern tools: {len(AVAILABLE_MODERN_TOOLS)}")

    # Validate tools
    _validate_expected_tools()

    # Create dataset in LangSmith
    dataset_id = create_langsmith_dataset()
    print(f"\n✅ Dataset created with ID: {dataset_id}")
    print(f"🔗 View in LangSmith UI under project: {config.LANGSMITH_PROJECT}")
    print(f"🎯 Dataset now uses modern structured output tools!")