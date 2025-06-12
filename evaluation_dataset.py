"""
Comprehensive Financial Evaluation Dataset for LangSmith
Contains realistic financial scenarios with expected answers and tool trajectories.
"""
from typing import List, Dict, Any
from langsmith import Client
import config

# Initialize LangSmith client
client = Client()

# Financial evaluation dataset with realistic scenarios
FINANCIAL_EVALUATION_DATASET = [
    {
        "input": "What is Tesla's current stock price and market cap? How does it compare to Apple's market cap?",
        "expected_response": "Based on current market data, Tesla's stock price is approximately $326-$330 per share with a market cap of around $1.0-1.1 trillion. Apple's market cap is approximately $3.0-3.1 trillion, making Apple roughly 3 times larger than Tesla by market capitalization. Tesla has seen significant growth but Apple remains one of the world's most valuable companies.",
        "expected_tools": ["financial_data_api", "tavily_search"],
        "category": "stock_analysis"
    },
    {
        "input": "Calculate Apple's revenue CAGR from 2019 to 2023 and explain the key growth factors.",
        "expected_response": "Apple's revenue CAGR from 2019 to 2023 was approximately 7-9%. Key growth factors included: strong iPhone sales (especially iPhone 12 and 13 cycles), significant growth in Services revenue (App Store, iCloud, Apple Music), expansion of the Mac and iPad product lines during the pandemic, and growth in wearables like AirPods and Apple Watch. The Services segment became increasingly important as a high-margin revenue driver.",
        "expected_tools": ["financial_data_api", "financial_calculator", "tavily_search"],
        "category": "growth_analysis"
    },
    {
        "input": "Compare Microsoft and Google's revenue growth over the past 3 years. Which company has grown faster?",
        "expected_response": "From 2021-2024, both Microsoft and Google showed strong growth. Microsoft's revenue grew from approximately $168B to $245B (CAGR of ~13-15%), driven by Azure cloud services, Office 365, and enterprise solutions. Google's revenue grew from approximately $183B to $280B+ (CAGR of ~15-17%), primarily driven by search advertising, YouTube, and Google Cloud. Google has shown slightly faster revenue growth, though both companies have benefited from digital transformation trends.",
        "expected_tools": ["financial_data_api", "financial_calculator", "tavily_search"],
        "category": "comparative_analysis"
    },
    {
        "input": "If I invest $10,000 in a portfolio with 60% stocks (expected 8% annual return) and 40% bonds (expected 4% annual return), what will it be worth in 10 years?",
        "expected_response": "With a $10,000 investment in a 60/40 portfolio: Portfolio expected return = (0.6 × 8%) + (0.4 × 4%) = 6.4% annually. Using compound interest: $10,000 × (1.064)^10 = approximately $18,771. This represents a total return of 87.7% over 10 years. This assumes annual rebalancing and consistent returns, though actual results will vary due to market volatility.",
        "expected_tools": ["financial_calculator"],
        "category": "investment_projections"
    },
    {
        "input": "What is Amazon's current debt-to-equity ratio and what does it indicate about the company's financial leverage?",
        "expected_response": "Amazon's current debt-to-equity ratio is approximately 0.17-0.18 (as of Q4 2024), which is relatively low and indicates conservative financial leverage. This means Amazon has about $0.17-0.18 of debt for every dollar of equity. This low ratio suggests: strong balance sheet health, low financial risk, significant borrowing capacity for growth investments, and efficient capital structure management. Amazon has improved this ratio significantly from higher levels in previous years.",
        "expected_tools": ["financial_data_api", "tavily_search"],
        "category": "financial_health"
    },
    {
        "input": "Research the latest trends in renewable energy stocks and identify potential investment opportunities.",
        "expected_response": "Current renewable energy trends include: strong growth in solar and wind installations, increasing government support through IRA and global climate policies, declining technology costs making renewables more competitive, and growing corporate renewable energy adoption. Key opportunities may include established players like NextEra Energy, solar companies benefiting from domestic manufacturing incentives, energy storage companies, and utilities with significant renewable portfolios. However, investors should consider policy risks, supply chain challenges, and valuation levels.",
        "expected_tools": ["tavily_search", "financial_data_api"],
        "category": "market_research"
    },
    {
        "input": "Calculate the P/E ratio for a company with earnings per share of $5.50 and a stock price of $82.50. Is this considered expensive or cheap?",
        "expected_response": "P/E ratio = Stock Price ÷ Earnings Per Share = $82.50 ÷ $5.50 = 15.0. A P/E ratio of 15 is generally considered reasonable to moderately valued. For context: the S&P 500 average P/E is typically 15-20, growth stocks often trade at 20-30+ P/E, and value stocks typically trade below 15 P/E. Whether this is expensive or cheap depends on the company's growth prospects, industry averages, and market conditions. A 15 P/E suggests the market expects moderate growth.",
        "expected_tools": ["financial_calculator"],
        "category": "valuation_analysis"
    },
    {
        "input": "How did Meta's financial performance change from 2022 to 2023? What were the key drivers?",
        "expected_response": "Meta showed significant improvement from 2022 to 2023. Revenue grew from $116.6B (2022) to $134.9B (2023), a 16% increase. More dramatically, net income surged from $23.2B to $39.1B, a 68% increase. Key drivers included: recovery in digital advertising spending, improved operational efficiency through cost-cutting measures, strong user growth across Facebook and Instagram, and better monetization of Reels. The 'Year of Efficiency' initiative significantly improved profit margins despite continued Reality Labs investments.",
        "expected_tools": ["financial_data_api", "tavily_search"],
        "category": "performance_analysis"
    },
    {
        "input": "What is Berkshire Hathaway's current portfolio allocation? What are Warren Buffett's top holdings?",
        "expected_response": "As of Q4 2024, Berkshire Hathaway's top holdings are: 1) Apple (~28% of portfolio, $69.9B value) - though reduced from previous quarters, 2) American Express (~14%, $41.1B), 3) Bank of America (~10%, reduced stake), 4) Coca-Cola (~$28.7B), and 5) Chevron (~$17.5B). The portfolio remains concentrated with ~70% in the top 5 holdings. Berkshire also holds a record $325+ billion in cash, reflecting Buffett's cautious approach to current market valuations. Recent moves include trimming Apple and bank holdings while building cash reserves.",
        "expected_tools": ["financial_data_api", "tavily_search"],
        "category": "portfolio_analysis"
    },
    {
        "input": "What is Nvidia's current stock price and market cap? How has it performed in the AI boom?",
        "expected_response": "Nvidia's current stock price is approximately $142-143 per share with a market cap of around $3.48-3.49 trillion, making it the 2nd most valuable company globally (behind Microsoft). Nvidia has been a major beneficiary of the AI boom, with its stock gaining over 170% in 2024 alone. The company's data center revenue has exploded due to demand for AI training chips (H100, A100), with quarterly data center revenue reaching $30+ billion. Nvidia's dominance in AI infrastructure has driven this exceptional performance.",
        "expected_tools": ["financial_data_api", "tavily_search"],
        "category": "stock_analysis"
    }
]

# Expected tool trajectories for trajectory evaluation
EXPECTED_TRAJECTORIES = {
    "stock_analysis": ["financial_data_api", "tavily_search_results_json"],
    "growth_analysis": ["financial_data_api", "financial_calculator", "tavily_search_results_json"],
    "investment_projection": ["financial_calculator"],
    "comparative_analysis": ["financial_data_api", "financial_calculator"],
    "financial_ratio_analysis": ["financial_data_api", "financial_calculator", "tavily_search_results_json"],
    "market_analysis": ["financial_data_api", "tavily_search_results_json"],
    "portfolio_analysis": ["financial_calculator"],
    "financial_performance": ["financial_data_api", "financial_calculator"],
    "risk_analysis": ["financial_calculator"],
    "market_research": ["tavily_search_results_json", "financial_data_api"]
}

def create_langsmith_dataset(dataset_name: str = "Financial-Agent-Evaluation-Dataset") -> str:
    """
    Create or update the evaluation dataset in LangSmith.

    Args:
        dataset_name: Name for the dataset in LangSmith

    Returns:
        Dataset ID
    """
    print(f"🗂️ Creating LangSmith dataset: {dataset_name}")

    # Prepare examples for LangSmith
    langsmith_examples = []

    for example in FINANCIAL_EVALUATION_DATASET:
        # Add expected trajectory to outputs
        category = example["category"]
        expected_trajectory = EXPECTED_TRAJECTORIES.get(category, [])

        langsmith_example = {
            "inputs": {"question": example["input"]},
            "outputs": {
                "response": example["expected_response"],
                "expected_trajectory": expected_trajectory,
                "category": category,
                "complexity": "medium"
            }
        }
        langsmith_examples.append(langsmith_example)

    try:
        # Check if dataset exists
        existing_datasets = client.list_datasets()
        dataset = None

        # Look for existing dataset
        for ds in existing_datasets:
            if ds.name == dataset_name:
                dataset = ds
                print(f"📝 Dataset '{dataset_name}' already exists. Updating...")
                break

        if dataset is None:
            # Create new dataset
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Comprehensive financial agent evaluation dataset with realistic scenarios covering stock analysis, calculations, comparisons, and market research."
            )
            print(f"✅ Created new dataset: {dataset_name}")

        # Add examples to dataset
        client.create_examples(
            dataset_id=dataset.id,
            examples=langsmith_examples
        )

        print(f"📊 Added {len(langsmith_examples)} examples to dataset")
        print(f"🏷️ Categories covered: {list(EXPECTED_TRAJECTORIES.keys())}")

        return dataset.id

    except Exception as e:
        print(f"❌ Error creating dataset: {str(e)}")
        raise

def analyze_dataset_coverage():
    """Analyze the dataset for coverage of different scenarios."""

    categories = {}
    complexity_levels = {}
    tool_requirements = {
        "calculation": 0,
        "comparison": 0,
        "real_time_data": 0
    }

    for example in FINANCIAL_EVALUATION_DATASET:
        # Count categories
        category = example["category"]
        categories[category] = categories.get(category, 0) + 1

        # Count complexity levels
        complexity = "medium"
        complexity_levels[complexity] = complexity_levels.get(complexity, 0) + 1

        # Count requirements
        expected_tools = example["expected_tools"]
        for tool in expected_tools:
            if tool in EXPECTED_TRAJECTORIES:
                tool_requirements["calculation"] += 1
                tool_requirements["comparison"] += 1
                tool_requirements["real_time_data"] += 1

    print("📊 Dataset Coverage Analysis:")
    print("="*50)
    print(f"Total Examples: {len(FINANCIAL_EVALUATION_DATASET)}")
    print(f"\n📁 Categories:")
    for category, count in categories.items():
        print(f"  - {category}: {count}")

    print(f"\n⚡ Complexity Levels:")
    for level, count in complexity_levels.items():
        print(f"  - {level}: {count}")

    print(f"\n🛠️ Tool Requirements:")
    for req, count in tool_requirements.items():
        print(f"  - {req}: {count}")

if __name__ == "__main__":
    # Analyze dataset coverage
    analyze_dataset_coverage()

    # Create dataset in LangSmith
    print("\n" + "="*60)
    dataset_id = create_langsmith_dataset()
    print(f"\n✅ Dataset created with ID: {dataset_id}")
    print(f"🔗 View in LangSmith UI under project: {config.LANGSMITH_PROJECT}")