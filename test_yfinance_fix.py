#!/usr/bin/env python3
"""
Test script to verify yfinance HTTP 401 error fixes
This script tests the financial tools that were previously failing with HTTP 401 errors.
"""

import yfinance as yf
import time
from financial_tools import get_stock_price, get_company_info, get_financial_history

def test_basic_yfinance():
    """Test basic yfinance functionality"""
    print("=" * 60)
    print("TESTING BASIC YFINANCE FUNCTIONALITY")
    print("=" * 60)

    try:
        # Test 1: Basic stock data download
        print("\n1. Testing yf.download()...")
        data = yf.download("AAPL", period="5d", progress=False)
        if not data.empty:
            print("   ✅ yf.download() - SUCCESS")
            print(f"   Retrieved {len(data)} days of data for AAPL")
        else:
            print("   ❌ yf.download() - FAILED (empty data)")

    except Exception as e:
        print(f"   ❌ yf.download() - FAILED: {e}")

    try:
        # Test 2: Ticker info
        print("\n2. Testing Ticker.info...")
        ticker = yf.Ticker("MSFT")
        info = ticker.info
        if info and 'symbol' in info:
            print("   ✅ Ticker.info - SUCCESS")
            print(f"   Retrieved info for {info.get('symbol', 'Unknown')}")
        else:
            print("   ❌ Ticker.info - FAILED (no data)")

    except Exception as e:
        print(f"   ❌ Ticker.info - FAILED: {e}")

    try:
        # Test 3: Historical data
        print("\n3. Testing Ticker.history()...")
        ticker = yf.Ticker("TSLA")
        hist = ticker.history(period="1mo")
        if not hist.empty:
            print("   ✅ Ticker.history() - SUCCESS")
            print(f"   Retrieved {len(hist)} days of historical data for TSLA")
        else:
            print("   ❌ Ticker.history() - FAILED (empty data)")

    except Exception as e:
        print(f"   ❌ Ticker.history() - FAILED: {e}")

def test_financial_tools():
    """Test our custom financial tools"""
    print("\n" + "=" * 60)
    print("TESTING CUSTOM FINANCIAL TOOLS")
    print("=" * 60)

    # Test symbols that were previously failing
    test_symbols = ["AAPL", "MSFT", "TSLA", "GOOGL"]

    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")

        try:
            # Test get_stock_price
            print(f"  Testing get_stock_price({symbol})...")
            result = get_stock_price.invoke({"symbol": symbol})
            if result.error:
                print(f"    ❌ get_stock_price - ERROR: {result.error}")
            else:
                print(f"    ✅ get_stock_price - SUCCESS")
                print(f"    Current price: ${result.current_price}")

        except Exception as e:
            print(f"    ❌ get_stock_price - EXCEPTION: {e}")

        try:
            # Test get_company_info
            print(f"  Testing get_company_info({symbol})...")
            result = get_company_info.invoke({"symbol": symbol})
            if result.error:
                print(f"    ❌ get_company_info - ERROR: {result.error}")
            else:
                print(f"    ✅ get_company_info - SUCCESS")
                print(f"    Company: {result.name}")

        except Exception as e:
            print(f"    ❌ get_company_info - EXCEPTION: {e}")

        try:
            # Test get_financial_history
            print(f"  Testing get_financial_history({symbol})...")
            result = get_financial_history.invoke({"symbol": symbol, "period": "1y"})
            if result.error:
                print(f"    ❌ get_financial_history - ERROR: {result.error}")
            else:
                print(f"    ✅ get_financial_history - SUCCESS")
                print(f"    Total return: {result.total_return_percent}%")

        except Exception as e:
            print(f"    ❌ get_financial_history - EXCEPTION: {e}")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

def test_problematic_endpoints():
    """Test endpoints that were specifically mentioned as problematic"""
    print("\n" + "=" * 60)
    print("TESTING PREVIOUSLY PROBLEMATIC ENDPOINTS")
    print("=" * 60)

    try:
        print("\n1. Testing income statement...")
        ticker = yf.Ticker("AAPL")
        income_stmt = ticker.income_stmt
        if income_stmt is not None and not income_stmt.empty:
            print("   ✅ income_stmt - SUCCESS")
        else:
            print("   ❌ income_stmt - FAILED (no data)")

    except Exception as e:
        print(f"   ❌ income_stmt - FAILED: {e}")

    try:
        print("\n2. Testing insider roster holders...")
        ticker = yf.Ticker("AAPL")
        insider_roster = ticker.insider_roster_holders
        if insider_roster is not None and not insider_roster.empty:
            print("   ✅ insider_roster_holders - SUCCESS")
        else:
            print("   ❌ insider_roster_holders - FAILED (no data)")

    except Exception as e:
        print(f"   ❌ insider_roster_holders - FAILED: {e}")

    try:
        print("\n3. Testing upgrades/downgrades...")
        ticker = yf.Ticker("AAPL")
        upgrades_downgrades = ticker.upgrades_downgrades
        if upgrades_downgrades is not None and not upgrades_downgrades.empty:
            print("   ✅ upgrades_downgrades - SUCCESS")
        else:
            print("   ❌ upgrades_downgrades - FAILED (no data)")

    except Exception as e:
        print(f"   ❌ upgrades_downgrades - FAILED: {e}")

def main():
    """Run all tests"""
    print("🧪 YFINANCE HTTP 401 ERROR FIX VERIFICATION")
    print("=" * 60)
    print(f"yfinance version: {yf.__version__}")
    print("=" * 60)

    # Run all test suites
    test_basic_yfinance()
    test_financial_tools()
    test_problematic_endpoints()

    print("\n" + "=" * 60)
    print("✅ TEST SUITE COMPLETED")
    print("=" * 60)
    print("\nIf you see mostly ✅ symbols above, the HTTP 401 errors have been resolved!")
    print("If you see ❌ symbols, there may still be issues that need addressing.")

if __name__ == "__main__":
    main()