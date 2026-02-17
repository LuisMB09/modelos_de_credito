"""
test_client.py
Example client demonstrating how to interact with the Altman Z-Score API.

Run the API first with:
    uvicorn altman_api:app --reload

Then run this script:
    python test_client.py
"""

import requests
from typing import Optional


BASE_URL = "http://localhost:8000"


def analyze_ticker(ticker: str) -> Optional[dict]:
    """Call the /analyze endpoint for a given ticker."""
    try:
        response = requests.get(f"{BASE_URL}/analyze/{ticker}")
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.json()}")
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def get_supported_sectors() -> Optional[dict]:
    """Retrieve the list of supported sectors."""
    try:
        response = requests.get(f"{BASE_URL}/sectors")
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def print_analysis(result: dict) -> None:
    """Pretty-print the analysis result."""
    print(f"\n{'='*70}")
    print(f"ALTMAN Z-SCORE ANALYSIS: {result['ticker']}")
    print(f"{'='*70}")
    print(f"Sector:          {result['sector']}")
    print(f"Model Used:      {result['model']}")
    print(f"Z-Score:         {result['score']:.4f}")
    print(f"Risk Zone:       {result['zone']}")
    print(f"Recommendation:  {result['recommendation']}")
    print(f"\nRatio Components:")
    for ratio_name, ratio_value in result['ratios'].items():
        print(f"  • {ratio_name:.<50} {ratio_value:.4f}")
    print(f"{'='*70}\n")


def main():
    """Run example API calls."""
    print("🚀 Altman Z-Score API Test Client")
    print("="*70)
    
    # Health check
    try:
        health = requests.get(f"{BASE_URL}/health").json()
        print(f"✅ API Status: {health['status']}\n")
    except:
        print("❌ API is not running. Start it with: uvicorn altman_api:app --reload\n")
        return

    # Get supported sectors
    print("\n📋 Supported Sectors:")
    sectors = get_supported_sectors()
    if sectors:
        print(f"\n  Z-Score (Manufacturing):")
        for sector in sectors['Z_score_sectors']:
            print(f"    • {sector}")
        print(f"\n  Z''-Score (Non-Manufacturing):")
        for sector in sectors['Z_double_prime_sectors']:
            print(f"    • {sector}")

    # Example analyses
    test_tickers = ['PG','AMZN','V','RL']
    
    for ticker in test_tickers:
        print(f"\n🔍 Analyzing {ticker}...")
        result = analyze_ticker(ticker)
        if result:
            print_analysis(result)
        else:
            print(f"⚠️  Could not analyze {ticker}\n")

    # Interactive mode
    print("\n" + "="*70)
    print("🎯 Interactive Mode")
    print("="*70)
    
    while True:
        ticker = input("\nEnter ticker symbol (or 'quit' to exit): ").strip().upper()
        
        if ticker in ['QUIT', 'Q', 'EXIT']:
            print("👋 Goodbye!")
            break
        
        if not ticker:
            continue
        
        result = analyze_ticker(ticker)
        if result:
            print_analysis(result)


if __name__ == "__main__":
    main()