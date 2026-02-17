import pandas as pd
import yfinance as yf


def fetch_company_data(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, float, int, str]:
    """Fetch the balance sheet, income statement, current stock price, number of shares outstanding, and sector for a given company ticker symbol.

    Args:
        ticker (str): The ticker symbol of the company.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, float, int, str]: A tuple containing the balance sheet, income statement, current stock price, 
        number of shares outstanding, and sector for the company.
    """
    company = yf.Ticker(ticker)

    bs = company.balance_sheet.iloc[:, :-1]
    ist = company.income_stmt.iloc[:, :-1]

    price = yf.download(ticker, period="5d", progress=False, auto_adjust=True)["Close"].dropna().iloc[-1].values[0]

    info = company.info
    shares = info["sharesOutstanding"]
    sector = info["sector"]

    return bs, ist, price, shares, sector

def _row(df: pd.DataFrame, label: str) -> float:
    """Return the most-recent-period value for *label* from a statement."""
    return float(df.loc[label].iloc[0])
    
def working_capital(bs: pd.DataFrame) -> float:
    return _row(bs, "Working Capital")

def total_assets(bs: pd.DataFrame) -> float:
    return _row(bs, "Total Assets")

def retained_earnings(bs: pd.DataFrame) -> float:
    return _row(bs, "Retained Earnings")

def ebit(ist: pd.DataFrame) -> float:
    return _row(ist, "EBIT")

def total_revenue(ist: pd.DataFrame) -> float:
    return _row(ist, "Total Revenue")

def total_liabilities(bs: pd.DataFrame) -> float:
    return _row(bs, "Total Liabilities Net Minority Interest")

def book_equity(bs: pd.DataFrame) -> float:
    return _row(bs, "Stockholders Equity")

def market_value_of_equity(price: float, shares: int) -> float:
    return price * shares

def x1(bs: pd.DataFrame) -> float:
    # Calculate the first variable of the Altman Z-score, which is working capital divided by total assets.
    return working_capital(bs) / total_assets(bs)

def x2(bs: pd.DataFrame) -> float:
    # Calculate the second variable of the Altman Z-score, which is retained earnings divided by total assets.
    return retained_earnings(bs) / total_assets(bs)

def x3(bs: pd.DataFrame, ist: pd.DataFrame) -> float:
    # Calculate the third variable of the Altman 
    return ebit(ist) / total_assets(bs)

def x4(bs: pd.DataFrame, price: float, shares: int) -> float:
    # Calculate the fourth variable of the Altman Z-score, which is market value of equity divided by total liabilities.
    return market_value_of_equity(price, shares) / total_liabilities(bs)

def x4_modified(bs: pd.DataFrame) -> float:
    # Calculate the modified fourth variable of the Altman Z''-score, which is book value of equity divided by total liabilities.
    return book_equity(bs) / total_liabilities(bs)

def x5(bs: pd.DataFrame, ist: pd.DataFrame) -> float:
    # Calculate the fifth variable of the Altman Z-score, which is sales divided by total assets.
    return total_revenue(ist) / total_assets(bs)

def score(coefficients: tuple[float, ...], ratios: tuple[float, ...]) -> float:
    """Compute a weighted sum of the ratios using the provided coefficients."""
    return sum(c * r for c, r in zip(coefficients, ratios))


def altman_z_score(bs: pd.DataFrame, ist: pd.DataFrame, price: float, shares: int, coefficients: tuple[float, ...]) -> float:
    """A function that calculates the Altman Z-score given the balance sheet, income statement, current stock price, and number of shares outstanding for a company.

    Args:
        bs (pd.DataFrame): The balance sheet of the company.
        ist (pd.DataFrame): The income statement of the company.
        price (float): The current stock price of the company.
        shares (int): The number of shares outstanding for the company.

    Returns:
        float: The Altman Z-score for the company.
    """
    X1 = x1(bs)
    X2 = x2(bs)
    X3 = x3(bs, ist)
    X4 = x4(bs, price, shares)
    X5 = x5(bs, ist)
    return score(coefficients, [X1, X2, X3, X4, X5])

def altman_z_double_prime_score(bs: pd.DataFrame, ist: pd.DataFrame, coefficients: tuple[float, ...]) -> float:
    """A function that calculates the Altman Z''-score given the balance sheet and income statement for a company.

    Args:
        bs (pd.DataFrame): The balance sheet of the company.
        ist (pd.DataFrame): The income statement of the company.

    Returns:
        float: The Altman Z''-score for the company.
    """
    X1 = x1(bs)
    X2 = x2(bs)
    X3 = x3(bs, ist)
    X4_modified = x4_modified(bs)
    return score(coefficients, [X1, X2, X3, X4_modified])

def classify_z(z: float) -> str:
    """Classify a company based on its Altman Z-score.

    Args:
        z (float): The Altman Z-score of the company.

    Returns:
        str: A string indicating whether the company is in the distress zone, grey zone, or safe zone.
    """
    if z < 1.8:
        return "Distress zone. Do not approve for a loan."
    elif z < 3.0:
        return "Grey zone. Further analysis is needed."
    else:
        return "Safe zone. Approve for a loan."

def classify_z_double_prime(z_double_prime: float) -> str:
    """Classify a company based on its Altman Z''-score.

    Args:
        z_double_prime (float): The Altman Z''-score of the company.

    Returns:
        str: A string indicating whether the company is in the distress zone, grey zone, or safe zone.
    """
    if z_double_prime < 1.1:
        return "Distress zone. Do not approve for a loan."
    elif z_double_prime < 2.6:
        return "Grey zone. Further analysis is needed."
    else:
        return "Safe zone. Approve for a loan."

if __name__ == "__main__":
    ticker = input("Enter the ticker symbol of the company: ").upper()

    bs, ist, price, shares, sector = fetch_company_data(ticker)

    # According to the company sector is the original Altman Z-score or the modified Altman Z''-score used.
    z_sectors = ['Energy','Materials','Industrials','Consumer Defensive']
    z_double_prime_sectors = ['Consumer Cyclical', 'Healthcare', 'Technology', 'Communication Services', 'Financial Services', 'Utilities']

    # Original Altman Z-score (1968) — public manufacturing firms
    Z_SCORE_COEFFICIENTS = (1.2, 1.4, 3.3, 0.6, 1.0)

    # Altman Z''-score — non-manufacturing / emerging-market firms
    Z_DOUBLE_PRIME_COEFFICIENTS = (6.56, 3.26, 6.72, 1.05)

    if sector in z_sectors:
        # Compute the Altman Z-score for the company and classify it based on the score.
        z = altman_z_score(bs, ist, price, shares, Z_SCORE_COEFFICIENTS)
        print(f"The Altman Z-score for the company with ticker {ticker} is {z:.4f}")
        print(classify_z(z))

    elif sector in z_double_prime_sectors:
        # Compute the Altman Z''-score for the company and classify it based on the score.
        z_double_prime = altman_z_double_prime_score(bs, ist, Z_DOUBLE_PRIME_COEFFICIENTS)
        print(f"The Altman Z''-score for the company with ticker {ticker} is {z_double_prime:.4f}")
        print(classify_z_double_prime(z_double_prime))
       
    else:
        print(f"The sector of the company with ticker {ticker} is not recognized for Altman Z-score analysis.")
