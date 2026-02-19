import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import fsolve


def get_financials(ticker: str) -> pd.DataFrame:
    bs = yf.Ticker(ticker).balance_sheet.iloc[:, :-1]
    ist = yf.Ticker(ticker).income_stmt.iloc[:, :-1]
    cf = yf.Ticker(ticker).cash_flow.iloc[:, :-1]
    return bs, ist, cf

def price_book_value(ticker: str) -> float:
    """Retrieve the price-to-book ratio for a given ticker."""
    company = yf.Ticker(ticker)
    return company.info.get("priceToBook")

def _row(df: pd.DataFrame, label: str) -> float:
    """Return the most-recent-period value for *label* from a statement."""
    return float(df.loc[label].iloc[0])

def assets(bs: pd.DataFrame) -> float:
    """Calculate total assets."""
    return _row(bs, "Total Assets")

def debt(bs: pd.DataFrame) -> float:
    """Calculate the debt using KMV methodology: STD + 0.5*LTD."""
    #return _row(bs, 'Current Debt') + _row(bs, 'Long Term Debt')
    return _row(bs, 'Total Debt')

def market_value_equity(ticker: str) -> float:
    """Calculate the market value of equity."""
    company = yf.Ticker(ticker)
    return company.info['sharesOutstanding'] * company.info['currentPrice']

def sigma_equity(ticker: str) -> float:
    """Calculate the volatility of equity returns."""
    data = yf.download(ticker, period='2y', progress=False, auto_adjust=True)['Close']
    rt = data.pct_change().dropna()
    return (rt.std() * np.sqrt(252)).values[0]

def merton_equations(x: np.ndarray, E: float, sigma_E: float, D: float, 
                     r: float, T: float) -> np.ndarray:
    """
    System of equations to solve for A and sigma_A.
    
    Eq 1: E = A·N(d1) - D·exp(-rT)·N(d2)
    Eq 2: σ_E = (A/E)·N(d1)·σ_A
    """
    A, sigma_A = x
    
    # Prevent invalid values
    if A <= 0 or sigma_A <= 0:
        return np.array([1e10, 1e10])
    
    d1 = (np.log(A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    # Equation 1: Black-Scholes equity value
    eq1 = A * N_d1 - D * np.exp(-r * T) * N_d2 - E
    
    # Equation 2: Volatility relationship
    eq2 = (A / E) * N_d1 * sigma_A - sigma_E
    
    return np.array([eq1, eq2])

def sigma_assets(E: float, D: float, sigma_E: float, r: float, T: float) -> tuple:
    """
    Solve for asset volatility (sigma_A) and asset value (A) using Merton model.
    
    Returns:
    --------
    tuple: (A, sigma_A)
    """
    # Initial guesses
    A0 = E + D
    sigma_A0 = sigma_E * (E / (E + D))
    
    # Solve system of equations
    solution = fsolve(merton_equations, x0=[A0, sigma_A0], 
                     args=(E, sigma_E, D, r, T))
    
    A, sigma_A = solution
    
    return A, sigma_A

def distance_to_default(A: float, D: float, rf: float, sigma_A: float, T: float) -> float:
    """Calculate the distance to default."""
    return (np.log(A / D) + (rf - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))

def probability_of_default(ticker: str, rf: float, T: float|int = 2.0) -> float:
    """A function to calculate the probability of default using the Merton model.

    Args:
        ticker (str): The stock ticker symbol of the company.
        rf (float): The risk-free rate.
        T (float, optional): Time to maturity in years. Defaults to 2.0.

    Returns:
        float: The probability of default.
    """
    # Get financial data
    bs, _, _ = get_financials(ticker)
    D = debt(bs)
    E = market_value_equity(ticker)
    sigma_E = sigma_equity(ticker)
    p_bv = price_book_value(ticker)
    
    # Solve for asset value and volatility
    A, sigma_A = sigma_assets(E, D, sigma_E, rf, T)
    
    # Calculate distance to default
    dd = distance_to_default(A, D, rf, sigma_A, T)
    
    # Probability of default: P(A_T < D) = 1 - N(DD)
    pd = 1 - norm.cdf(dd)

    summary = {
        "Ticker": ticker,
        "Asset Value (A)": A,
        "Asset Volatility (σ_A)": sigma_A,
        "Distance to Default (DD)": dd,
        "Probability of Default (PD)": pd,
        "Price-to-Book Ratio": p_bv
    }
    
    return summary

def classify_risk(pd: float) -> str:
    """Classify the risk level based on probability of default."""
    if pd < 0.0030:
        return "At least A rating. Loan approved"
    else:
        return "Below A rating. Loan denied"

if __name__ == "__main__":

    ticker = input("Enter the stock ticker symbol: ").upper()
    rf = float(input("Enter the risk-free rate (e.g., 0.035 for 3.5%): "))
    T = float(input("Enter time to maturity in years (default 2): ") or 2)

    bs, ist, cf = get_financials(ticker)
    A = assets(bs)
    D = debt(bs)
    E = market_value_equity(ticker)

    summary = probability_of_default(ticker, rf, T)
    print(f"\nMerton Model Summary for {ticker}")
    print(f'Probability of Default: {summary["Probability of Default (PD)"]:.4%}')
    print(f'Distance to Default (DD): {summary["Distance to Default (DD)"]:.4f}')
    print(f'Asset Volatility (σ_A): {summary["Asset Volatility (σ_A)"]:.4f}')
    print(f'Market Assets Value (A): ${summary["Asset Value (A)"]:,.2f}')
    print(f'Price-to-Book Ratio: {summary["Price-to-Book Ratio"]:.4f}\n')

    print(f'Book Value of Assets (A): ${A:,.2f}')
    print(f'Book Value of Debt (D): ${D:,.2f}')
    print(f'Market Value of Equity (E): ${E:,.2f}')

    print(f'\nRisk Classification: {classify_risk(summary["Probability of Default (PD)"])}')
