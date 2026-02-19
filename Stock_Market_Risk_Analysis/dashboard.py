import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')

# ===============================
# -------- ALTMAN MODEL --------
# ===============================

@st.cache_data(show_spinner=False)
def fetch_company_data(ticker: str):
    company = yf.Ticker(ticker)

    bs = company.balance_sheet.iloc[:, :-1]
    ist = company.income_stmt.iloc[:, :-1]

    price_data = yf.download(
        ticker,
        period="5d",
        progress=False,
        auto_adjust=True
    )["Close"].dropna()

    price = float(price_data.iloc[-1])

    info = company.info
    shares = info.get("sharesOutstanding")
    sector = info.get("sector")

    return bs, ist, price, shares, sector


def _row(df: pd.DataFrame, label: str) -> float:
    try:
        return float(df.loc[label].iloc[0])
    except:
        return 0.0


def working_capital(bs): return _row(bs, "Working Capital")
def total_assets(bs): return _row(bs, "Total Assets")
def retained_earnings(bs): return _row(bs, "Retained Earnings")
def ebit(ist): return _row(ist, "EBIT")
def total_revenue(ist): return _row(ist, "Total Revenue")
def total_liabilities(bs): return _row(bs, "Total Liabilities Net Minority Interest")
def book_equity(bs): return _row(bs, "Stockholders Equity")


def market_value_of_equity(price, shares):
    return price * shares


def x1(bs): return working_capital(bs) / total_assets(bs)
def x2(bs): return retained_earnings(bs) / total_assets(bs)
def x3(bs, ist): return ebit(ist) / total_assets(bs)
def x4(bs, price, shares): return market_value_of_equity(price, shares) / total_liabilities(bs)
def x4_modified(bs): return book_equity(bs) / total_liabilities(bs)
def x5(bs, ist): return total_revenue(ist) / total_assets(bs)


def score(coefficients, ratios):
    return sum(c * r for c, r in zip(coefficients, ratios))


def altman_z_score(bs, ist, price, shares):
    Z_SCORE_COEFFICIENTS = (1.2, 1.4, 3.3, 0.6, 1.0)
    ratios = (x1(bs), x2(bs), x3(bs, ist), x4(bs, price, shares), x5(bs, ist))
    return score(Z_SCORE_COEFFICIENTS, ratios)


def altman_z_double_prime_score(bs, ist):
    Z_DOUBLE_PRIME_COEFFICIENTS = (6.56, 3.26, 6.72, 1.05)
    ratios = (x1(bs), x2(bs), x3(bs, ist), x4_modified(bs))
    return score(Z_DOUBLE_PRIME_COEFFICIENTS, ratios)


def classify_z(z):
    if z < 1.8:
        return "distress", "Distress zone. ❌ High bankruptcy risk."
    elif z < 3.0:
        return "grey", "Grey zone. ⚠️ Further analysis is needed."
    else:
        return "safe", "Safe zone. ✅ Low bankruptcy risk."


def classify_z_double_prime(z):
    if z < 1.1:
        return "distress", "Distress zone. ❌ High bankruptcy risk."
    elif z < 2.6:
        return "grey", "Grey zone. ⚠️ Further analysis is needed."
    else:
        return "safe", "Safe zone. ✅ Low bankruptcy risk."


# ===============================
# -------- MERTON MODEL --------
# ===============================

def get_financials_merton(ticker: str):
    bs = yf.Ticker(ticker).balance_sheet.iloc[:, :-1]
    ist = yf.Ticker(ticker).income_stmt.iloc[:, :-1]
    cf = yf.Ticker(ticker).cash_flow.iloc[:, :-1]
    return bs, ist, cf


def price_book_value(ticker: str) -> float:
    """Retrieve the price-to-book ratio for a given ticker."""
    company = yf.Ticker(ticker)
    return company.info.get("priceToBook", np.nan)


def assets(bs: pd.DataFrame) -> float:
    """Calculate total assets."""
    return _row(bs, "Total Assets")


def debt(bs: pd.DataFrame) -> float:
    """Calculate the debt."""
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
    """
    A, sigma_A = x
    
    if A <= 0 or sigma_A <= 0:
        return np.array([1e10, 1e10])
    
    d1 = (np.log(A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    eq1 = A * N_d1 - D * np.exp(-r * T) * N_d2 - E
    eq2 = (A / E) * N_d1 * sigma_A - sigma_E
    
    return np.array([eq1, eq2])


def sigma_assets(E: float, D: float, sigma_E: float, r: float, T: float) -> tuple:
    """
    Solve for asset volatility (sigma_A) and asset value (A) using Merton model.
    """
    A0 = E + D
    sigma_A0 = sigma_E * (E / (E + D))
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            solution, info, ier, msg = fsolve(
                merton_equations, 
                x0=[A0, sigma_A0], 
                args=(E, sigma_E, D, r, T),
                full_output=True
            )
        
        A, sigma_A = solution
        
        if ier == 1 and A > 0 and sigma_A > 0:
            return A, sigma_A, True
        else:
            return A0, sigma_A0, False
            
    except Exception:
        return A0, sigma_A0, False


def distance_to_default(A: float, D: float, rf: float, sigma_A: float, T: float) -> float:
    """Calculate the distance to default."""
    return (np.log(A / D) + (rf - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))


def probability_of_default(ticker: str, rf: float, T: float = 2.0) -> dict:
    """
    Calculate the probability of default using the Merton model.
    """
    bs, _, _ = get_financials_merton(ticker)
    A_book = assets(bs)
    D = debt(bs)
    E = market_value_equity(ticker)
    sigma_E = sigma_equity(ticker)
    p_bv = price_book_value(ticker)
    
    A, sigma_A, converged = sigma_assets(E, D, sigma_E, rf, T)
    
    dd = distance_to_default(A, D, rf, sigma_A, T)
    pd = norm.cdf(-dd)
    
    A_market_to_book = A / A_book if A_book > 0 else np.nan
    leverage_market = D / A if A > 0 else np.nan
    
    return {
        'ticker': ticker,
        'A_market': A,
        'sigma_A': sigma_A,
        'E': E,
        'sigma_E': sigma_E,
        'D': D,
        'A_book': A_book,
        'r': rf,
        'T': T,
        'DD': dd,
        'PD': pd,
        'P_BV': p_bv,
        'A_market_to_book': A_market_to_book,
        'leverage_market': leverage_market,
        'converged': converged
    }


def classify_merton_risk(pd: float) -> tuple:
    """Classify the risk level based on probability of default."""
    if pd < 0.0030:
        return "low", "Low default risk. ✅ PD < 0.30%"
    else:
        return "high", "High default risk. ❌ PD ≥ 0.30%"


# ===============================
# ----- COMBINED DECISION ------
# ===============================

def final_loan_decision(altman_zone: str, merton_zone: str, z_score: float, pd: float) -> tuple:
    """
    Make final loan approval decision based on both models.
    
    Rules:
    - Altman safe + Merton low risk → Approve
    - Altman distress OR (Altman grey + Merton high risk) → Deny
    - Altman grey + Merton low risk → Further analysis needed
    """
    
    if altman_zone == "safe" and merton_zone == "low":
        return "approve", f"✅ **LOAN APPROVED**\n\nAltman Z-Score: {z_score:.2f} (Safe zone)\n\nMerton PD: {pd:.2%} (Low risk)\n\nBoth models indicate low credit risk."
    
    elif altman_zone == "distress" or (altman_zone == "grey" and merton_zone == "high"):
        return "deny", f"❌ **LOAN DENIED**\n\nAltman Z-Score: {z_score:.2f} ({altman_zone.title()} zone)\n\nMerton PD: {pd:.2%} ({merton_zone.title()} risk)\n\nCredit risk is too high for loan approval."
    
    elif altman_zone == "grey" and merton_zone == "low":
        return "analysis", f"⚠️ **FURTHER ANALYSIS REQUIRED**\n\nAltman Z-Score: {z_score:.2f} (Grey zone)\n\nMerton PD: {pd:.2%} (Low risk)\n\nMixed signals require deeper credit analysis before decision."
    
    else:
        return "analysis", f"⚠️ **FURTHER ANALYSIS REQUIRED**\n\nAltman Z-Score: {z_score:.2f}\n\nMerton PD: {pd:.2%}\n\nAdditional evaluation needed."


# ===============================
# ------- UI COMPONENTS --------
# ===============================

def show_risk_message(message: str):
    """Display the recommendation with background color depending on the risk zone."""
    
    if "Safe zone" in message or "Low default" in message:
        bg_color = "#d4edda"
        text_color = "#155724"
    elif "Grey zone" in message or "Further analysis" in message:
        bg_color = "#fff3cd"
        text_color = "#856404"
    else:  # Distress
        bg_color = "#f8d7da"
        text_color = "#721c24"

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            color:{text_color};
            padding:18px;
            border-radius:10px;
            font-size:17px;
            font-weight:600;
            text-align:center;">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def show_final_decision(decision_type: str, message: str):
    """Display final loan decision with appropriate styling."""
    
    if decision_type == "approve":
        bg_color = "#d4edda"
        text_color = "#155724"
    elif decision_type == "deny":
        bg_color = "#f8d7da"
        text_color = "#721c24"
    else:  # analysis
        bg_color = "#fff3cd"
        text_color = "#856404"

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            color:{text_color};
            padding:25px;
            border-radius:12px;
            font-size:18px;
            font-weight:700;
            text-align:center;
            border: 3px solid {'#28a745' if decision_type == 'approve' else '#dc3545' if decision_type == 'deny' else '#ffc107'};
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            {message.replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True
    )


# ===============================
# -------- STREAMLIT UI --------
# ===============================

st.set_page_config(page_title="Credit Risk Analysis Dashboard", layout="wide")

st.title("📊 Credit Risk Analysis Dashboard")
st.markdown("Comprehensive credit evaluation using **Altman Z-Score** and **Merton Structural Model**")
st.markdown("The Altman method is used to score the company with a time horizon of two years, therefore, the Merton model is calculated with a time to maturity of two years to match both models.")
st.markdown("**Disclaimer:** The dashboard only works for American companies with financial data available on Yahoo Finance. Banks are exluded due to differences on financial statements reports.")

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter company ticker symbol:", "").upper()
with col2:
    rf = st.number_input("Risk-free rate (%):", min_value=0.0, max_value=20.0, value=3.5, step=0.1) / 100

if ticker:
    try:
        with st.spinner("Fetching financial data..."):
            bs, ist, price, shares, sector = fetch_company_data(ticker)

        if bs.empty or ist.empty:
            st.error("❌ Ticker does not exist or has no financial data.")
            st.stop()
            
        if not shares or not sector:
            st.error("❌ Insufficient company information.")
            st.stop()

        # Company Information
        st.markdown("---")
        st.subheader("📋 Company Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ticker", ticker)
        with col2:
            st.metric("Sector", sector)
        with col3:
            st.metric("Current Price", f"${price:,.2f}")

        st.markdown("---")
        
        # ==================
        # ALTMAN Z-SCORE
        # ==================
        
        st.subheader("📈 Altman Z-Score Analysis")
        
        z_sectors = ['Energy','Materials','Industrials','Consumer Defensive']
        z_double_prime_sectors = [
            'Consumer Cyclical',
            'Healthcare',
            'Technology',
            'Communication Services',
            'Financial Services',
            'Utilities'
        ]

        altman_zone = None
        z_value = None
        
        if sector in z_sectors:
            z_value = altman_z_score(bs, ist, price, shares)
            altman_zone, recommendation = classify_z(z_value)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Altman Z-Score (1968)", f"{z_value:.4f}")
            with col2:
                show_risk_message(recommendation)

        elif sector in z_double_prime_sectors:
            z_value = altman_z_double_prime_score(bs, ist)
            altman_zone, recommendation = classify_z_double_prime(z_value)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Altman Z''-Score (1995)", f"{z_value:.4f}")
            with col2:
                show_risk_message(recommendation)

        else:
            st.warning("⚠️ Sector not recognized for Altman model classification.")
            altman_zone = "unknown"

        st.markdown("---")
        
        # ==================
        # MERTON MODEL
        # ==================
        
        st.subheader("🎯 Merton Structural Model Analysis")
        
        with st.spinner("Calculating Merton model..."):
            merton_results = probability_of_default(ticker, rf, T=1.0)
        
        # Merton metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Probability of Default",
                f"{merton_results['PD']:.2%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Distance to Default",
                f"{merton_results['DD']:.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Asset Volatility (σ_A)",
                f"{merton_results['sigma_A']:.2%}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Market Leverage",
                f"{merton_results['leverage_market']:.1%}",
                delta=None
            )
        
        # Merton classification
        merton_zone, merton_msg = classify_merton_risk(merton_results['PD'])
        show_risk_message(merton_msg)
        
        # Detailed Merton metrics in expander
        with st.expander("📊 View Detailed Merton Model Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Market Values**")
                st.write(f"Implied Market Assets: ${merton_results['A_market']:,.0f}")
                st.write(f"Market Equity: ${merton_results['E']:,.0f}")
                st.write(f"Total Debt: ${merton_results['D']:,.0f}")
                st.write(f"Equity Volatility (σ_E): {merton_results['sigma_E']:.2%}")
            
            with col2:
                st.markdown("**Valuation Ratios**")
                st.write(f"Book Assets: ${merton_results['A_book']:,.0f}")
                st.write(f"Market/Book Assets: {merton_results['A_market_to_book']:.2f}x")
                st.write(f"Price-to-Book: {merton_results['P_BV']:.2f}x")
                st.write(f"Risk-Free Rate: {merton_results['r']:.2%}")
            
            if not merton_results['converged']:
                st.warning("⚠️ Solver used naive estimate (convergence issue)")

        st.markdown("---")
        
        # ==================
        # FINAL DECISION
        # ==================
        
        st.subheader("🎯 Final Loan Decision")
        
        if altman_zone and altman_zone != "unknown" and z_value is not None:
            decision_type, decision_message = final_loan_decision(
                altman_zone, 
                merton_zone, 
                z_value, 
                merton_results['PD']
            )
            
            show_final_decision(decision_type, decision_message)
            
            # Summary table
            st.markdown("---")
            st.markdown("**📋 Summary Table**")
            
            summary_data = {
                "Model": ["Altman Z-Score", "Merton Structural"],
                "Score/Metric": [f"{z_value:.4f}", f"{merton_results['PD']:.2%} PD"],
                "Classification": [
                    recommendation.split('.')[0],
                    merton_msg.split('.')[0]
                ],
                "Risk Level": [altman_zone.title(), merton_zone.title()]
            }
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        else:
            st.info("ℹ️ Complete Altman analysis required for final decision.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())
