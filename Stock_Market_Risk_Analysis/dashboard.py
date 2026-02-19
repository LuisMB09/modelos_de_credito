import streamlit as st
import pandas as pd
import yfinance as yf


# ===============================
# -------- DATA LAYER ----------
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
    return float(df.loc[label].iloc[0])


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
        return "Distress zone. ❌ Do not approve for a loan."
    elif z < 3.0:
        return "Grey zone. ⚠️ Further analysis is needed."
    else:
        return "Safe zone. ✅ Approve for a loan."


def classify_z_double_prime(z):
    if z < 1.1:
        return "Distress zone. ❌ Do not approve for a loan."
    elif z < 2.6:
        return "Grey zone. ⚠️ Further analysis is needed."
    else:
        return "Safe zone. ✅ Approve for a loan."

def show_risk_message(message: str):
    """
    Display the recommendation with background color
    depending on the risk zone.
    """

    if "Safe zone" in message:
        bg_color = "#d4edda"
        text_color = "#155724"
    elif "Grey zone" in message:
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


# ===============================
# -------- STREAMLIT UI --------
# ===============================

st.set_page_config(page_title="Altman Z-Score Dashboard", layout="centered")

st.title("📊 Altman Z-Score Dashboard")
st.markdown("Evaluate corporate credit risk using Altman's model.")

ticker = st.text_input("Enter company ticker symbol:", "").upper()

if ticker:

    try:
        with st.spinner("Fetching financial data..."):
            bs, ist, price, shares, sector = fetch_company_data(ticker)

        if bs.empty or ist.empty:
            st.error("❌ Ticker does not exist or has no financial data.")
            st.stop()
            
        if not shares or not sector:
            st.error("Insufficient company information.")
            st.stop()

        st.subheader("Company Information")
        st.write(f"**Ticker:** {ticker}")
        st.write(f"**Sector:** {sector}")
        st.write(f"**Current Price:** ${price:,.2f}")

        z_sectors = ['Energy','Materials','Industrials','Consumer Defensive']
        z_double_prime_sectors = [
            'Consumer Cyclical',
            'Healthcare',
            'Technology',
            'Communication Services',
            'Financial Services',
            'Utilities'
        ]

        if sector in z_sectors:
            z = altman_z_score(bs, ist, price, shares)
            recommendation = classify_z(z)

            st.subheader("Altman Z-Score (1968)")
            st.metric("Z-Score", f"{z:.4f}")
            show_risk_message(recommendation)


        elif sector in z_double_prime_sectors:
            z = altman_z_double_prime_score(bs, ist)
            recommendation = classify_z_double_prime(z)

            st.subheader("Altman Z''-Score (1995)")
            st.metric("Z''-Score", f"{z:.4f}")
            show_risk_message(recommendation)


        else:
            st.warning("Sector not recognized for Altman model classification.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
