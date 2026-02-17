"""
altman_api.py
FastAPI service for Altman Z-score and Z''-score credit risk analysis.

Run with:
    uvicorn altman_api:app --reload

API documentation will be available at:
    http://localhost:8000/docs
"""

from typing import Literal, Optional
from enum import Enum

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Altman Z-Score API",
    description="Credit risk assessment using Altman Z-score and Z''-score models",
    version="1.0.0",
)

# CORS middleware — adjust origins as needed for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Configuration — model coefficients and sector mappings
# ---------------------------------------------------------------------------

Z_SCORE_COEFFICIENTS = (1.2, 1.4, 3.3, 0.6, 1.0)
Z_DOUBLE_PRIME_COEFFICIENTS = (6.56, 3.26, 6.72, 1.05)

MANUFACTURING_SECTORS = ["Energy", "Materials", "Industrials", "Consumer Defensive"]
NON_MANUFACTURING_SECTORS = [
    "Consumer Cyclical",
    "Healthcare",
    "Technology",
    "Communication Services",
    "Financial Services",
    "Utilities",
]


# ---------------------------------------------------------------------------
# Pydantic models for request/response validation
# ---------------------------------------------------------------------------

class RiskZone(str, Enum):
    """Credit risk classification zones."""
    SAFE = "Safe zone"
    GREY = "Grey zone"
    DISTRESS = "Distress zone"


class ModelType(str, Enum):
    """Altman model variant."""
    Z = "Z"
    Z_DOUBLE_PRIME = "Z''"


class ZScoreResponse(BaseModel):
    """Complete Z-score analysis result."""
    ticker: str = Field(..., description="Company ticker symbol")
    sector: str = Field(..., description="Industry sector")
    model: ModelType = Field(..., description="Model used (Z or Z'')")
    score: float = Field(..., description="Computed Z-score value")
    zone: RiskZone = Field(..., description="Risk classification")
    recommendation: str = Field(..., description="Lending recommendation")
    ratios: dict[str, float] = Field(..., description="Individual ratio components")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "sector": "Technology",
                "model": "Z''",
                "score": 4.23,
                "zone": "Safe zone",
                "recommendation": "Approve for a loan.",
                "ratios": {
                    "X1_working_capital_to_assets": 0.15,
                    "X2_retained_earnings_to_assets": 0.82,
                    "X3_ebit_to_assets": 0.29,
                    "X4_equity_to_liabilities": 2.51,
                    "X5_sales_to_assets": 1.12,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    ticker: Optional[str] = None


# ---------------------------------------------------------------------------
# Core financial data functions (from original code)
# ---------------------------------------------------------------------------

def fetch_company_data(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, float, int, str]:
    """Fetch balance sheet, income statement, price, shares, and sector."""
    try:
        company = yf.Ticker(ticker)

        bs = company.balance_sheet.iloc[:, :-1]
        ist = company.income_stmt.iloc[:, :-1]

        price = (
            yf.download(ticker, period="5d", progress=False, auto_adjust=True)["Close"]
            .dropna()
            .iloc[-1]
            .values[0]
        )

        info = company.info
        shares = info["sharesOutstanding"]
        sector = info.get("sector", "Unknown")

        return bs, ist, price, shares, sector

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Unable to fetch data for ticker '{ticker}': {str(e)}",
        )


def _row(df: pd.DataFrame, label: str) -> float:
    """Extract most recent value for a given row label."""
    try:
        return float(df.loc[label].iloc[0])
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"Required financial data '{label}' not found in statement",
        )


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


# ---------------------------------------------------------------------------
# Ratio calculations (X1 … X5)
# ---------------------------------------------------------------------------

def x1(bs: pd.DataFrame) -> float:
    """Working capital / Total assets."""
    return working_capital(bs) / total_assets(bs)


def x2(bs: pd.DataFrame) -> float:
    """Retained earnings / Total assets."""
    return retained_earnings(bs) / total_assets(bs)


def x3(bs: pd.DataFrame, ist: pd.DataFrame) -> float:
    """EBIT / Total assets."""
    return ebit(ist) / total_assets(bs)


def x4(bs: pd.DataFrame, price: float, shares: int) -> float:
    """Market value of equity / Total liabilities."""
    return market_value_of_equity(price, shares) / total_liabilities(bs)


def x4_modified(bs: pd.DataFrame) -> float:
    """Book value of equity / Total liabilities."""
    return book_equity(bs) / total_liabilities(bs)


def x5(bs: pd.DataFrame, ist: pd.DataFrame) -> float:
    """Total revenue / Total assets."""
    return total_revenue(ist) / total_assets(bs)


def score(coefficients: tuple[float, ...], ratios: list[float]) -> float:
    """Weighted sum of ratios."""
    return sum(c * r for c, r in zip(coefficients, ratios))


# ---------------------------------------------------------------------------
# Z-score computation functions
# ---------------------------------------------------------------------------

def altman_z_score(
    bs: pd.DataFrame, ist: pd.DataFrame, price: float, shares: int
) -> tuple[float, dict[str, float]]:
    """Compute original Altman Z-score and return score + ratios."""
    X1 = x1(bs)
    X2 = x2(bs)
    X3 = x3(bs, ist)
    X4 = x4(bs, price, shares)
    X5 = x5(bs, ist)

    z = score(Z_SCORE_COEFFICIENTS, [X1, X2, X3, X4, X5])

    ratios = {
        "X1_working_capital_to_assets": X1,
        "X2_retained_earnings_to_assets": X2,
        "X3_ebit_to_assets": X3,
        "X4_market_equity_to_liabilities": X4,
        "X5_sales_to_assets": X5,
    }

    return z, ratios


def altman_z_double_prime_score(
    bs: pd.DataFrame, ist: pd.DataFrame
) -> tuple[float, dict[str, float]]:
    """Compute Altman Z''-score and return score + ratios."""
    X1 = x1(bs)
    X2 = x2(bs)
    X3 = x3(bs, ist)
    X4_mod = x4_modified(bs)

    z = score(Z_DOUBLE_PRIME_COEFFICIENTS, [X1, X2, X3, X4_mod])

    ratios = {
        "X1_working_capital_to_assets": X1,
        "X2_retained_earnings_to_assets": X2,
        "X3_ebit_to_assets": X3,
        "X4_book_equity_to_liabilities": X4_mod,
    }

    return z, ratios


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

def classify_z(z: float) -> tuple[RiskZone, str]:
    """Classify original Z-score."""
    if z < 1.8:
        return RiskZone.DISTRESS, "Do not approve for a loan."
    elif z < 3.0:
        return RiskZone.GREY, "Further analysis is needed."
    else:
        return RiskZone.SAFE, "Approve for a loan."


def classify_z_double_prime(z: float) -> tuple[RiskZone, str]:
    """Classify Z''-score."""
    if z < 1.1:
        return RiskZone.DISTRESS, "Do not approve for a loan."
    elif z < 2.6:
        return RiskZone.GREY, "Further analysis is needed."
    else:
        return RiskZone.SAFE, "Approve for a loan."


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    """API health check and info."""
    return {
        "service": "Altman Z-Score API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analyze": "/analyze/{ticker}",
            "docs": "/docs",
            "health": "/health",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health check endpoint."""
    return {"status": "healthy"}


@app.get(
    "/analyze/{ticker}",
    response_model=ZScoreResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Ticker not found"},
        422: {"model": ErrorResponse, "description": "Missing financial data"},
    },
    tags=["Analysis"],
)
async def analyze_company(
    ticker: str = Path(
        ...,
        description="Company ticker symbol (e.g., AAPL, TSLA, MSFT)",
        min_length=1,
        max_length=10,
    )
):
    """
    Perform Altman Z-score credit risk analysis for a given company.

    The API automatically selects the appropriate model based on the company's sector:
    - **Z-score**: Manufacturing sectors (Energy, Materials, Industrials, Consumer Defensive)
    - **Z''-score**: Non-manufacturing sectors (Technology, Healthcare, Financial Services, etc.)

    Returns the computed score, risk classification, and individual ratio components.
    """
    ticker = ticker.upper()

    # Fetch company data
    bs, ist, price, shares, sector = fetch_company_data(ticker)

    # Determine model and compute score
    if sector in MANUFACTURING_SECTORS:
        z_value, ratios = altman_z_score(bs, ist, price, shares)
        zone, recommendation = classify_z(z_value)
        model = ModelType.Z

    elif sector in NON_MANUFACTURING_SECTORS:
        z_value, ratios = altman_z_double_prime_score(bs, ist)
        zone, recommendation = classify_z_double_prime(z_value)
        model = ModelType.Z_DOUBLE_PRIME

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Sector '{sector}' is not recognized for Altman Z-score analysis. "
            f"Supported sectors: {MANUFACTURING_SECTORS + NON_MANUFACTURING_SECTORS}",
        )

    return ZScoreResponse(
        ticker=ticker,
        sector=sector,
        model=model,
        score=round(z_value, 4),
        zone=zone,
        recommendation=recommendation,
        ratios={k: round(v, 4) for k, v in ratios.items()},
    )


@app.get(
    "/sectors",
    response_model=dict[str, list[str]],
    tags=["Reference"],
)
async def get_supported_sectors():
    """
    Return the list of supported sectors for each model type.

    Useful for understanding which model will be applied to a given company.
    """
    return {
        "Z_score_sectors": MANUFACTURING_SECTORS,
        "Z_double_prime_sectors": NON_MANUFACTURING_SECTORS,
    }


# ---------------------------------------------------------------------------
# Run instructions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "altman_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )