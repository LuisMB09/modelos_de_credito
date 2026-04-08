import os
import shap
import warnings
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay

np.random.seed(42)
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# LAYER 0 — BASE EXTRACTOR
# Single responsibility: pull one column from a DataFrame.
# All feature functions compose from this.
# ══════════════════════════════════════════════════════════════════

def _col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a single column from a DataFrame as a Series."""
    return df[col]


# ══════════════════════════════════════════════════════════════════
# LAYER 1 — DATA INGESTION & PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def load_data(dataset_slug: str, filename: str) -> pd.DataFrame:
    """
    Download a dataset from Kaggle and load it into a DataFrame.

    Args:
        dataset_slug (str): Kaggle dataset identifier (e.g. 'user/dataset-name').
        filename     (str): CSV filename inside the downloaded folder.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    path = kagglehub.dataset_download(dataset_slug)
    return pd.read_csv(f"{path}/{filename}")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical (object-dtype) columns.

    Args:
        df (pd.DataFrame): Raw DataFrame with categorical columns.

    Returns:
        pd.DataFrame: DataFrame with categorical columns replaced by dummies.
    """
    cat_cols = df.select_dtypes(include='object').columns
    return pd.get_dummies(df, columns=cat_cols)


# ══════════════════════════════════════════════════════════════════
# LAYER 2 — MORTGAGE FEATURE FUNCTIONS
# Each function computes one ratio and composes from _col().
# Mirrors the x1(), x2(), x3()… structure of Altman Z-score.
# ══════════════════════════════════════════════════════════════════

def total_debt(df: pd.DataFrame) -> pd.Series:
    """Total debt: new loan request plus existing mortgage balance."""
    return _col(df, 'LOAN') + _col(df, 'MORTDUE')


def ltv(df: pd.DataFrame) -> pd.Series:
    """
    Loan-to-Value ratio: new loan vs. appraised property value.
    Primary collateral risk metric for mortgage underwriting.
    """
    return _col(df, 'LOAN') / _col(df, 'VALUE')


def cltv(df: pd.DataFrame) -> pd.Series:
    """
    Combined Loan-to-Value: total debt (new + existing) vs. property value.
    Captures full leverage exposure on the collateral.
    """
    return total_debt(df) / _col(df, 'VALUE')


def home_equity(df: pd.DataFrame) -> pd.Series:
    """Owner equity in the property: appraised value minus existing mortgage."""
    return _col(df, 'VALUE') - _col(df, 'MORTDUE')


def equity_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Equity ratio: owner equity as a fraction of property value.
    Complement of LTV for the existing mortgage.
    """
    return home_equity(df) / _col(df, 'VALUE')


def delinq_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Delinquency ratio: delinquent credit lines over total credit lines.
    +1 in denominator avoids division by zero for clients with no credit lines.
    """
    return _col(df, 'DELINQ') / (_col(df, 'CLNO') + 1)


def underwater(df: pd.DataFrame) -> pd.Series:
    """
    Underwater flag: 1 if total debt exceeds property value, 0 otherwise.
    Indicates negative equity — the strongest single predictor of LGD.
    """
    return (total_debt(df) > _col(df, 'VALUE')).astype(int)


# ══════════════════════════════════════════════════════════════════
# LAYER 3 — FEATURE ENGINEERING PIPELINE
# Composes all feature functions above into a single transformation.
# ══════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive mortgage-specific risk features from raw columns.

    Composes: total_debt(), ltv(), cltv(), home_equity(),
              equity_ratio(), delinq_ratio(), underwater().

    Note: HOME_EQUITY and TOTAL_DEBT are intermediate values used
    to compute other ratios. They are dropped before model training
    to prevent data leakage (they directly encode the target signal).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with raw columns.

    Returns:
        pd.DataFrame: DataFrame with derived features appended.
    """
    df = df.copy()
    df['TOTAL_DEBT']   = total_debt(df)
    df['LTV']          = ltv(df)
    df['CLTV']         = cltv(df)
    df['HOME_EQUITY']  = home_equity(df)
    df['EQUITY_RATIO'] = equity_ratio(df)
    df['DELINQ_RATIO'] = delinq_ratio(df)
    df['UNDERWATER']   = underwater(df)
    return df


# ══════════════════════════════════════════════════════════════════
# LAYER 4 — MODEL INPUTS
# ══════════════════════════════════════════════════════════════════

def prepare_model_inputs(
    df          : pd.DataFrame,
    target      : str,
    drop_cols   : list[str],
    test_size   : float = 0.2,
    random_state: int   = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into stratified train/test sets.

    Args:
        df           (pd.DataFrame): Full preprocessed DataFrame.
        target       (str)         : Name of the binary target column.
        drop_cols    (list[str])   : Columns to exclude from features
                                     (intermediates or leakage sources).
        test_size    (float)       : Fraction reserved for testing.
        random_state (int)         : Reproducibility seed.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# ══════════════════════════════════════════════════════════════════
# LAYER 5 — MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test : pd.DataFrame,
    y_test : pd.Series
) -> XGBClassifier:
    """
    Train an XGBoost binary classifier for PD estimation.

    Hyperparameters follow conservative settings to reduce overfitting:
    shallow trees (max_depth=8), low learning rate (0.01),
    and feature/row subsampling (0.6).

    Args:
        X_train, y_train: Training features and labels.
        X_test,  y_test : Validation set used for early-stopping eval.

    Returns:
        XGBClassifier: Fitted model.
    """
    model = XGBClassifier(
        objective        = 'binary:logistic',
        eval_metric      = 'auc',
        n_estimators     = 500,
        max_depth        = 8,
        learning_rate    = 0.01,
        min_child_weight = 3,
        subsample        = 0.6,
        colsample_bytree = 0.6,
        tree_method      = 'hist',
        random_state     = 42,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=0)
    return model


def curva_roc(probabilidades: np.ndarray, y_test: np.ndarray) -> float:
    """
    Plot the ROC curve and return the optimal classification threshold.

    The optimal threshold minimises the Euclidean distance to the
    perfect-classifier point (FPR=0, TPR=1) on the ROC curve.

    Args:
        probabilidades (np.ndarray): Predicted probabilities of default.
        y_test         (np.ndarray): True binary labels.

    Returns:
        float: Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_score=probabilidades, y_true=y_test)

    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_idx  = np.argmin(distances)
    best_thr  = thresholds[best_idx]
    best_fpr  = fpr[best_idx]
    best_tpr  = tpr[best_idx]
    auc_score = roc_auc_score(y_test, probabilidades)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2E86AB')
    ax.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'AUC = {auc_score:.3f}')
    ax.plot([0, 1], [0, 1], color='#E84855', lw=1.5, linestyle='--',
            label='Random classifier')
    ax.plot([best_fpr, best_fpr], [0, best_tpr],
            color='#F4A261', lw=1, linestyle=':')
    ax.plot([0, best_fpr], [best_tpr, best_tpr],
            color='#F4A261', lw=1, linestyle=':')
    ax.scatter(best_fpr, best_tpr, marker='*', s=400, color='#F4A261',
               zorder=5, label=(f'Best threshold = {best_thr:.3f}\n'
                                f'FPR={best_fpr:.3f}  TPR={best_tpr:.3f}'))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)',      fontsize=11)
    ax.set_title('ROC Curve — Credit Default XGBoost',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
    plt.tight_layout()
    plt.show()

    return best_thr


# ══════════════════════════════════════════════════════════════════
# LAYER 6 — EL COMPONENTS
# Three composable functions, one per Basel pillar: PD, LGD, EAD.
# Mirrors x4() / x4_modified() split in Altman.
# ══════════════════════════════════════════════════════════════════

def net_recovery(property_value: np.ndarray, haircut: float) -> np.ndarray:
    """
    Net recoverable value of the collateral after foreclosure costs.

    The haircut captures legal fees, maintenance during the judicial
    process, and the fire-sale discount. A 25% haircut is consistent
    with Mexican mortgage foreclosure timelines (2-4 years via juicio
    hipotecario) per CNBV supervisory guidance.

    Args:
        property_value (np.ndarray): Appraised value of the property.
        haircut        (float)     : Foreclosure cost as % of value.

    Returns:
        np.ndarray: Net recovery amount per loan.
    """
    return property_value * (1 - haircut)


def actual_recovery(
    ead           : np.ndarray,
    property_value: np.ndarray,
    haircut       : float
) -> np.ndarray:
    """
    Amount actually recovered by the bank at default.

    The bank can recover at most the outstanding balance (EAD) —
    surplus collateral value does not generate income.

    Args:
        ead            (np.ndarray): Exposure at default (MORTDUE).
        property_value (np.ndarray): Appraised property value.
        haircut        (float)     : Foreclosure cost fraction.

    Returns:
        np.ndarray: Actual recovery per loan, capped at EAD.
    """
    return np.minimum(ead, net_recovery(property_value, haircut))


def lgd_hipotecario(
    ead           : np.ndarray,
    property_value: np.ndarray,
    haircut       : float = 0.25
) -> np.ndarray:
    """
    Loss Given Default for mortgage loans via collateral recovery model.

    Composes net_recovery() and actual_recovery().

    LGD = 1 - actual_recovery / EAD, clipped to [0, 1].

    A 25% haircut is calibrated so that portfolio-average LGD is
    consistent with the ~35% supervisory estimate for Mexican
    residential mortgages (Basel II IRB Foundation approach).

    Args:
        ead            (np.ndarray): Outstanding mortgage balance (MORTDUE).
        property_value (np.ndarray): Appraised property value (VALUE).
        haircut        (float)     : Foreclosure cost fraction (default 25%).

    Returns:
        np.ndarray: LGD values in [0, 1], one per loan.
    """
    recovery = actual_recovery(ead, property_value, haircut)
    return np.clip(1 - recovery / ead, 0, 1)


def amortization(
    principal  : float | pd.Series,
    annual_rate: float,
    months     : int
) -> float | pd.Series:
    """
    Total amount paid over the life of a fixed-payment mortgage.

    Uses the standard French amortization formula:
        C = P * r / (1 - (1+r)^-n)
    where r = monthly rate, n = number of payments.

    Used to estimate opportunity cost: the foregone interest income
    on loans incorrectly rejected by the model.

    Args:
        principal   (float | pd.Series): Original loan amount(s).
        annual_rate (float)            : Annual rate as percentage (e.g. 11.5040).
        months      (int)              : Total number of monthly payments.

    Returns:
        float | pd.Series: Total amount paid (principal + interest).
    """
    r       = (annual_rate / 100) / 12
    payment = principal * (r / (1 - (1 + r) ** -months))
    return payment * months


# ══════════════════════════════════════════════════════════════════
# LAYER 7 — EL COMPUTATION 
# ══════════════════════════════════════════════════════════════════

def compute_el(data_test: pd.DataFrame, best_thr: float) -> pd.DataFrame:
    """
    Compute Expected Loss components and model classification for each loan.

    EL = PD × LGD × EAD, where:
        PD  = default_proba (XGBoost output)
        LGD = lgd_hipotecario(MORTDUE, VALUE)   [composes net_recovery, actual_recovery]
        EAD = MORTDUE                            [outstanding balance, observed]

    Also assigns binary model_prediction using best_thr from curva_roc().

    Args:
        data_test (pd.DataFrame): Test set with default_proba already added.
        best_thr  (float)       : Optimal classification threshold.

    Returns:
        pd.DataFrame: Input DataFrame with LGD, EL_pct, EL_amount,
                      and model_prediction columns added.
    """
    df         = data_test.copy()
    df['LGD']  = lgd_hipotecario(
                     ead            = df['MORTDUE'].values,
                     property_value = df['VALUE'].values
                 )
    df['EL_pct']           = df['default_proba'] * df['LGD']
    df['EL_amount']        = df['default_proba'] * df['LGD'] * df['MORTDUE']
    df['model_prediction'] = (df['default_proba'] >= best_thr).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════
# LAYER 8 — PORTFOLIO OUTPUTS
# ══════════════════════════════════════════════════════════════════

def portfolio_summary(
    data_test  : pd.DataFrame,
    annual_rate: float = 11.5040,
    months     : int   = 240
) -> None:
    """
    Print portfolio-level EL, actual loss, and opportunity cost.

    Actual loss    = EAD of loans where model predicted 0 but true label is 1
                     (missed defaults — credit risk materialises).
    Opportunity cost = foregone total payments on loans where model predicted 1
                       but true label is 0 (incorrectly rejected good clients).

    Args:
        data_test   (pd.DataFrame): Output of compute_el().
        annual_rate (float)       : Mortgage rate for amortization (%).
        months      (int)         : Loan term in months.
    """
    portfolio    = _col(data_test, 'MORTDUE').sum()
    el           = _col(data_test, 'EL_amount').sum()

    actual_loss  = data_test.loc[
        (data_test['model_prediction'] == 0) & (data_test['BAD'] == 1), 'MORTDUE'
    ].sum()

    rejected_loans = data_test.loc[
        (data_test['model_prediction'] == 1) & (data_test['BAD'] == 0), 'LOAN'
    ]
    opp_cost = np.sum(amortization(rejected_loans, annual_rate, months))

    print(f"Total Portfolio EAD:            ${portfolio:,.2f}")
    print(f"Expected Loss:                  ${el:,.2f}  ({el / portfolio:.2%})")
    print(f"Actual Loss (missed defaults):  ${actual_loss:,.2f}  ({actual_loss / portfolio:.2%})")
    print(f"Opportunity Cost:               ${opp_cost:,.2f}  ({opp_cost / portfolio:.2%})")


def risk_bucket_table(data_test: pd.DataFrame) -> pd.DataFrame:
    """
    Segment the portfolio into four PD risk buckets and summarise EL metrics.

    Buckets:
        Bajo     : PD < 5%
        Medio    : 5% ≤ PD < 15%
        Alto     : 15% ≤ PD < 30%
        Muy Alto : PD ≥ 30%

    Args:
        data_test (pd.DataFrame): Output of compute_el().

    Returns:
        pd.DataFrame: Summary table with n_creditos, EAD_total, EL_total,
                      PD_promedio, LGD_promedio, and EL_pct per bucket.
    """
    df        = data_test.copy()
    bins      = [0, 0.05, 0.15, 0.30, 1.0]
    labels    = ['Bajo', 'Medio', 'Alto', 'Muy Alto']
    df['risk_bucket'] = pd.cut(df['default_proba'], bins=bins, labels=labels)

    return df.groupby('risk_bucket', observed=True).agg(
        n_creditos   = ('EL_amount',     'count'),
        EAD_total    = ('MORTDUE',        'sum'),
        EL_total     = ('EL_amount',      'sum'),
        PD_promedio  = ('default_proba',  'mean'),
        LGD_promedio = ('LGD',            'mean'),
    ).assign(EL_pct=lambda x: x['EL_total'] / x['EAD_total'])


def model_validation(y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Print classification report and plot confusion matrix.

    Args:
        y_test (pd.Series): True binary labels.
        y_pred (pd.Series): Model binary predictions.
    """
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['No Default', 'Default'],
        cmap='Blues'
    )
    plt.title('Confusion Matrix (Test)')
    plt.tight_layout()
    plt.show()


def shap_analysis(model: XGBClassifier, X_test: pd.DataFrame, max_display: int = 15) -> None:
    """
    Generate SHAP feature importance plots for model interpretability.

    Produces two plots:
        1. Bar chart: mean absolute SHAP value per feature (global importance).
        2. Beeswarm: SHAP value distribution showing direction of impact.

    Args:
        model       (XGBClassifier): Fitted XGBoost model.
        X_test      (pd.DataFrame) : Test features.
        max_display (int)          : Number of top features to display.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, plot_type='bar',
                      max_display=max_display,
                      title='SHAP Feature Importance — Modelo PD Hipotecario')

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title('SHAP Summary (Impact & Direction) — Modelo PD', fontsize=14)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════
# LAYER 9 — MAIN PIPELINE
# Composes all layers end-to-end. Each notebook cell becomes
# one function call.
# ══════════════════════════════════════════════════════════════════

def pipeline(
    dataset_slug: str  = 'lchipham/home-equity-loan-default',
    filename    : str  = 'hmeq.csv',
    target      : str  = 'BAD',
    haircut     : float = 0.25,
    annual_rate : float = 11.5040,
    months      : int   = 240
) -> dict:
    """
    Full mortgage credit risk pipeline.

    Composes all layers:
        load_data → encode_categoricals → feature_engineering
        → prepare_model_inputs → train_xgboost → curva_roc
        → compute_el → portfolio_summary → risk_bucket_table
        → model_validation → shap_analysis

    Args:
        dataset_slug (str)  : Kaggle dataset identifier.
        filename     (str)  : CSV filename.
        target       (str)  : Binary target column name.
        haircut      (float): Foreclosure cost fraction for LGD.
        annual_rate  (float): Mortgage interest rate (%).
        months       (int)  : Loan term in months.

    Returns:
        dict: {
            'model'    : fitted XGBClassifier,
            'data_test': DataFrame with all EL columns,
            'threshold': optimal classification threshold,
            'resumen'  : risk bucket summary table
        }
    """
    # ── Ingestion & preprocessing ──────────────────────────────
    data = load_data(dataset_slug, filename)
    data = encode_categoricals(data)
    data = feature_engineering(data)

    # ── Model inputs ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = prepare_model_inputs(
        df        = data,
        target    = target,
        drop_cols = ['HOME_EQUITY', 'TOTAL_DEBT']
    )

    # ── Training & threshold selection ─────────────────────────
    model    = train_xgboost(X_train, y_train, X_test, y_test)
    y_proba  = model.predict_proba(X_test)[:, 1]
    best_thr = curva_roc(y_proba, y_test)

    # ── EL computation ─────────────────────────────────────────
    data_test                  = pd.concat([X_test, y_test], axis=1)
    data_test['default_proba'] = y_proba
    data_test                  = compute_el(data_test, best_thr)

    # ── Outputs ────────────────────────────────────────────────
    portfolio_summary(data_test, annual_rate, months)
    resumen = risk_bucket_table(data_test)
    model_validation(y_test, data_test['model_prediction'])
    shap_analysis(model, X_test)

    return {
        'model'    : model,
        'data_test': data_test,
        'threshold': best_thr,
        'resumen'  : resumen
    }


if __name__ == '__main__':
    results = pipeline()