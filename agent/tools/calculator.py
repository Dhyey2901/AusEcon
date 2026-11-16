"""Pure-Python mortgage repayment impact calculator.

No external dependencies and no LLM calls, so this module is covered by
`tests/test_calculator.py` and is safe to unit test directly.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger("agent.calculator")


def monthly_repayment(principal: float, annual_rate_pct: float, term_years: int) -> float:
    """Compute the fixed monthly repayment for a standard amortising loan.

    Args:
        principal: The loan amount in dollars.
        annual_rate_pct: The annual interest rate, as a percentage (e.g. 6.0 for 6%).
        term_years: The loan term in years.

    Returns:
        The monthly repayment amount in dollars.

    Raises:
        ValueError: If any input is non-positive, or the rate is negative.
    """
    if principal <= 0:
        raise ValueError(f"principal must be positive, got {principal}")
    if term_years <= 0:
        raise ValueError(f"term_years must be positive, got {term_years}")
    if annual_rate_pct < 0:
        raise ValueError(f"annual_rate_pct cannot be negative, got {annual_rate_pct}")

    n_payments = term_years * 12
    if annual_rate_pct == 0:
        return principal / n_payments

    monthly_rate = (annual_rate_pct / 100) / 12
    factor = (1 + monthly_rate) ** n_payments
    return principal * monthly_rate * factor / (factor - 1)


def calculate_rate_impact_raw(
    loan_amount: float,
    current_rate_pct: float,
    rate_change_pct: float,
    term_years: int = 30,
) -> dict:
    """Compute how a cash-rate change affects a mortgage's monthly repayment.

    Assumes the full rate change is passed through to the mortgage rate
    one-for-one, which is the simplifying assumption typically used when
    illustrating RBA cash rate decisions' impact on borrowers.

    Args:
        loan_amount: Outstanding loan principal in dollars.
        current_rate_pct: The current annual mortgage interest rate, as a
            percentage (e.g. 6.0 for 6%).
        rate_change_pct: The rate change in percentage points (e.g. 0.25 for
            a 25 basis point rise, -0.25 for a cut).
        term_years: Remaining loan term in years. Defaults to 30.

    Returns:
        A dict with the before/after monthly repayment, the dollar and
        percentage change, and the input parameters echoed back.

    Raises:
        ValueError: If inputs are invalid (see `monthly_repayment`), or if
            `rate_change_pct` would drive the rate below zero.
    """
    new_rate_pct = current_rate_pct + rate_change_pct
    if new_rate_pct < 0:
        raise ValueError(
            f"rate_change_pct of {rate_change_pct} would drive the rate below zero "
            f"(current {current_rate_pct}%)"
        )

    before = monthly_repayment(loan_amount, current_rate_pct, term_years)
    after = monthly_repayment(loan_amount, new_rate_pct, term_years)
    dollar_change = after - before
    pct_change = (dollar_change / before) * 100 if before else 0.0

    return {
        "loan_amount": loan_amount,
        "term_years": term_years,
        "current_rate_pct": current_rate_pct,
        "rate_change_pct": rate_change_pct,
        "new_rate_pct": new_rate_pct,
        "monthly_repayment_before": round(before, 2),
        "monthly_repayment_after": round(after, 2),
        "monthly_change_dollars": round(dollar_change, 2),
        "monthly_change_pct": round(pct_change, 2),
    }


@tool
def calculate_rate_impact(
    loan_amount: float,
    current_rate_pct: float,
    rate_change_pct: float,
    term_years: int = 30,
) -> str:
    """Calculate how a change in the cash rate affects a mortgage's monthly repayment.

    Use this when a user asks about the dollar or percentage impact of a rate
    rise/cut on their home loan repayments.

    Args:
        loan_amount: Outstanding loan principal in dollars.
        current_rate_pct: Current annual mortgage interest rate as a percentage (e.g. 6.0).
        rate_change_pct: Rate change in percentage points (e.g. 0.25 for a 25bp rise,
            -0.25 for a cut).
        term_years: Remaining loan term in years (default 30).

    Returns:
        A human-readable summary of the before/after monthly repayment and the change.
    """
    try:
        result = calculate_rate_impact_raw(
            loan_amount, current_rate_pct, rate_change_pct, term_years
        )
    except ValueError as exc:
        logger.warning("calculate_rate_impact failed: %s", exc)
        return f"Could not calculate rate impact: {exc}"

    direction = "increase" if result["monthly_change_dollars"] >= 0 else "decrease"
    return (
        f"For a ${result['loan_amount']:,.0f} loan over {result['term_years']} years, "
        f"a {result['rate_change_pct']:+.2f} percentage point rate change "
        f"(from {result['current_rate_pct']:.2f}% to {result['new_rate_pct']:.2f}%) "
        f"would {direction} the monthly repayment from "
        f"${result['monthly_repayment_before']:,.2f} to ${result['monthly_repayment_after']:,.2f} "
        f"(a change of ${abs(result['monthly_change_dollars']):,.2f}, "
        f"{abs(result['monthly_change_pct']):.2f}%)."
    )
