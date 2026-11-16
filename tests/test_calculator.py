"""Unit tests for the pure-Python mortgage rate-impact calculator."""

from __future__ import annotations

import pytest

from agent.tools.calculator import (
    calculate_rate_impact,
    calculate_rate_impact_raw,
    monthly_repayment,
)


class TestMonthlyRepayment:
    def test_known_amortisation_value(self):
        # $500,000 at 6% over 30 years is a textbook amortisation figure.
        result = monthly_repayment(500_000, 6.0, 30)
        assert result == pytest.approx(2997.75, abs=0.5)

    def test_zero_rate_is_simple_division(self):
        assert monthly_repayment(120_000, 0.0, 10) == pytest.approx(1000.0)

    def test_rejects_non_positive_principal(self):
        with pytest.raises(ValueError):
            monthly_repayment(0, 5.0, 30)

    def test_rejects_non_positive_term(self):
        with pytest.raises(ValueError):
            monthly_repayment(100_000, 5.0, 0)

    def test_rejects_negative_rate(self):
        with pytest.raises(ValueError):
            monthly_repayment(100_000, -1.0, 30)


class TestCalculateRateImpact:
    def test_rate_rise_increases_repayment(self):
        result = calculate_rate_impact_raw(
            loan_amount=700_000, current_rate_pct=6.0, rate_change_pct=0.5, term_years=25
        )
        assert result["monthly_repayment_after"] > result["monthly_repayment_before"]
        assert result["monthly_change_dollars"] > 0
        assert result["new_rate_pct"] == pytest.approx(6.5)

    def test_rate_cut_decreases_repayment(self):
        result = calculate_rate_impact_raw(
            loan_amount=700_000, current_rate_pct=6.0, rate_change_pct=-0.5, term_years=25
        )
        assert result["monthly_repayment_after"] < result["monthly_repayment_before"]
        assert result["monthly_change_dollars"] < 0

    def test_zero_change_is_a_no_op(self):
        result = calculate_rate_impact_raw(
            loan_amount=500_000, current_rate_pct=5.5, rate_change_pct=0.0, term_years=30
        )
        assert result["monthly_change_dollars"] == pytest.approx(0.0, abs=0.01)

    def test_rejects_rate_change_that_drives_rate_negative(self):
        with pytest.raises(ValueError):
            calculate_rate_impact_raw(
                loan_amount=500_000, current_rate_pct=1.0, rate_change_pct=-2.0, term_years=30
            )

    def test_default_term_is_thirty_years(self):
        result = calculate_rate_impact_raw(
            loan_amount=400_000, current_rate_pct=6.0, rate_change_pct=0.25
        )
        assert result["term_years"] == 30


class TestCalculateRateImpactTool:
    def test_tool_returns_readable_string(self):
        output = calculate_rate_impact.invoke(
            {
                "loan_amount": 500_000,
                "current_rate_pct": 6.0,
                "rate_change_pct": 0.25,
                "term_years": 30,
            }
        )
        assert "$" in output
        assert "increase" in output

    def test_tool_handles_invalid_input_gracefully(self):
        output = calculate_rate_impact.invoke(
            {
                "loan_amount": -100,
                "current_rate_pct": 6.0,
                "rate_change_pct": 0.25,
                "term_years": 30,
            }
        )
        assert "Could not calculate" in output
