"""
Tests for DCF Excel export/import round-trip.

Requires openpyxl: pip install openpyxl
"""

import os
import pytest

openpyxl = pytest.importorskip("openpyxl", reason="openpyxl not installed")

from IdleFinance.models.dcf_excel import export_dcf_to_excel, import_dcf_from_excel


COMPANY     = "TestCorp"
PRICE       = 100.0
BASE_REV    = 5000.0
NET_DEBT    = 25000.0
BETA        = 1.2
RF_RATES    = [0.035, 0.04, 0.045]   # last value (0.045) is used for WACC
GROWTH_RATES = [0.08, 0.07, 0.06, 0.05, 0.04]
EBIT_MARGINS = [0.25, 0.25, 0.26, 0.26, 0.27]


@pytest.fixture(scope="module")
def dcf_file(tmp_path_factory):
    """Export a DCF workbook once and return its path."""
    path = tmp_path_factory.mktemp("dcf") / "test_dcf.xlsx"
    export_dcf_to_excel(
        path,
        company_name=COMPANY,
        current_price=PRICE,
        base_revenue=BASE_REV,
        revenue_growth_rates=GROWTH_RATES,
        ebit_margins=EBIT_MARGINS,
        risk_free_rate=RF_RATES,
        beta=BETA,
        market_risk_premium=0.06,
        cost_of_debt_pre_tax=0.07,
        tax_rate=0.30,
        shares_outstanding=1000,
        net_debt=NET_DEBT,
    )
    return path


@pytest.fixture(scope="module")
def dcf_params(dcf_file):
    """Import parameters back from the exported workbook."""
    return import_dcf_from_excel(dcf_file)


class TestExport:
    def test_file_is_created(self, dcf_file):
        assert dcf_file.exists()

    def test_file_is_not_empty(self, dcf_file):
        assert dcf_file.stat().st_size > 0

    def test_expected_sheets_exist(self, dcf_file):
        wb = openpyxl.load_workbook(dcf_file)
        assert set(wb.sheetnames) >= {"Dashboard", "WACC", "Valuation", "Sensitivity"}


class TestImportRoundtrip:
    def test_company_name(self, dcf_params):
        assert dcf_params["company_name"] == COMPANY

    def test_current_price(self, dcf_params):
        assert abs(dcf_params["current_price"] - PRICE) < 1e-6

    def test_base_revenue(self, dcf_params):
        assert abs(dcf_params["base_revenue"] - BASE_REV) < 1e-6

    def test_net_debt(self, dcf_params):
        assert abs(dcf_params["net_debt"] - NET_DEBT) < 1e-6

    def test_risk_free_rate_uses_last_value(self, dcf_params):
        # When a list is passed, the last (most forward-looking) rate is used
        assert abs(dcf_params["risk_free_rate"] - RF_RATES[-1]) < 1e-6

    def test_beta(self, dcf_params):
        assert abs(dcf_params["beta"] - BETA) < 1e-6

    def test_forecast_years(self, dcf_params):
        assert dcf_params["years"] == len(GROWTH_RATES)
