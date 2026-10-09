"""Ticker validation: an AI-named company or symbol only counts if it is really listed."""
import pytest

from backend import pipeline

# What Yahoo's search returned for these queries when the validation was written.
SEARCH_RESULTS = {
    "Anthropic": [
        {"symbol": "ANTW", "quoteType": "ETF", "shortname": "Anthropic AI Lab Ecosystem ETF"},
        {"symbol": "ANTHROPIC-USD", "quoteType": "CRYPTOCURRENCY", "shortname": "Anthropic tokenized stock (PreStocks) USD"},
    ],
    "SpaceX": [
        {"symbol": "SPCX", "quoteType": "EQUITY", "shortname": "Space Exploration Technologies", "longname": "Space Exploration Technologies Corp."},
        {"symbol": "LOFF", "quoteType": "EQUITY", "shortname": "Direxion Daily SpaceX Bull 2X E", "longname": "Direxion Daily SpaceX Bull 2X ETF"},
    ],
    "SPCX": [{"symbol": "SPCX", "quoteType": "EQUITY", "longname": "Space Exploration Technologies Corp."}],
    "Oracle": [{"symbol": "ORCL", "quoteType": "EQUITY", "longname": "Oracle Corporation"}],
    "Reliance Industries": [{"symbol": "RELIANCE.NS", "quoteType": "EQUITY", "longname": "Reliance Industries Limited"}],
    "RELIANCE.NS": [{"symbol": "RELIANCE.NS", "quoteType": "EQUITY", "longname": "Reliance Industries Limited"}],
    "GC=F": [{"symbol": "GC=F", "quoteType": "FUTURE", "shortname": "Gold Dec 26"}],
    "KION": [{"symbol": "KGX.DE", "quoteType": "EQUITY", "longname": "KION GROUP AG"}],
    "XYZFAKE": [],
}


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    searched = []

    class FakeSearch:
        def __init__(self, query, timeout=None):
            searched.append(query)
            if query not in SEARCH_RESULTS:
                raise ConnectionError("search unavailable")
            self.quotes = SEARCH_RESULTS[query]

    monkeypatch.setattr(pipeline.yf, "Search", FakeSearch)
    return searched


@pytest.mark.parametrize("query, expected", [
    ("Oracle", "ORCL"),
    ("Reliance Industries", "RELIANCE.NS"),
    ("RELIANCE.NS", "RELIANCE.NS"),
    ("GC=F", "GC=F"),  # exact symbols are accepted whatever their type
    ("SPCX", "SPCX"),
    ("KION", "KGX.DE"),  # a company name, resolved to where it is actually listed
])
def test_real_listings_resolve(query, expected):
    assert pipeline.resolve_ticker(query) == expected


@pytest.mark.parametrize("query", [
    "Anthropic",  # private: the only hits are a thematic ETF and a token
    "SpaceX",  # the name match is a leveraged fund that Yahoo labels as a stock
    "XYZFAKE",
    "",
    None,
])
def test_lookalikes_and_unknowns_are_rejected(query):
    assert pipeline.resolve_ticker(query) is None


def test_results_are_cached(fake_search):
    pipeline.resolve_ticker("Oracle")
    pipeline.resolve_ticker("Oracle")
    assert fake_search == ["Oracle"]


def test_a_failed_search_is_not_cached(fake_search):
    assert pipeline.resolve_ticker("Unknown Co") is None
    assert pipeline.resolve_ticker("Unknown Co") is None
    assert fake_search == ["Unknown Co", "Unknown Co"]  # retried, since the failure may be temporary


def test_valid_tickers_drops_what_cannot_be_verified():
    assert pipeline._valid_tickers(["Oracle", "XYZFAKE", "ORACLE_TYPO", None, "Oracle", "GC=F"]) == ["ORCL", "GC=F"]


def test_resolving_records_the_listed_name():
    pipeline.resolve_ticker("Reliance Industries")
    pipeline.resolve_ticker("Oracle")
    assert pipeline.ticker_names(["RELIANCE.NS", "ORCL", "XYZFAKE"]) == {
        "RELIANCE.NS": "Reliance Industries Limited", "ORCL": "Oracle Corporation",
    }


@pytest.mark.parametrize("symbol, expected", [
    ("INR", ("INR=X", False)),  # Yahoo's INR=X is rupees per dollar: it falls as the rupee strengthens
    ("INR=X", ("INR=X", False)),
    ("USDINR=X", ("INR=X", False)),
    ("INRUSD=X", ("INR=X", False)),
    ("jpy", ("JPY=X", False)),
    ("EUR", ("EURUSD=X", True)),  # quoted as dollars per euro: moves with the euro
    ("GBPUSD=X", ("GBPUSD=X", True)),
    ("USD", ("DX-Y.NYB", True)),
    ("AAPL", None),
    ("GC=F", None),
    ("COP", None),  # ConocoPhillips, not the Colombian peso
    ("IBM", None),
    ("", None),
    (None, None),
])
def test_currency_instrument(symbol, expected):
    assert pipeline.currency_instrument(symbol) == expected


def test_a_currency_call_lands_on_the_side_its_pair_moves():
    # Checked on the live scorecard: forex calls were right 33% of the time at one day, because
    # "rupee strengthens" was published as buy INR=X — a pair that falls when the rupee rises.
    assert pipeline.verified_sides(["INR", "Oracle"], ["EUR", "XYZFAKE"]) == (["ORCL"], ["INR=X", "EURUSD=X"])
    assert pipeline.verified_sides([], ["JPY", "GBP"]) == (["JPY=X"], ["GBPUSD=X"])
    assert pipeline.ticker_names(["INR=X", "EURUSD=X", "DX-Y.NYB"]) == {"INR=X": "USD/INR", "EURUSD=X": "EUR/USD", "DX-Y.NYB": "US Dollar Index"}
