"""The keyword gate, the noise block list, and the index/commodity detectors."""
import pytest

from backend import pipeline

MARKET_MOVING = [
    "CPI rises 0.4% in September, hotter than expected",
    "Powell signals no rush to cut",
    "US Treasuries sell off as jobs report beats forecasts",
    "Nonfarm payrolls jump by 250,000",
    "RBI keeps repo rate unchanged at 6.5%",
    "Rupee hits record low against greenback",
    "Tesla deliveries miss estimates",
    "Apple cuts revenue guidance on weak iPhone demand",
    "Boeing to buy Spirit AeroSystems in $4.7 billion deal",
    "Goldman downgrades Microsoft to neutral",
    "Infosys Q2 net profit rises 5%, beats estimates",
    "SEBI bans five entities from securities market",
    "Intel announces $10 billion buyback",
    "Pfizer recalls blood pressure drug",
    "ECB holds, Lagarde hints at December move",
    "Trading halted in Super Micro after auditor resigns",
    "Amazon faces FTC antitrust lawsuit",
    "OpenAI unveils new AI model",
    "Reliance announces bonus issue, dividend",
    "US Initial Jobless Claims Slip to 197,000, Lowest Since July",
    "Global M&A deal rush fades in third quarter",
    "Adani Ports bags Rs 2,000 crore order",
    "Gold hits record as dollar weakens",
    "OPEC+ agrees surprise output cut",
    "Copper slides on China demand worries",
]

# Each of these contains a keyword only as part of a longer word ("rate" in "corporate"),
# which the old substring check let through.
NOT_WHOLE_WORDS = [
    "Celebrity couple shared photos from their wedding",
    "Corporate retreat ideas for small teams",
    "Software tips to speed up your laptop",
    "Actor wins award at film festival",
    "How to make your response sound more confident",
    "Supermarket shoppers love this new snack",
    "Generally speaking, kids need more sleep",
    "Fabulous fabric trends for fall fashion",
    "Stockholm hosts annual jazz festival",
    "Las Vegas hotel opens new pool",
    "Hernandez scores twice in league win",
]

# Market words are present, but these are commentary, listicles, previews or price lists.
NOISE = [
    "Why Did SanDisk Stock Jump?",
    "How Risky Is Home Depot Stock?",
    "5 Energy ETFs That Have Soared in 2026",
    "Jim Cramer sees a huge catalyst for Apple",
    "The September jobs report will be released Friday. Here's what to expect",
    "Here’s the bond-market alternative as U.S. debt deteriorates",
    "Horizons Middle East & Africa 10/2/2026 (Video)",
    "FTSE 100 Live: UK Markets Show Signs of Calm",
    "Best credit cards for travel in 2026",
    "AM Markets Need to Know: Broadcom preps $60B AI debt",
    "Gold Rate Today in Katni 2nd October 2026 : 22 & 24 Carat",
    "Gold price in Pakistan for today, October 02, 2026",
    "AI Is Great At Science. This Professor Explains Why",
]


@pytest.mark.parametrize("headline", MARKET_MOVING)
def test_market_moving_headlines_pass(headline):
    assert pipeline.is_market_relevant(headline)


@pytest.mark.parametrize("headline", NOT_WHOLE_WORDS)
def test_keywords_only_match_whole_words(headline):
    assert not pipeline.is_market_relevant(headline)


@pytest.mark.parametrize("headline", NOISE)
def test_noise_is_blocked_even_with_market_words(headline):
    assert not pipeline.is_market_relevant(headline)


def test_plurals_match():
    assert pipeline.is_market_relevant("Rates could stay higher for longer")
    assert pipeline.is_market_relevant("Treasuries extend their slide")


def test_watchlist_ticker_counts_as_relevant():
    pipeline.state.watchlist = [{"symbol": "ZZTOP", "label": "ZZTOP"}]
    assert pipeline.is_market_relevant("ZZTOP names a new chief of staff")
    # Case-sensitive: the ticker, not the same letters inside ordinary text.
    assert not pipeline.is_market_relevant("zztop names a new chief of staff")


@pytest.mark.parametrize("headline, expected", [
    ("Gold hits record as dollar weakens", "Gold Spot"),
    ("Brent crude jumps 3%", "Brent Crude Oil"),
    ("Copper slides on China demand worries", "Copper"),
    ("Oil prices tumble after OPEC meeting", "WTI Crude Oil"),
])
def test_commodity_detected(headline, expected):
    assert pipeline.detect_commodity(headline)[2] == expected


@pytest.mark.parametrize("headline", [
    "Goldman Sachs Names Top China Battery Stocks",  # "gold" in Goldman
    "Brentford sold to US investors",  # "brent" in Brentford
    "Tesla cornered by price war",  # "corn" in cornered
])
def test_commodity_not_detected_inside_other_words(headline):
    assert pipeline.detect_commodity(headline) is None


@pytest.mark.parametrize("headline, expected", [
    ("Nifty ends at record high", "Nifty 50"),
    ("Sensex falls 800 points", "Sensex"),
    ("Bank Nifty slips as HDFC drags", "Bank Nifty"),
])
def test_index_detected(headline, expected):
    assert pipeline.detect_index(headline)[1] == expected


@pytest.mark.parametrize("headline", [
    "Bank Indonesia Chief Says FX Defense Unchanged",  # "nse" in Defense
    "Fed officials observe slowing growth",  # "bse" in observe
])
def test_index_not_detected_inside_other_words(headline):
    assert pipeline.detect_index(headline) is None
