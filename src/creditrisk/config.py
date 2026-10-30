"""Central configuration. Every business assumption lives here so it can be
challenged in one place (and explained in an interview)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# UCI "Default of Credit Card Clients" (Yeh & Lien, 2009). 30,000 Taiwanese
# card holders, April-September 2005. If this URL changes, download the file
# manually from the UCI ML Repository (dataset id 350) into data/.
UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

TARGET = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Protected attributes are excluded from the model (fair-lending practice).
# They are kept in the data only to run a fairness check after scoring.
PROTECTED = ["SEX", "MARRIAGE"]

# --- Scorecard scaling --------------------------------------------------------
# A score of 600 means good:bad odds of 50:1, and every 20 points doubles the odds.
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

# --- WoE binning --------------------------------------------------------------
N_FINE_BINS = 10       # initial quantile bins before merging
MIN_BIN_SHARE = 0.05   # each final bin must hold >= 5% of observations
MIN_IV = 0.02          # features below this Information Value are dropped
MAX_WOE_CORR = 0.70    # of two WoE features more correlated than this, keep the higher-IV one

# --- IFRS 9 / Ind AS 109 assumptions (NOT in the data; documented as judgement) ---
# Staging
STAGE3_MIN_MONTHS_DELAY = 3   # 90+ days past due => credit-impaired
STAGE2_MIN_MONTHS_DELAY = 1   # 30+ days past due backstop => SICR
STAGE2_PD_THRESHOLD = 0.30    # absolute 12m PD trigger (no origination PD available)

# Exposure and loss
CCF = 0.50                    # credit conversion factor on the undrawn limit
LGD = 0.65                    # unsecured cards; dataset has no recovery data
LIFETIME_YEARS = 3            # behavioural life of a revolving card for Stage 2

# Forward-looking scenarios: shift applied to PD in log-odds space.
SCENARIOS = {
    "Upside":   {"weight": 0.20, "logit_shift": -0.25},
    "Base":     {"weight": 0.50, "logit_shift": 0.00},
    "Downside": {"weight": 0.30, "logit_shift": 0.50},
}

# Basel IRB asset correlation for qualifying revolving retail exposures.
BASEL_QRRE_RHO = 0.04
BASEL_CONFIDENCE = 0.999
