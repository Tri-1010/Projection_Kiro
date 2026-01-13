"""
Configuration for Credit Risk Markov-Chain Projection Model.
All column names and parameters are centralized here.
"""

CFG = {
    # Column names (mapped to actual data columns)
    "loan_id": "AGREEMENT_ID",
    "mob": "MOB",
    "bucket": "STATE_MODEL",
    "ead": "PRINCIPLE_OUTSTANDING",
    "orig_date": "DISBURSAL_DATE",
    "cutoff": "CUTOFF_DATE",
    "cohort": "cohort",
    
    # Derived columns
    "from_state": "from_state",
    "to_state": "to_state",
    "segment_key": "segment_key",
}

# Segment columns for hierarchy: FULL -> COARSE -> GLOBAL
SEGMENT_COLS = ["PRODUCT_TYPE"]

# Canonical bucket order (delinquency states) - mapped to actual data values
BUCKETS_CANON = ["DPD0", "DPD1+", "DPD30+", "DPD60+", "DPD90+", "WRITEOFF", "PREPAY"]

# Buckets for DEL metrics
BUCKETS_30P = ["DPD30+", "DPD60+", "DPD90+", "WRITEOFF"]  # 30+ days past due
BUCKETS_60P = ["DPD60+", "DPD90+", "WRITEOFF"]            # 60+ days past due
BUCKETS_90P = ["DPD90+", "WRITEOFF"]                       # 90+ days past due

# Absorbing states (one-hot rows in transition matrix)
ABSORBING_BASE = ["DPD90+", "WRITEOFF", "PREPAY"]

# Model parameters
MAX_MOB = 24
MIN_COUNT = 30
WEIGHT_MODE = "ead"  # "ead" or "count"

# Hierarchical shrinkage prior strengths
PRIOR_STRENGTH_FULL = 50.0
PRIOR_STRENGTH_COARSE = 100.0

# Tail pooling settings
# Tail pooling giúp giảm noise ở MOB cao bằng cách gộp matrices
# Tail pooling helps reduce noise at high MOBs by averaging matrices
TAIL_POOL_ENABLED = False  # True: bật tail pooling, False: tắt
TAIL_POOL_START = 18       # MOB bắt đầu pooling (chỉ áp dụng khi TAIL_POOL_ENABLED=True)

# Calibration settings
CALIBRATION_MODE = "matrix"  # "matrix" or "vector"
K_CLIP = (0.5, 2.0)  # (k_min, k_max) for calibration factors

# Denominator level for DEL computation
DENOM_LEVEL = "cohort_segment"  # "cohort" or "cohort_segment"
