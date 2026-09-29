"""12B full-chain no-degradation validation support (Phase 12B-FC V2).

The validation profile is OFF by default.  It is only enabled on a dedicated
validation API instance (SENTRIX_12B_FULL_CHAIN_VALIDATION=1) so normal
production traffic never sees it.  See full_chain_profile.py for the flags and
model_call_ledger.py for the per-call proof ledger.
"""
