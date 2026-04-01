# Implementation Plan: ADX Auto-Tuning + Lot Size + BO Investigation

## 1. LOT SIZE: 0.04 → 0.16

**Analysis on $933 account with 1:500 leverage:**

| Metric | Current 0.04 | Proposed 0.16 |
|--------|-------------|---------------|
| PC1 (25%) | 0.01 lots | 0.04 lots |
| PC2 (25%) | 0.01 lots | 0.04 lots |
| Trail (50%) | 0.02 lots | 0.08 lots |
| Pip value | $0.40/pip | $1.60/pip |
| SL loss EU (25p) | $10 (1.1%) | $40 (4.3%) |
| SL loss GU (35p) | $14 (1.5%) | $56 (6.0%) |
| Margin per trade | ~$10 | ~$40 |

**Risk assessment:**
- Single trade risk: 4-6% of balance — acceptable for aggressive growth
- Worst case (both pairs, full recovery stacks): $539 margin (58%), but ADX hard stops prevent recovery in trending markets so full stacks are rare
- PC splits cleanly: 0.16 × 0.25 = 0.04 (minimum lot on ICMarkets)

**Change:** Single line in `strategy_config.py` — `INITIAL_LOT_SIZE` or equivalent.

---

## 2. BO INVESTIGATION: Why No BO Trades Since Feb 13

**Finding: BO logic is working correctly. No trades because market conditions didn't align.**

Last BO trade: Feb 13 01:04 UTC

What happened after Feb 13:
- Feb 13 16:00-23:00: ADX in range BUT DI gap < 5 (no direction) — BO blocked
- Feb 14-15: Weekend (no trading)
- Feb 16 (all BO hours): ADX 12-17 on both pairs — too LOW (needs 25+)
- Feb 17 (today): ADX 17-27 — EURUSD just reaching 25, GBPUSD still below

The market went from a strong trend (ADX 30+, BO-ready) into a ranging/consolidation phase (ADX 12-22) where breakouts can't form. This is exactly what the ADX 25-40 gate is designed to filter. BO logic is correct — it's just waiting for the next trend to develop.

**No code changes needed for BO.**

---

## 3. PER-INSTRUMENT ADX AUTO-TUNING

### File 1: `trading_bot/config/strategy_config.py`
- Add `SYMBOL_ADX_SETTINGS` dict (after SYMBOL_SMC_SETTINGS)
  - Per-symbol: `mr_adx_threshold`, `mr_max_adx`, `bo_adx_min`, `bo_adx_max`
  - Initial values match current globals (no-op at deployment)
  - DEFAULT fallback key
- Add `get_adx_settings(symbol)` helper (follows get_di_settings pattern)

### File 2: `trading_bot/indicators/adx.py`
- Add `max_adx=40.0` parameter to `should_trade_based_on_trend()`
- Replace hardcoded `if adx_value > 40` with `if adx_value > max_adx`

### File 3: `trading_bot/strategies/signal_detector.py`
- Import `get_adx_settings`
- Fetch `sym_adx_settings = get_adx_settings(symbol)` alongside DI/SMC
- BO routing: use `sym_adx_settings['bo_adx_min/max']` instead of globals
- MR trend filter: pass `mr_adx_threshold` and `mr_max_adx` to `should_trade_based_on_trend()`

### File 4: `ml_system/auto_tuner.py`
- Add `_recommend_adx_settings()` method:
  - Buckets trades by symbol + ADX range (5-point buckets)
  - Finds optimal MR max ADX (highest bucket where WR>50% and P/L>0)
  - Finds optimal MR threshold (where WR drops below 55%)
  - Finds optimal BO range from profitable buckets
  - Safety: min 10 trades/bucket, max ±5 change per tuning run
- Wire into `_calculate_recommendations()` as step 8
- Add `SYMBOL_ADX_SETTINGS` to `_load_current_config()` dict list
- Add regex writer in `_apply_changes()` for SYMBOL_ADX_SETTINGS
