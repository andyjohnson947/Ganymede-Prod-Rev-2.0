#!/usr/bin/env python3
"""
Historical Factor Analysis for Per-Symbol Auto-Tuning

Replays through all snapshots per symbol, determines which confluence factors
fired at each bar, checks forward outcomes (MR/BO 1R hit), and produces
per-symbol factor win rates. Results are used to set data-driven weights
in _PER_SYMBOL_HTF_WEIGHTS and _PER_SYMBOL_ENTRY_WEIGHTS.

Usage:
    python -m ml_system.market_state.historical_factor_analysis
    python -m ml_system.market_state.historical_factor_analysis --symbol AUDUSD
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / 'trading_bot'))

from ml_system.market_state.config import SYMBOLS, SNAPSHOT_DIR
from ml_system.market_state.bar_replay_engine import load_snapshot_data


# ── Factor extraction from snapshot fields ──────────────────────────────

# Entry-level factors (from snapshot booleans)
ENTRY_FACTOR_MAP = {
    'vwap_band_1':  lambda s: s.get('vwap_in_band_1', False),
    'vwap_band_2':  lambda s: s.get('vwap_in_band_2', False),
    'poc':          lambda s: s.get('vp_at_poc', False),
    'swing_high':   lambda s: s.get('vp_at_swing_high', False),
    'swing_low':    lambda s: s.get('vp_at_swing_low', False),
    'above_vah':    lambda s: s.get('vp_above_vah', False),
    'below_val':    lambda s: s.get('vp_below_val', False),
    'lvn':          lambda s: s.get('vp_at_lvn', False),
}

# HTF-level factors (from htf_factors list in snapshot)
HTF_FACTORS = [
    'prev_day_vah', 'prev_day_val', 'prev_day_poc', 'prev_day_high', 'prev_day_low',
    'daily_hvn', 'daily_poc',
    'weekly_hvn', 'weekly_poc',
    'prev_week_swing_low', 'prev_week_swing_high', 'prev_week_vwap',
    'bullish_ob', 'bearish_ob',
]

# Liquidity factors
LIQUIDITY_FACTOR_MAP = {
    'eqh': lambda s: s.get('near_eqh', False),
    'eql': lambda s: s.get('near_eql', False),
    'bullish_ob': lambda s: s.get('at_bullish_ob', False),
    'bearish_ob': lambda s: s.get('at_bearish_ob', False),
}

# ADX/DI context
def get_di_context(snap: dict) -> dict:
    plus_di = snap.get('plus_di', 0) or 0
    minus_di = snap.get('minus_di', 0) or 0
    adx = snap.get('adx', 0) or 0
    separation = abs(plus_di - minus_di)
    dominant = 'bullish' if plus_di > minus_di else 'bearish'
    return {
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'separation': separation,
        'dominant': dominant,
    }

def get_smc_context(snap: dict) -> dict:
    return {
        'htf_bias': snap.get('smc_htf_bias', ''),
        'etf_bias': snap.get('smc_etf_bias', ''),
        'alignment': snap.get('smc_alignment', ''),
        'last_bos_type': snap.get('smc_last_bos_type', ''),
        'bos_bars_ago': snap.get('smc_last_bos_bars_ago', 999),
    }


def analyze_symbol(symbol: str, snapshots: List[Dict]) -> Dict:
    """
    Analyze factor effectiveness for a single symbol.

    For each snapshot where factors fired:
    - Record which factors were active
    - Check if MR buy/sell 1R hit (forward outcome)
    - Build win rate per factor

    Also analyze DI separation effectiveness and SMC filter effectiveness.
    """

    # Factor stats: {factor_name: {'wins': N, 'losses': N, 'total': N}}
    entry_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
    htf_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})

    # DI stats: bucket by separation range
    di_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})

    # SMC stats: aligned vs divergent
    smc_stats = {
        'aligned': {'wins': 0, 'losses': 0, 'total': 0},
        'divergent': {'wins': 0, 'losses': 0, 'total': 0},
    }

    # ADX stats: bucket by range
    adx_stats = defaultdict(lambda: {'mr_wins': 0, 'mr_total': 0, 'bo_wins': 0, 'bo_total': 0})

    # Hour stats
    hour_stats = defaultdict(lambda: {'mr_wins': 0, 'mr_total': 0, 'bo_wins': 0, 'bo_total': 0})

    skipped = 0
    processed = 0

    for snap in snapshots:
        # Need forward outcomes
        mr_buy_1r = snap.get('fwd_mr_buy_1r')
        mr_sell_1r = snap.get('fwd_mr_sell_1r')
        bo_buy_1r = snap.get('fwd_bo_buy_1r')
        bo_sell_1r = snap.get('fwd_bo_sell_1r')

        if mr_buy_1r is None and mr_sell_1r is None:
            skipped += 1
            continue

        processed += 1

        # Determine MR direction from VWAP position (preferred)
        # MR logic: below VWAP -> BUY, above VWAP -> SELL
        # Fallback: DI direction when VWAP unavailable (e.g. AUDUSD has no volume)
        vwap_dir = snap.get('vwap_direction', '')
        if vwap_dir == 'below':
            mr_win = bool(mr_buy_1r)
            bo_win = bool(bo_buy_1r)
        elif vwap_dir == 'above':
            mr_win = bool(mr_sell_1r)
            bo_win = bool(bo_sell_1r)
        else:
            # No VWAP — use DI direction as fallback
            plus_di = snap.get('plus_di', 0) or 0
            minus_di = snap.get('minus_di', 0) or 0
            di_gap = plus_di - minus_di
            if di_gap > 1.0:
                # Bullish DI dominance -> check BUY direction
                mr_win = bool(mr_buy_1r)
                bo_win = bool(bo_buy_1r)
            elif di_gap < -1.0:
                # Bearish DI dominance -> check SELL direction
                mr_win = bool(mr_sell_1r)
                bo_win = bool(bo_sell_1r)
            else:
                # DI too close to call — skip this bar entirely
                skipped += 1
                continue

        # ── Entry factors ────────────────────────────────
        for factor_name, check_fn in ENTRY_FACTOR_MAP.items():
            if check_fn(snap):
                entry_stats[factor_name]['total'] += 1
                if mr_win:
                    entry_stats[factor_name]['wins'] += 1
                else:
                    entry_stats[factor_name]['losses'] += 1

        # ── HTF factors ─────────────────────────────────
        htf_factors = snap.get('htf_factors', [])
        if isinstance(htf_factors, list):
            for factor_name in htf_factors:
                # Normalize factor name (snapshot might use different case/format)
                norm = factor_name.lower().replace(' ', '_')
                htf_stats[norm]['total'] += 1
                if mr_win:
                    htf_stats[norm]['wins'] += 1
                else:
                    htf_stats[norm]['losses'] += 1

        # ── DI separation analysis ──────────────────────
        di = get_di_context(snap)
        sep = di['separation']
        if sep < 2:
            bucket = '0-2'
        elif sep < 5:
            bucket = '2-5'
        elif sep < 10:
            bucket = '5-10'
        elif sep < 15:
            bucket = '10-15'
        else:
            bucket = '15+'

        di_stats[bucket]['total'] += 1
        if mr_win:
            di_stats[bucket]['wins'] += 1
        else:
            di_stats[bucket]['losses'] += 1

        # ── SMC alignment analysis ──────────────────────
        smc = get_smc_context(snap)
        if 'aligned' in smc.get('alignment', ''):
            smc_stats['aligned']['total'] += 1
            if mr_win:
                smc_stats['aligned']['wins'] += 1
            else:
                smc_stats['aligned']['losses'] += 1
        elif 'conflict' in smc.get('alignment', '') or 'diverge' in smc.get('alignment', ''):
            smc_stats['divergent']['total'] += 1
            if mr_win:
                smc_stats['divergent']['wins'] += 1
            else:
                smc_stats['divergent']['losses'] += 1

        # ── ADX bucket analysis ─────────────────────────
        adx = snap.get('adx', 0) or 0
        if adx < 15:
            adx_bucket = '0-15'
        elif adx < 20:
            adx_bucket = '15-20'
        elif adx < 25:
            adx_bucket = '20-25'
        elif adx < 30:
            adx_bucket = '25-30'
        elif adx < 35:
            adx_bucket = '30-35'
        elif adx < 40:
            adx_bucket = '35-40'
        else:
            adx_bucket = '40+'

        adx_stats[adx_bucket]['mr_total'] += 1
        adx_stats[adx_bucket]['bo_total'] += 1
        if mr_win:
            adx_stats[adx_bucket]['mr_wins'] += 1
        if bo_win:
            adx_stats[adx_bucket]['bo_wins'] += 1

        # ── Hour analysis ───────────────────────────────
        hour = snap.get('hour_utc', 0)
        hour_stats[hour]['mr_total'] += 1
        hour_stats[hour]['bo_total'] += 1
        if mr_win:
            hour_stats[hour]['mr_wins'] += 1
        if bo_win:
            hour_stats[hour]['bo_wins'] += 1

    return {
        'symbol': symbol,
        'processed': processed,
        'skipped': skipped,
        'entry_factors': dict(entry_stats),
        'htf_factors': dict(htf_stats),
        'di_stats': dict(di_stats),
        'smc_stats': smc_stats,
        'adx_stats': dict(adx_stats),
        'hour_stats': dict(hour_stats),
    }


def recommend_weights(results: Dict) -> Dict:
    """
    Convert factor win rates into recommended weights.

    Weight scale:
    - WR >= 75%: weight 3 (strong)
    - WR >= 65%: weight 2 (moderate)
    - WR >= 55%: weight 1 (weak but positive)
    - WR < 55%:  weight 0 (drop — no edge)

    Minimum 100 samples to change from default.
    """
    MIN_SAMPLES = 100

    htf_weights = {}
    entry_weights = {}

    # HTF factors
    for factor, stats in results.get('htf_factors', {}).items():
        total = stats['total']
        if total < MIN_SAMPLES:
            continue
        wr = stats['wins'] / total * 100 if total else 0
        if wr >= 75:
            htf_weights[factor] = 3
        elif wr >= 65:
            htf_weights[factor] = 2
        elif wr >= 55:
            htf_weights[factor] = 1
        else:
            htf_weights[factor] = 0

    # Entry factors
    for factor, stats in results.get('entry_factors', {}).items():
        total = stats['total']
        if total < MIN_SAMPLES:
            continue
        wr = stats['wins'] / total * 100 if total else 0
        if wr >= 75:
            entry_weights[factor] = 3
        elif wr >= 65:
            entry_weights[factor] = 2
        elif wr >= 55:
            entry_weights[factor] = 1
        else:
            entry_weights[factor] = 0

    # DI separation recommendation
    di_rec = None
    di_stats = results.get('di_stats', {})
    best_di_wr = 0
    for bucket, stats in sorted(di_stats.items()):
        total = stats['total']
        if total < MIN_SAMPLES:
            continue
        wr = stats['wins'] / total * 100 if total else 0
        if wr > best_di_wr:
            best_di_wr = wr
            # Extract lower bound of bucket as recommended min separation
            try:
                di_rec = float(bucket.split('-')[0])
            except:
                di_rec = 5.0

    # ADX sweet spot
    adx_rec = {}
    adx_stats = results.get('adx_stats', {})
    best_mr_wr = 0
    best_mr_bucket = None
    for bucket, stats in sorted(adx_stats.items()):
        total = stats.get('mr_total', 0)
        if total < MIN_SAMPLES:
            continue
        wr = stats['mr_wins'] / total * 100 if total else 0
        if wr > best_mr_wr:
            best_mr_wr = wr
            best_mr_bucket = bucket

    return {
        'htf_weights': htf_weights,
        'entry_weights': entry_weights,
        'di_min_separation': di_rec,
        'best_adx_bucket': best_mr_bucket,
        'best_adx_mr_wr': best_mr_wr,
    }


def print_results(results: Dict, recommendations: Dict):
    """Print formatted analysis results."""
    symbol = results['symbol']

    print(f"\n{'='*70}")
    print(f"  HISTORICAL FACTOR ANALYSIS: {symbol}")
    print(f"  Snapshots: {results['processed']:,} processed, {results['skipped']:,} skipped")
    print(f"{'='*70}")

    # Entry factors
    print(f"\n  --- Entry Factors ---")
    print(f"  {'Factor':<20} | {'Total':>7} | {'Wins':>7} | {'WR':>6} | {'Rec Weight':>10}")
    print(f"  {'-'*62}")
    for factor in sorted(ENTRY_FACTOR_MAP.keys()):
        stats = results['entry_factors'].get(factor, {'wins': 0, 'losses': 0, 'total': 0})
        total = stats['total']
        wr = stats['wins'] / total * 100 if total else 0
        rec = recommendations['entry_weights'].get(factor, '-')
        print(f"  {factor:<20} | {total:>7,} | {stats['wins']:>7,} | {wr:>5.1f}% | {rec:>10}")

    # HTF factors
    print(f"\n  --- HTF Factors ---")
    print(f"  {'Factor':<25} | {'Total':>7} | {'Wins':>7} | {'WR':>6} | {'Rec Weight':>10}")
    print(f"  {'-'*67}")
    for factor in sorted(results['htf_factors'].keys()):
        stats = results['htf_factors'][factor]
        total = stats['total']
        wr = stats['wins'] / total * 100 if total else 0
        rec = recommendations['htf_weights'].get(factor, '-')
        print(f"  {factor:<25} | {total:>7,} | {stats['wins']:>7,} | {wr:>5.1f}% | {rec:>10}")

    # DI separation
    print(f"\n  --- DI Separation Effectiveness ---")
    print(f"  {'Bucket':<10} | {'Total':>7} | {'Wins':>7} | {'WR':>6}")
    print(f"  {'-'*42}")
    for bucket in sorted(results['di_stats'].keys()):
        stats = results['di_stats'][bucket]
        total = stats['total']
        wr = stats['wins'] / total * 100 if total else 0
        print(f"  {bucket:<10} | {total:>7,} | {stats['wins']:>7,} | {wr:>5.1f}%")
    if recommendations.get('di_min_separation') is not None:
        print(f"  -> Recommended min separation: {recommendations['di_min_separation']}")

    # SMC alignment
    print(f"\n  --- SMC Alignment ---")
    for label, stats in results['smc_stats'].items():
        total = stats['total']
        wr = stats['wins'] / total * 100 if total else 0
        print(f"  {label:<12}: {total:>7,} trades, {wr:.1f}% WR")

    # ADX buckets
    print(f"\n  --- ADX Buckets ---")
    print(f"  {'ADX Range':<10} | {'MR Total':>8} | {'MR WR':>6} | {'BO Total':>8} | {'BO WR':>6}")
    print(f"  {'-'*50}")
    for bucket in ['0-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40+']:
        stats = results['adx_stats'].get(bucket, {'mr_wins': 0, 'mr_total': 0, 'bo_wins': 0, 'bo_total': 0})
        mr_t = stats['mr_total']
        mr_wr = stats['mr_wins'] / mr_t * 100 if mr_t else 0
        bo_t = stats['bo_total']
        bo_wr = stats['bo_wins'] / bo_t * 100 if bo_t else 0
        print(f"  {bucket:<10} | {mr_t:>8,} | {mr_wr:>5.1f}% | {bo_t:>8,} | {bo_wr:>5.1f}%")

    # Recommended weights summary
    print(f"\n  --- RECOMMENDED PER-SYMBOL WEIGHTS ---")
    print(f"  HTF: {recommendations['htf_weights']}")
    print(f"  Entry: {recommendations['entry_weights']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Historical factor analysis for per-symbol tuning')
    parser.add_argument('--symbol', type=str, help='Analyze specific symbol only')
    parser.add_argument('--apply', action='store_true', help='Apply recommended weights to strategy_config.py')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    all_results = {}
    all_recommendations = {}

    for symbol in symbols:
        print(f"\nLoading {symbol} snapshots...")
        snapshots = load_snapshot_data(symbol)
        if not snapshots:
            print(f"  [SKIP] No snapshots for {symbol}")
            continue

        print(f"  Analyzing {len(snapshots):,} snapshots...")
        results = analyze_symbol(symbol, snapshots)
        recommendations = recommend_weights(results)

        all_results[symbol] = results
        all_recommendations[symbol] = recommendations

        print_results(results, recommendations)

    # Apply to config if requested
    if args.apply and all_recommendations:
        apply_to_config(all_recommendations)

    # Save results
    output_path = Path(__file__).parent / 'data' / 'models' / 'historical_factor_analysis.json'
    with open(output_path, 'w') as f:
        json.dump({
            'results': {s: {k: v for k, v in r.items() if k != 'hour_stats'}
                       for s, r in all_results.items()},
            'recommendations': all_recommendations,
        }, f, indent=2, default=str)
    print(f"\n[SAVED] {output_path}")

    return all_results, all_recommendations


def apply_to_config(recommendations: Dict):
    """Apply recommended weights to strategy_config.py per-symbol dicts."""
    import re

    config_path = _project_root / 'trading_bot' / 'config' / 'strategy_config.py'

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    changes = []

    for symbol, rec in recommendations.items():
        htf_w = rec.get('htf_weights', {})
        entry_w = rec.get('entry_weights', {})

        if htf_w:
            # Find and update the symbol's entry in _PER_SYMBOL_HTF_WEIGHTS
            pattern = rf"('{symbol}':\s*)\{{[^}}]+\}}"
            # Build new dict string — merge with existing defaults
            # Read current values first
            current_match = re.search(
                rf"_PER_SYMBOL_HTF_WEIGHTS\s*=\s*\{{[^}}]*'{symbol}':\s*(\{{[^}}]+\}})",
                content, re.DOTALL
            )
            if current_match:
                try:
                    current = eval(current_match.group(1))
                except:
                    current = {}

                # Merge: update existing with recommendations
                merged = {**current, **htf_w}
                new_dict_str = str(merged)

                # Replace in _PER_SYMBOL_HTF_WEIGHTS section
                old_section = current_match.group(0)
                new_section = old_section.replace(current_match.group(1), new_dict_str)
                content = content.replace(old_section, new_section)
                changes.append(f"  {symbol} HTF weights: {htf_w}")

        if entry_w:
            current_match = re.search(
                rf"_PER_SYMBOL_ENTRY_WEIGHTS\s*=\s*\{{[^}}]*'{symbol}':\s*(\{{[^}}]+\}})",
                content, re.DOTALL
            )
            if current_match:
                try:
                    current = eval(current_match.group(1))
                except:
                    current = {}

                merged = {**current, **entry_w}
                new_dict_str = str(merged)

                old_section = current_match.group(0)
                new_section = old_section.replace(current_match.group(1), new_dict_str)
                content = content.replace(old_section, new_section)
                changes.append(f"  {symbol} Entry weights: {entry_w}")

        # DI separation
        di_sep = rec.get('di_min_separation')
        if di_sep is not None:
            pattern = rf"('{symbol}':\s*\{{'enabled':\s*True,\s*'min_separation':\s*)[\d.]+"
            new_content = re.sub(pattern, rf"\g<1>{di_sep}", content)
            if new_content != content:
                content = new_content
                changes.append(f"  {symbol} DI min_separation: {di_sep}")

    if content != original:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n[APPLIED] {len(changes)} per-symbol changes to strategy_config.py:")
        for c in changes:
            print(c)
    else:
        print("\n[OK] No changes to apply")


if __name__ == '__main__':
    main()
