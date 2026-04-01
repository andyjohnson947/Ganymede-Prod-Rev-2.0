"""
FVG Proximity Impact on Breakout (BO) Trade Win Rates
=====================================================
Analyzes whether being near a Fair Value Gap improves BO outcomes
by cross-tabulating FVG proximity with BO 1R/2R hit rates.
"""

import json
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"C:\Users\Administrator\.claude-worktrees\Ganymede-Prod-Rev-2.0\funny-meninsky\ml_system\market_state\data")

def load_snapshots(symbol):
    """Load JSONL snapshots into a DataFrame."""
    records = []
    fpath = DATA_DIR / "snapshots" / f"{symbol}_snapshots.jsonl"
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)

def load_bucketed(symbol):
    """Load bucketed CSV."""
    fpath = DATA_DIR / "bucketed" / f"{symbol}_bucketed.csv"
    return pd.read_csv(fpath)

def derive_fvg_proximity(row):
    """Derive FVG proximity category from raw snapshot booleans."""
    bull = row.get('near_bullish_fvg', False)
    bear = row.get('near_bearish_fvg', False)
    if bull and bear:
        return 'BOTH'
    elif bull:
        return 'BULL_FVG'
    elif bear:
        return 'BEAR_FVG'
    else:
        return 'NONE'

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def crosstab_winrate(df, group_col, outcome_col, label=""):
    """Print win rate by group."""
    grouped = df.groupby(group_col)[outcome_col].agg(['sum', 'count', 'mean'])
    grouped.columns = ['Wins', 'Total', 'WinRate']
    grouped['WinRate'] = grouped['WinRate'] * 100
    print(f"\n  {label} ({outcome_col}):")
    print(f"  {'Category':<15} {'Wins':>6} / {'Total':>6} = {'WinRate':>7}")
    print(f"  {'-'*45}")
    for idx, row in grouped.iterrows():
        print(f"  {str(idx):<15} {int(row['Wins']):>6} / {int(row['Total']):>6} = {row['WinRate']:>6.1f}%")
    return grouped

def main():
    all_snap = []
    all_buck = []

    for symbol in ['EURUSD', 'GBPUSD']:
        print(f"\nLoading {symbol}...")
        snap = load_snapshots(symbol)
        buck = load_bucketed(symbol)
        print(f"  Snapshots: {len(snap)} rows, Bucketed: {len(buck)} rows")
        all_snap.append(snap)
        all_buck.append(buck)

    df_snap = pd.concat(all_snap, ignore_index=True)
    df_buck = pd.concat(all_buck, ignore_index=True)

    print(f"\nCombined snapshots: {len(df_snap)} rows")
    print(f"Combined bucketed:  {len(df_buck)} rows")

    # ---------------------------------------------------------------
    # PART 0: Check bucketed FVG distribution
    # ---------------------------------------------------------------
    print_section("PART 0: FVG Distribution in Bucketed Data")
    print(f"\n  Bucketed fvg_proximity value counts:")
    vc = df_buck['fvg_proximity'].value_counts()
    for val, cnt in vc.items():
        print(f"    {val}: {cnt} ({cnt/len(df_buck)*100:.1f}%)")

    # ---------------------------------------------------------------
    # Derive FVG proximity from raw snapshot booleans
    # ---------------------------------------------------------------
    df_snap['fvg_proximity'] = df_snap.apply(derive_fvg_proximity, axis=1)
    print(f"\n  Snapshot-derived fvg_proximity value counts:")
    vc2 = df_snap['fvg_proximity'].value_counts()
    for val, cnt in vc2.items():
        print(f"    {val}: {cnt} ({cnt/len(df_snap)*100:.1f}%)")

    # Check which outcome columns exist
    outcome_cols = [c for c in df_snap.columns if c.startswith('fwd_')]
    bo_cols = [c for c in outcome_cols if 'bo' in c]
    mr_cols = [c for c in outcome_cols if 'mr' in c]
    print(f"\n  BO outcome columns: {bo_cols}")
    print(f"  MR outcome columns: {mr_cols}")

    # ---------------------------------------------------------------
    # PART 1: FVG proximity vs BO win rates (overall)
    # ---------------------------------------------------------------
    print_section("PART 1: FVG Proximity vs BO Win Rates (All Data)")

    for col in ['fwd_bo_buy_1r', 'fwd_bo_sell_1r', 'fwd_bo_buy_2r', 'fwd_bo_sell_2r']:
        if col in df_snap.columns:
            crosstab_winrate(df_snap, 'fvg_proximity', col, "Overall")

    # ---------------------------------------------------------------
    # PART 2: FVG proximity vs MR win rates (for comparison)
    # ---------------------------------------------------------------
    print_section("PART 2: FVG Proximity vs MR Win Rates (Comparison)")

    for col in ['fwd_mr_buy_1r', 'fwd_mr_sell_1r', 'fwd_mr_buy_2r', 'fwd_mr_sell_2r']:
        if col in df_snap.columns:
            crosstab_winrate(df_snap, 'fvg_proximity', col, "Overall")

    # ---------------------------------------------------------------
    # PART 3: Direction-aligned FVG analysis
    # ---------------------------------------------------------------
    print_section("PART 3: Direction-Aligned FVG Analysis")

    print("\n  Key question: Does BULL_FVG help BO BUY? Does BEAR_FVG help BO SELL?")

    # BO BUY when near different FVG types
    for col in ['fwd_bo_buy_1r', 'fwd_bo_buy_2r']:
        if col in df_snap.columns:
            print(f"\n  --- {col} by FVG type ---")
            for fvg_val in ['NONE', 'BULL_FVG', 'BEAR_FVG', 'BOTH']:
                subset = df_snap[df_snap['fvg_proximity'] == fvg_val]
                if len(subset) > 0:
                    wr = subset[col].mean() * 100
                    n = len(subset)
                    wins = subset[col].sum()
                    print(f"  {fvg_val:<15} {int(wins):>6} / {n:>6} = {wr:>6.1f}%")

    # BO SELL when near different FVG types
    for col in ['fwd_bo_sell_1r', 'fwd_bo_sell_2r']:
        if col in df_snap.columns:
            print(f"\n  --- {col} by FVG type ---")
            for fvg_val in ['NONE', 'BULL_FVG', 'BEAR_FVG', 'BOTH']:
                subset = df_snap[df_snap['fvg_proximity'] == fvg_val]
                if len(subset) > 0:
                    wr = subset[col].mean() * 100
                    n = len(subset)
                    wins = subset[col].sum()
                    print(f"  {fvg_val:<15} {int(wins):>6} / {n:>6} = {wr:>6.1f}%")

    # ---------------------------------------------------------------
    # PART 4: FVG + Momentum Regime Interaction
    # ---------------------------------------------------------------
    print_section("PART 4: FVG + Momentum Regime Interaction")

    # Join momentum_regime from bucketed to snapshots via bar_index + symbol
    # Both have bar_index and symbol
    df_buck_slim = df_buck[['bar_index', 'symbol', 'momentum_regime']].copy()
    df_merged = df_snap.merge(df_buck_slim, on=['bar_index', 'symbol'], how='left', suffixes=('', '_buck'))

    # Use bucketed momentum_regime if available
    if 'momentum_regime' in df_merged.columns:
        print(f"\n  Momentum regime distribution:")
        vc3 = df_merged['momentum_regime'].value_counts()
        for val, cnt in vc3.items():
            print(f"    {val}: {cnt}")

        # Cross-tab: FVG x Momentum -> BO BUY 1R
        print(f"\n  --- BO BUY 1R: FVG x Momentum ---")
        print(f"  {'FVG':<15} {'Momentum':<15} {'Wins':>6} / {'Total':>6} = {'WR':>7}")
        print(f"  {'-'*58}")

        for fvg_val in ['NONE', 'BULL_FVG', 'BEAR_FVG', 'BOTH']:
            for mom_val in sorted(df_merged['momentum_regime'].dropna().unique()):
                subset = df_merged[(df_merged['fvg_proximity'] == fvg_val) &
                                   (df_merged['momentum_regime'] == mom_val)]
                if len(subset) >= 10:  # min sample size
                    wr = subset['fwd_bo_buy_1r'].mean() * 100
                    wins = int(subset['fwd_bo_buy_1r'].sum())
                    print(f"  {fvg_val:<15} {mom_val:<15} {wins:>6} / {len(subset):>6} = {wr:>6.1f}%")

        # Cross-tab: FVG x Momentum -> BO SELL 1R
        print(f"\n  --- BO SELL 1R: FVG x Momentum ---")
        print(f"  {'FVG':<15} {'Momentum':<15} {'Wins':>6} / {'Total':>6} = {'WR':>7}")
        print(f"  {'-'*58}")

        for fvg_val in ['NONE', 'BULL_FVG', 'BEAR_FVG', 'BOTH']:
            for mom_val in sorted(df_merged['momentum_regime'].dropna().unique()):
                subset = df_merged[(df_merged['fvg_proximity'] == fvg_val) &
                                   (df_merged['momentum_regime'] == mom_val)]
                if len(subset) >= 10:  # min sample size
                    wr = subset['fwd_bo_sell_1r'].mean() * 100
                    wins = int(subset['fwd_bo_sell_1r'].sum())
                    print(f"  {fvg_val:<15} {mom_val:<15} {wins:>6} / {len(subset):>6} = {wr:>6.1f}%")

    # ---------------------------------------------------------------
    # PART 5: Aligned vs Misaligned FVG
    # ---------------------------------------------------------------
    print_section("PART 5: Aligned vs Misaligned FVG Signal")

    print("\n  'Aligned' = BULL_FVG for BUY, BEAR_FVG for SELL")
    print("  'Misaligned' = BEAR_FVG for BUY, BULL_FVG for SELL")
    print("  'None' = No FVG nearby")

    # For BO BUY
    for r_label, col in [('1R', 'fwd_bo_buy_1r'), ('2R', 'fwd_bo_buy_2r')]:
        if col not in df_snap.columns:
            continue
        print(f"\n  BO BUY {r_label}:")
        for label, mask in [
            ('Aligned (BULL_FVG)', df_snap['fvg_proximity'] == 'BULL_FVG'),
            ('Misaligned (BEAR_FVG)', df_snap['fvg_proximity'] == 'BEAR_FVG'),
            ('No FVG (NONE)', df_snap['fvg_proximity'] == 'NONE'),
            ('Both FVGs', df_snap['fvg_proximity'] == 'BOTH'),
        ]:
            subset = df_snap[mask]
            if len(subset) > 0:
                wr = subset[col].mean() * 100
                wins = int(subset[col].sum())
                print(f"    {label:<25} {wins:>6} / {len(subset):>6} = {wr:>6.1f}%")

    # For BO SELL
    for r_label, col in [('1R', 'fwd_bo_sell_1r'), ('2R', 'fwd_bo_sell_2r')]:
        if col not in df_snap.columns:
            continue
        print(f"\n  BO SELL {r_label}:")
        for label, mask in [
            ('Aligned (BEAR_FVG)', df_snap['fvg_proximity'] == 'BEAR_FVG'),
            ('Misaligned (BULL_FVG)', df_snap['fvg_proximity'] == 'BULL_FVG'),
            ('No FVG (NONE)', df_snap['fvg_proximity'] == 'NONE'),
            ('Both FVGs', df_snap['fvg_proximity'] == 'BOTH'),
        ]:
            subset = df_snap[mask]
            if len(subset) > 0:
                wr = subset[col].mean() * 100
                wins = int(subset[col].sum())
                print(f"    {label:<25} {wins:>6} / {len(subset):>6} = {wr:>6.1f}%")

    # ---------------------------------------------------------------
    # PART 6: By Symbol Breakdown
    # ---------------------------------------------------------------
    print_section("PART 6: By Symbol Breakdown")

    for symbol in ['EURUSD', 'GBPUSD']:
        sym_df = df_snap[df_snap['symbol'] == symbol]
        print(f"\n  === {symbol} (n={len(sym_df)}) ===")

        for col in ['fwd_bo_buy_1r', 'fwd_bo_sell_1r']:
            if col not in sym_df.columns:
                continue
            print(f"\n  {col}:")
            for fvg_val in ['NONE', 'BULL_FVG', 'BEAR_FVG', 'BOTH']:
                subset = sym_df[sym_df['fvg_proximity'] == fvg_val]
                if len(subset) > 0:
                    wr = subset[col].mean() * 100
                    wins = int(subset[col].sum())
                    print(f"    {fvg_val:<15} {wins:>6} / {len(subset):>6} = {wr:>6.1f}%")

    # ---------------------------------------------------------------
    # PART 7: Statistical significance check (simple chi-squared)
    # ---------------------------------------------------------------
    print_section("PART 7: Significance Check (FVG vs No-FVG)")

    try:
        from scipy.stats import chi2_contingency
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("\n  scipy not available, skipping chi-squared test")

    if has_scipy:
        for col in ['fwd_bo_buy_1r', 'fwd_bo_sell_1r']:
            if col not in df_snap.columns:
                continue
            # FVG present (any) vs NONE
            fvg_present = df_snap[df_snap['fvg_proximity'] != 'NONE']
            fvg_none = df_snap[df_snap['fvg_proximity'] == 'NONE']

            if len(fvg_present) > 0 and len(fvg_none) > 0:
                # Build contingency table
                a = int(fvg_present[col].sum())  # FVG present & win
                b = len(fvg_present) - a          # FVG present & lose
                c = int(fvg_none[col].sum())       # No FVG & win
                d = len(fvg_none) - c              # No FVG & lose

                table = [[a, b], [c, d]]
                chi2, p, dof, expected = chi2_contingency(table)
                print(f"\n  {col}: FVG Present vs NONE")
                print(f"    FVG Present: {a}/{a+b} = {a/(a+b)*100:.1f}%")
                print(f"    No FVG:      {c}/{c+d} = {c/(c+d)*100:.1f}%")
                print(f"    Chi2={chi2:.3f}, p={p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

    # ---------------------------------------------------------------
    # PART 8: Aligned FVG + Favorable Momentum combo
    # ---------------------------------------------------------------
    print_section("PART 8: Best Combo - Aligned FVG + Favorable Momentum")

    if 'momentum_regime' in df_merged.columns:
        # BO BUY: BULL_FVG + STRONG_UP momentum
        bull_up_combos = [
            ('BULL_FVG + STRONG_UP', (df_merged['fvg_proximity'] == 'BULL_FVG') & (df_merged['momentum_regime'] == 'STRONG_UP')),
            ('BULL_FVG + MODERATE_UP', (df_merged['fvg_proximity'] == 'BULL_FVG') & (df_merged['momentum_regime'] == 'MODERATE_UP')),
            ('BULL_FVG + any', df_merged['fvg_proximity'] == 'BULL_FVG'),
            ('NONE + STRONG_UP', (df_merged['fvg_proximity'] == 'NONE') & (df_merged['momentum_regime'] == 'STRONG_UP')),
            ('NONE + any', df_merged['fvg_proximity'] == 'NONE'),
            ('ALL DATA', pd.Series(True, index=df_merged.index)),
        ]

        print(f"\n  BO BUY 1R - Best combos:")
        print(f"  {'Combo':<30} {'Wins':>6} / {'Total':>6} = {'WR':>7}")
        print(f"  {'-'*55}")
        for label, mask in bull_up_combos:
            subset = df_merged[mask]
            if len(subset) >= 5:
                col = 'fwd_bo_buy_1r'
                wr = subset[col].mean() * 100
                wins = int(subset[col].sum())
                print(f"  {label:<30} {wins:>6} / {len(subset):>6} = {wr:>6.1f}%")

        # BO SELL: BEAR_FVG + STRONG_DOWN momentum
        bear_down_combos = [
            ('BEAR_FVG + STRONG_DOWN', (df_merged['fvg_proximity'] == 'BEAR_FVG') & (df_merged['momentum_regime'] == 'STRONG_DOWN')),
            ('BEAR_FVG + MODERATE_DOWN', (df_merged['fvg_proximity'] == 'BEAR_FVG') & (df_merged['momentum_regime'] == 'MODERATE_DOWN')),
            ('BEAR_FVG + any', df_merged['fvg_proximity'] == 'BEAR_FVG'),
            ('NONE + STRONG_DOWN', (df_merged['fvg_proximity'] == 'NONE') & (df_merged['momentum_regime'] == 'STRONG_DOWN')),
            ('NONE + any', df_merged['fvg_proximity'] == 'NONE'),
            ('ALL DATA', pd.Series(True, index=df_merged.index)),
        ]

        print(f"\n  BO SELL 1R - Best combos:")
        print(f"  {'Combo':<30} {'Wins':>6} / {'Total':>6} = {'WR':>7}")
        print(f"  {'-'*55}")
        for label, mask in bear_down_combos:
            subset = df_merged[mask]
            if len(subset) >= 5:
                col = 'fwd_bo_sell_1r'
                wr = subset[col].mean() * 100
                wins = int(subset[col].sum())
                print(f"  {label:<30} {wins:>6} / {len(subset):>6} = {wr:>6.1f}%")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print_section("SUMMARY")

    # Quick summary stats
    none_df = df_snap[df_snap['fvg_proximity'] == 'NONE']
    bull_df = df_snap[df_snap['fvg_proximity'] == 'BULL_FVG']
    bear_df = df_snap[df_snap['fvg_proximity'] == 'BEAR_FVG']

    print(f"\n  Sample sizes: NONE={len(none_df)}, BULL_FVG={len(bull_df)}, BEAR_FVG={len(bear_df)}")

    if len(bull_df) > 0:
        bo_buy_none = none_df['fwd_bo_buy_1r'].mean() * 100
        bo_buy_bull = bull_df['fwd_bo_buy_1r'].mean() * 100
        bo_buy_bear = bear_df['fwd_bo_buy_1r'].mean() * 100 if len(bear_df) > 0 else float('nan')
        print(f"\n  BO BUY 1R baseline (NONE): {bo_buy_none:.1f}%")
        print(f"  BO BUY 1R + BULL_FVG:      {bo_buy_bull:.1f}% (delta: {bo_buy_bull - bo_buy_none:+.1f}pp)")
        if len(bear_df) > 0:
            print(f"  BO BUY 1R + BEAR_FVG:      {bo_buy_bear:.1f}% (delta: {bo_buy_bear - bo_buy_none:+.1f}pp)")

    if len(bear_df) > 0:
        bo_sell_none = none_df['fwd_bo_sell_1r'].mean() * 100
        bo_sell_bear = bear_df['fwd_bo_sell_1r'].mean() * 100
        bo_sell_bull = bull_df['fwd_bo_sell_1r'].mean() * 100 if len(bull_df) > 0 else float('nan')
        print(f"\n  BO SELL 1R baseline (NONE): {bo_sell_none:.1f}%")
        print(f"  BO SELL 1R + BEAR_FVG:     {bo_sell_bear:.1f}% (delta: {bo_sell_bear - bo_sell_none:+.1f}pp)")
        if len(bull_df) > 0:
            print(f"  BO SELL 1R + BULL_FVG:     {bo_sell_bull:.1f}% (delta: {bo_sell_bull - bo_sell_none:+.1f}pp)")

    print(f"\n{'='*70}")
    print(f"  Analysis complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
