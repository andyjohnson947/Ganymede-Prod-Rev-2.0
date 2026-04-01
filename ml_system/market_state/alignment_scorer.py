#!/usr/bin/env python3
"""
Alignment Scoring Engine (Phase 5)

Computes a continuous alignment score (0.0 to 1.0) for a proposed trade
based on how well the current market state matches the cluster's
dominant strategy profile.

Replaces binary go/no-go with a weighted, nuanced scoring system.

Usage (standalone test):
    python alignment_scorer.py --symbol EURUSD
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Path setup
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / 'trading_bot'))

from ml_system.market_state.config import (
    SYMBOLS, MODEL_DIR, ALIGNMENT_WEIGHTS, ALIGNMENT_THRESHOLD,
)
from ml_system.market_state.feature_buckets import FeatureBucketizer
from ml_system.market_state.environment_clustering import EnvironmentClusterer
from ml_system.market_state.strategy_dominance import StrategyDominanceModel


class AlignmentScorer:
    """
    Score how well current conditions align with the dominant strategy profile.

    Sub-scores (weighted):
        regime    (40%) - Does the cluster favor the proposed strategy?
        structure (25%) - HTF confluence strength
        momentum  (20%) - Momentum alignment with strategy type
        location  (15%) - VWAP band, swing proximity, VP position

    Final score: 0.0 to 1.0 with configurable threshold.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bucketizer = FeatureBucketizer()
        self.clusterer = EnvironmentClusterer()
        self.dominance_model = StrategyDominanceModel()
        self.weights = ALIGNMENT_WEIGHTS
        self.threshold = ALIGNMENT_THRESHOLD
        self._loaded = False
        # Snapshot cache: recommendation + alignment veto share same snapshot
        self._last_snapshot = None
        self._last_snapshot_bar_index = -1

    def load_models(self) -> bool:
        """
        Load cluster model + dominance table for this symbol.

        Returns:
            True if all models loaded successfully, False otherwise.
        """
        try:
            if not self.clusterer.load_model(self.symbol):
                print(f"  [WARN] No cluster model for {self.symbol}")
                return False

            if self.dominance_model.load_dominance_table(self.symbol) is None:
                print(f"  [WARN] No dominance table for {self.symbol}")
                return False

            self._loaded = True
            return True

        except Exception as e:
            print(f"  [ERROR] Loading models for {self.symbol}: {e}")
            return False

    def score_current_state(
        self,
        snapshot_dict: Dict,
        proposed_strategy: str,
        proposed_direction: str,
    ) -> Dict:
        """
        Score a single market state snapshot against the proposed trade.

        Args:
            snapshot_dict: Raw snapshot dict (or dict with same fields)
            proposed_strategy: 'MR' or 'BO'
            proposed_direction: 'buy' or 'sell'

        Returns:
            Dict with alignment_score, sub_scores, should_trade, cluster_id, etc.
        """
        if not self._loaded:
            return self._empty_result("Models not loaded")

        try:
            # 1. Bucketize the snapshot
            bucketed = self.bucketizer.bucketize_snapshot(snapshot_dict)
            bucketed_df = pd.DataFrame([bucketed])

            # 2. One-hot encode
            encoded = self.bucketizer.one_hot_encode(bucketed_df)

            # 3. Predict cluster
            cluster_id = int(self.clusterer.predict(encoded)[0])

            # 4. Get dominance info for this cluster
            cluster_info = self.dominance_model.get_cluster_strategy(cluster_id)

            # 5. Compute sub-scores
            regime_score = self._score_regime(
                cluster_info, proposed_strategy, proposed_direction
            )
            structure_score = self._score_structure(snapshot_dict)
            momentum_score = self._score_momentum(
                snapshot_dict, proposed_strategy, proposed_direction
            )
            location_score = self._score_location(
                snapshot_dict, proposed_strategy, proposed_direction
            )

            # 6. Weighted combination
            alignment_score = (
                self.weights['regime'] * regime_score +
                self.weights['structure'] * structure_score +
                self.weights['momentum'] * momentum_score +
                self.weights['location'] * location_score
            )

            # Clamp to [0, 1]
            alignment_score = max(0.0, min(1.0, alignment_score))

            should_trade = alignment_score >= self.threshold

            return {
                'alignment_score': round(alignment_score, 4),
                'should_trade': should_trade,
                'threshold': self.threshold,
                'cluster_id': cluster_id,
                'cluster_strategy': cluster_info['strategy'],
                'cluster_direction': cluster_info['direction'],
                'cluster_confidence': cluster_info['confidence'],
                'proposed_strategy': proposed_strategy,
                'proposed_direction': proposed_direction,
                'sub_scores': {
                    'regime': round(regime_score, 4),
                    'structure': round(structure_score, 4),
                    'momentum': round(momentum_score, 4),
                    'location': round(location_score, 4),
                },
            }

        except Exception as e:
            return self._empty_result(f"Scoring error: {e}")

    def _build_live_snapshot(
        self,
        h1_data: pd.DataFrame,
        d1_data: pd.DataFrame,
        w1_data: pd.DataFrame,
    ):
        """
        Build a market state snapshot from live OHLC data.
        Uses a cache so recommendation + alignment veto share the same snapshot.

        Returns:
            Snapshot object or None on failure.
        """
        bar_index = len(h1_data) - 1

        # Check cache
        if (self._last_snapshot is not None
                and self._last_snapshot_bar_index == bar_index):
            return self._last_snapshot

        from ml_system.market_state.bar_replay_engine import BarReplayEngine
        engine = BarReplayEngine(include_smc=True)

        from indicators.adx import calculate_adx
        from indicators.technical import calculate_rsi, calculate_macd

        h1_with_adx = calculate_adx(h1_data.copy(), period=14)
        h1_with_vwap = engine.vwap.calculate(h1_data.copy())
        rsi_series = calculate_rsi(h1_data.copy(), period=14)
        macd_line, macd_signal, macd_histogram = calculate_macd(h1_data.copy())

        high_low = h1_data['high'] - h1_data['low']
        high_close = np.abs(h1_data['high'] - h1_data['close'].shift())
        low_close = np.abs(h1_data['low'] - h1_data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=14).mean()

        h1_with_time = h1_data.copy()
        if 'time' not in h1_with_time.columns:
            h1_with_time['time'] = h1_with_time.index

        snapshot = engine._process_bar(
            symbol=self.symbol,
            bar_index=bar_index,
            h1_data=h1_data,
            h1_with_adx=h1_with_adx,
            h1_with_vwap=h1_with_vwap,
            h1_with_time=h1_with_time,
            rsi_series=rsi_series,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            atr_series=atr_series,
            d1_data=d1_data,
            w1_data=w1_data,
        )

        # Cache the snapshot
        if snapshot is not None:
            self._last_snapshot = snapshot
            self._last_snapshot_bar_index = bar_index

        return snapshot

    def score_from_live_data(
        self,
        h1_data: pd.DataFrame,
        d1_data: pd.DataFrame,
        w1_data: pd.DataFrame,
        proposed_strategy: str,
        proposed_direction: str,
    ) -> Dict:
        """
        Score using raw OHLC data (builds snapshot on-the-fly).
        This is the interface for live bot integration.

        Args:
            h1_data: H1 DataFrame with at least 200 bars
            d1_data: D1 DataFrame
            w1_data: W1 DataFrame
            proposed_strategy: 'MR' or 'BO'
            proposed_direction: 'buy' or 'sell'

        Returns:
            Alignment result dict
        """
        if not self._loaded:
            return self._empty_result("Models not loaded")

        try:
            snapshot = self._build_live_snapshot(h1_data, d1_data, w1_data)
            if snapshot is None:
                return self._empty_result("Failed to build snapshot")

            return self.score_current_state(
                snapshot.to_dict(), proposed_strategy, proposed_direction
            )

        except Exception as e:
            return self._empty_result(f"Live scoring error: {e}")

    def get_strategy_recommendation(
        self,
        h1_data: pd.DataFrame,
        d1_data: pd.DataFrame,
        w1_data: pd.DataFrame,
    ) -> Dict:
        """
        PRE-SIGNAL: Query market state for strategy recommendation.

        Returns what strategy the current cluster favors WITHOUT requiring
        a proposed_strategy. This drives strategy SELECTION, not just veto.

        Returns:
            Dict with strategy_priority list, cluster info, skip flags.
        """
        if not self._loaded:
            return self._default_recommendation("Models not loaded")

        try:
            snapshot = self._build_live_snapshot(h1_data, d1_data, w1_data)
            if snapshot is None:
                return self._default_recommendation("Failed to build snapshot")

            # Bucketize -> predict cluster
            bucketed = self.bucketizer.bucketize_snapshot(snapshot.to_dict())
            bucketed_df = pd.DataFrame([bucketed])
            encoded = self.bucketizer.one_hot_encode(bucketed_df)
            cluster_id = int(self.clusterer.predict(encoded)[0])

            # Get enriched cluster info with confidence threshold
            cluster_info = self.dominance_model.get_cluster_strategy(cluster_id)

            strategy = cluster_info['strategy']
            confidence = cluster_info['confidence']
            mr_wr = cluster_info['mr_best_wr']
            bo_wr = cluster_info['bo_best_wr']

            # Determine strategy priority and skip flags
            strategy_priority, skip_mr, skip_bo = self._compute_priority(
                strategy, confidence, mr_wr, bo_wr
            )

            return {
                'strategy_priority': strategy_priority,
                'cluster_id': cluster_id,
                'cluster_strategy': strategy,
                'cluster_direction': cluster_info['direction'],
                'confidence': confidence,
                'mr_expectancy': cluster_info['mr_expectancy'],
                'bo_expectancy': cluster_info['bo_expectancy'],
                'mr_best_wr': mr_wr,
                'bo_best_wr': bo_wr,
                'skip_bo': skip_bo,
                'skip_mr': skip_mr,
                'raw_dominant': cluster_info.get('raw_dominant', strategy),
            }

        except Exception as e:
            return self._default_recommendation(f"Recommendation error: {e}")

    def _compute_priority(
        self,
        strategy: str,
        confidence: float,
        mr_wr: float,
        bo_wr: float,
    ) -> tuple:
        """
        Compute strategy_priority list and skip flags.

        Logic:
          - SKIP: empty priority (don't trade)
          - MR (high conf): ['MR'] only, skip_bo=True
          - BO (high conf): ['BO'] only, skip_mr=True
          - BOTH (low conf): order by win rate, higher WR first
          - MR/BO with moderate conf: prioritize but allow fallback

        Returns:
            (strategy_priority, skip_mr, skip_bo)
        """
        from ml_system.market_state.config import HIGH_DOMINANCE_CONFIDENCE

        if strategy == 'SKIP':
            return ([], False, False)

        if strategy == 'MR':
            if confidence >= HIGH_DOMINANCE_CONFIDENCE:
                return (['MR'], False, True)   # MR only, block BO
            else:
                return (['MR', 'BO'], False, False)  # MR first, BO fallback

        if strategy == 'BO':
            if confidence >= HIGH_DOMINANCE_CONFIDENCE:
                return (['BO'], True, False)   # BO only, block MR
            else:
                return (['BO', 'MR'], False, False)  # BO first, MR fallback

        if strategy == 'BOTH':
            # Low confidence -- order by win rate (higher WR first)
            if mr_wr >= bo_wr:
                return (['MR', 'BO'], False, False)
            else:
                return (['BO', 'MR'], False, False)

        return (['MR', 'BO'], False, False)  # Safe default

    def _default_recommendation(self, reason: str) -> Dict:
        """Return safe fallback recommendation (preserves current MR-first behavior)."""
        return {
            'strategy_priority': ['MR', 'BO'],
            'cluster_id': -1,
            'cluster_strategy': 'UNKNOWN',
            'cluster_direction': 'none',
            'confidence': 0,
            'mr_expectancy': 0,
            'bo_expectancy': 0,
            'mr_best_wr': 0,
            'bo_best_wr': 0,
            'skip_bo': False,
            'skip_mr': False,
            'error': reason,
        }

    # ═══════════════════════════════════════════════════════════════
    # Sub-score computation
    # ═══════════════════════════════════════════════════════════════

    def _score_regime(
        self, cluster_info: Dict, proposed_strategy: str, proposed_direction: str
    ) -> float:
        """
        Score: Does the cluster favor the proposed strategy?

        1.0 = Cluster's dominant strategy matches proposed + direction matches
        0.7 = Strategy matches but direction is 'both' or different
        0.5 = Cluster says SKIP (neutral)
        0.2 = Cluster favors opposite strategy
        0.0 = Cluster says SKIP with negative expectancy
        """
        cluster_strat = cluster_info.get('strategy', 'SKIP')
        cluster_dir = cluster_info.get('direction', 'none')
        confidence = cluster_info.get('confidence', 0)

        if cluster_strat == 'SKIP':
            return 0.3  # Neutral -- not actively opposing

        if cluster_strat == 'BOTH':
            # Low confidence -- both strategies viable
            # Score based on direction match only (strategy is neutral)
            if cluster_dir == proposed_direction or cluster_dir == 'both':
                return 0.7  # Supportive direction, no strategy preference
            else:
                return 0.5  # Neutral on strategy, direction mismatch

        if cluster_strat == proposed_strategy:
            # Strategy matches!
            if cluster_dir == proposed_direction or cluster_dir == 'both':
                return 1.0
            else:
                return 0.7  # Strategy matches but direction differs
        else:
            # Opposite strategy dominant
            if confidence > 5:
                return 0.1  # Strong opposition
            else:
                return 0.3  # Weak opposition -- still might work

    def _score_structure(self, snapshot: Dict) -> float:
        """
        Score: HTF confluence strength.

        Higher HTF score = better structural alignment.
        Normalized: score / 10 capped at 1.0.
        """
        htf_score = snapshot.get('htf_confluence_score', 0) or 0
        return min(1.0, htf_score / 10.0)

    def _score_momentum(
        self, snapshot: Dict, proposed_strategy: str, proposed_direction: str
    ) -> float:
        """
        Score: Momentum alignment with strategy type.

        MR strategy: Momentum AGAINST proposed direction = GOOD (mean reversion setup)
        BO strategy: Momentum WITH proposed direction = GOOD (trend continuation)
        """
        momentum = snapshot.get('momentum_20_pct', 0) or 0
        candle_alignment = snapshot.get('candle_alignment', 'mixed') or 'mixed'
        macd_hist = snapshot.get('macd_histogram', 0) or 0
        rsi = snapshot.get('rsi_14', 50) or 50

        # Is momentum bullish or bearish?
        mom_bullish = momentum > 0.05 or macd_hist > 0
        mom_bearish = momentum < -0.05 or macd_hist < 0

        # Compute alignment
        if proposed_strategy == 'MR':
            # Mean Reversion: momentum AGAINST direction = good setup
            if proposed_direction == 'buy':
                # Want bearish momentum (oversold, selling exhaustion)
                score = 0.5
                if mom_bearish:
                    score += 0.3
                if rsi < 35:
                    score += 0.2
                if 'bearish' in candle_alignment:
                    score += 0.1  # Selling into support = MR buy
                # Fading signal
                if snapshot.get('momentum_sell_is_fading'):
                    score += 0.15
                return min(1.0, score)
            else:
                # MR sell: want bullish momentum (overbought)
                score = 0.5
                if mom_bullish:
                    score += 0.3
                if rsi > 65:
                    score += 0.2
                if 'bullish' in candle_alignment:
                    score += 0.1
                if snapshot.get('momentum_buy_is_fading'):
                    score += 0.15
                return min(1.0, score)

        elif proposed_strategy == 'BO':
            # Breakout: momentum WITH direction = good
            if proposed_direction == 'buy':
                score = 0.5
                if mom_bullish:
                    score += 0.3
                if 'bullish' in candle_alignment:
                    score += 0.2
                if rsi > 55:
                    score += 0.1
                return min(1.0, score)
            else:
                score = 0.5
                if mom_bearish:
                    score += 0.3
                if 'bearish' in candle_alignment:
                    score += 0.2
                if rsi < 45:
                    score += 0.1
                return min(1.0, score)

        return 0.5  # Default neutral

    def _score_location(
        self, snapshot: Dict, proposed_strategy: str, proposed_direction: str
    ) -> float:
        """
        Score: VWAP band, swing proximity, VP position favorability.

        MR buy: Better at lower VWAP bands, swing lows, below VP value area
        MR sell: Better at upper VWAP bands, swing highs, above VP value area
        BO buy: Better above VWAP, breaking swing highs
        BO sell: Better below VWAP, breaking swing lows
        """
        vwap_dir = snapshot.get('vwap_direction', '') or ''
        vwap_dist = abs(snapshot.get('vwap_distance_pct', 0) or 0)
        at_swing_high = snapshot.get('vp_at_swing_high', False)
        at_swing_low = snapshot.get('vp_at_swing_low', False)
        below_val = snapshot.get('vp_below_val', False)
        above_vah = snapshot.get('vp_above_vah', False)
        at_poc = snapshot.get('vp_at_poc', False)
        in_band_1 = snapshot.get('vwap_in_band_1', False)
        in_band_2 = snapshot.get('vwap_in_band_2', False)
        in_band_3 = snapshot.get('vwap_in_band_3', False)

        score = 0.5  # Baseline

        if proposed_strategy == 'MR':
            if proposed_direction == 'buy':
                # MR buy: want price LOW (stretched below VWAP)
                if vwap_dir == 'below':
                    score += 0.15
                if vwap_dist > 0.2:
                    score += 0.1  # Extended
                if at_swing_low:
                    score += 0.15
                if below_val:
                    score += 0.1
                if snapshot.get('near_bullish_fvg'):
                    score += 0.1
            else:
                # MR sell: want price HIGH
                if vwap_dir == 'above':
                    score += 0.15
                if vwap_dist > 0.2:
                    score += 0.1
                if at_swing_high:
                    score += 0.15
                if above_vah:
                    score += 0.1
                if snapshot.get('near_bearish_fvg'):
                    score += 0.1

        elif proposed_strategy == 'BO':
            if proposed_direction == 'buy':
                # BO buy: price already above VWAP, breaking higher
                if vwap_dir == 'above':
                    score += 0.15
                if above_vah:
                    score += 0.15  # Breaking above value area
                if at_swing_high:
                    score += 0.1  # Near resistance (potential breakout)
            else:
                # BO sell: price below VWAP, breaking lower
                if vwap_dir == 'below':
                    score += 0.15
                if below_val:
                    score += 0.15
                if at_swing_low:
                    score += 0.1

        return min(1.0, score)

    def _empty_result(self, reason: str) -> Dict:
        """Return a safe empty result when scoring fails."""
        return {
            'alignment_score': 0.0,
            'should_trade': False,
            'threshold': self.threshold,
            'cluster_id': -1,
            'cluster_strategy': 'UNKNOWN',
            'cluster_direction': 'none',
            'cluster_confidence': 0,
            'proposed_strategy': '',
            'proposed_direction': '',
            'sub_scores': {
                'regime': 0, 'structure': 0, 'momentum': 0, 'location': 0,
            },
            'error': reason,
        }


def main():
    """CLI test: score the most recent snapshot for each symbol."""
    import argparse
    from ml_system.market_state.bar_replay_engine import load_snapshot_data

    parser = argparse.ArgumentParser(description='Alignment scorer test')
    parser.add_argument('--symbol', type=str, help='Test specific symbol')
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  ALIGNMENT SCORING TEST: {symbol}")
        print(f"{'='*60}")

        scorer = AlignmentScorer(symbol)
        if not scorer.load_models():
            print(f"  [SKIP] No models for {symbol}")
            continue

        snapshots = load_snapshot_data(symbol)
        if not snapshots:
            print(f"  [SKIP] No snapshots for {symbol}")
            continue

        # Score last 10 snapshots with different strategies
        print(f"\n  Last 10 snapshots scored:")
        print(f"  {'Bar':>6} | {'Strat':>4} | {'Dir':>4} | {'Score':>6} | {'Trade':>5} | "
              f"{'Regime':>6} | {'Struct':>6} | {'Mom':>6} | {'Loc':>6} | Cluster")
        print(f"  {'-'*85}")

        for snap in snapshots[-10:]:
            for strat in ['MR', 'BO']:
                for direction in ['buy', 'sell']:
                    result = scorer.score_current_state(snap, strat, direction)
                    ss = result['sub_scores']
                    print(f"  {snap['bar_index']:6d} | {strat:>4} | {direction:>4} | "
                          f"{result['alignment_score']:>5.3f} | "
                          f"{'YES' if result['should_trade'] else 'NO':>5} | "
                          f"{ss['regime']:>5.3f} | {ss['structure']:>5.3f} | "
                          f"{ss['momentum']:>5.3f} | {ss['location']:>5.3f} | "
                          f"C{result['cluster_id']}")


if __name__ == '__main__':
    main()
