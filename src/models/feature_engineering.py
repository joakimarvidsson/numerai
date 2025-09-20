"""Advanced feature engineering utilities for Numerai Signals models."""
from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Create row-wise, cross-sectional, and interaction-based features."""

    def __init__(self, config: Dict):
        self.config = config
        self.fe_config = config.get("feature_engineering", {})
        self.model_config = config.get("model", {})

    # ------------------------------------------------------------------
    def create_all_features(self, df: pd.DataFrame, is_live: bool = False) -> pd.DataFrame:
        """Create feature matrix including engineered features."""

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "numerai_ticker" in df.columns:
            df.sort_values(["numerai_ticker", "date"], inplace=True)

        base_features = self._detect_feature_columns(df)
        if not base_features:
            LOGGER.warning("No base feature columns detected; downstream models may fail.")

        df = self._add_price_features(df)
        df = self._add_row_statistics(df, base_features)
        df = self._add_cross_sectional_statistics(df, base_features)

        if self.fe_config.get("target_lags", {}).get("enabled", True) and not is_live:
            df = self._add_target_lags(df)

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fe_config.get("interaction", {}).get("enabled", True):
            return df

        df = df.copy()
        base_features = self._detect_feature_columns(df)
        top_k = int(self.fe_config.get("interaction", {}).get("top_features", 10))
        selected = base_features[:top_k]
        operations = self.fe_config.get("interaction", {}).get(
            "operations", ["product", "ratio"]
        )

        new_cols = {}
        for f1, f2 in itertools.combinations(selected, 2):
            if "product" in operations:
                new_cols[f"{f1}_x_{f2}"] = df[f1] * df[f2]
            if "difference" in operations:
                new_cols[f"{f1}_minus_{f2}"] = df[f1] - df[f2]
            if "ratio" in operations:
                ratio = df[f1] / df[f2]
                ratio = ratio.replace([np.inf, -np.inf], np.nan)
                new_cols[f"{f1}_div_{f2}"] = ratio

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def apply_cross_sectional_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        norm_config = self.fe_config.get("cross_sectional_normalization", {})
        if not norm_config.get("enabled", True):
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        protected_cols = self._protected_columns(df)
        target_columns = self._target_columns(df)
        exclude_cols = set(protected_cols + target_columns)

        clip_value = float(norm_config.get("clip_value", 5.0))

        for col in numeric_cols:
            if col in exclude_cols:
                continue
            df[col] = df.groupby("date")[col].transform(self._zscore_clip, clip=clip_value)

        return df

    # ------------------------------------------------------------------
    def _detect_feature_columns(self, df: pd.DataFrame) -> List[str]:
        prefixes = self.fe_config.get("feature_prefixes")
        if not prefixes:
            prefixes = ["feature_"]

        feature_cols = [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
        feature_cols.sort()
        return feature_cols

    def _target_columns(self, df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if col.startswith("target")]

    def _protected_columns(self, df: pd.DataFrame) -> List[str]:
        protected = [
            "numerai_ticker",
            "ticker",
            "date",
            "data_type",
            "country",
            "sector",
            "industry",
            self.model_config.get("meta_model_column", "meta_model"),
        ]
        return [col for col in protected if col in df.columns]

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fe_config.get("include_price_features", True):
            return df
        price_cols = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        available = [c for c in price_cols if c in df.columns]
        if not available:
            return df

        df = df.copy()
        grouped = df.groupby("numerai_ticker", group_keys=False)

        if {"Close", "Open"}.issubset(available):
            df["gap_return"] = (df["Open"] / grouped["Close"].shift(1)).replace([np.inf, -np.inf], np.nan)
            df["close_return"] = grouped["Close"].pct_change()
            df["log_close_return"] = np.log1p(df["close_return"])

        if {"High", "Low"}.issubset(available):
            df["intraday_range"] = (df["High"] - df["Low"]) / df["Low"].replace(0, np.nan)

        if "Volume" in available:
            df["volume_zscore"] = grouped["Volume"].transform(
                lambda s: (s - s.rolling(63, min_periods=5).mean())
                / s.rolling(63, min_periods=5).std()
            )
            df["volume_log"] = np.log1p(df["Volume"])

        # Rolling features
        windows = [5, 10, 21, 63, 126]
        for window in windows:
            if "Close" in available:
                df[f"close_ma_{window}"] = grouped["Close"].transform(
                    lambda s: s.rolling(window, min_periods=2).mean()
                )
                df[f"close_std_{window}"] = grouped["Close"].transform(
                    lambda s: s.rolling(window, min_periods=2).std()
                )
            if "close_return" in df.columns:
                df[f"momentum_{window}"] = grouped["close_return"].transform(
                    lambda s: s.rolling(window, min_periods=2).sum()
                )

        return df

    def _add_row_statistics(self, df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
        if not feature_cols:
            return df

        stats = self.fe_config.get("row_statistics", [])
        if not stats:
            return df

        feature_values = df[feature_cols]
        def percentile(arr: np.ndarray, q: float) -> np.ndarray:
            return np.nanpercentile(arr, q, axis=1)

        summary = {}
        if "mean" in stats:
            summary["row_mean"] = feature_values.mean(axis=1)
        if "median" in stats:
            summary["row_median"] = feature_values.median(axis=1)
        if "std" in stats:
            summary["row_std"] = feature_values.std(axis=1)
        if "min" in stats:
            summary["row_min"] = feature_values.min(axis=1)
        if "max" in stats:
            summary["row_max"] = feature_values.max(axis=1)
        if "sum" in stats:
            summary["row_sum"] = feature_values.sum(axis=1)
        if "skew" in stats:
            summary["row_skew"] = feature_values.apply(
                lambda row: row.dropna().skew(), axis=1
            )
        if "kurt" in stats:
            summary["row_kurt"] = feature_values.apply(
                lambda row: row.dropna().kurt(), axis=1
            )
        if "nan_count" in stats:
            summary["row_nan_count"] = feature_values.isna().sum(axis=1)
        if "q25" in stats:
            summary["row_q25"] = percentile(feature_values.values, 25)
        if "q75" in stats:
            summary["row_q75"] = percentile(feature_values.values, 75)
        if "iqr" in stats:
            q75 = summary.get("row_q75") or percentile(feature_values.values, 75)
            q25 = summary.get("row_q25") or percentile(feature_values.values, 25)
            summary["row_iqr"] = q75 - q25

        if summary:
            df = pd.concat([df, pd.DataFrame(summary, index=df.index)], axis=1)
        return df

    def _add_cross_sectional_statistics(
        self, df: pd.DataFrame, feature_cols: Sequence[str]
    ) -> pd.DataFrame:
        config = self.fe_config.get("date_group_statistics", {})
        if not config.get("enabled", True) or not feature_cols:
            return df

        stats = config.get("statistics", ["mean", "std"])
        df = df.copy()

        for stat in stats:
            transformed = df.groupby("date")[feature_cols].transform(stat)
            renamed = {
                col: f"{col}_date_{stat}" for col in feature_cols
            }
            df = df.join(transformed.rename(columns=renamed))

        group_cols = [c for c in self.fe_config.get("group_columns", []) if c in df.columns]
        for group_col in group_cols:
            for stat in stats:
                transformed = df.groupby(["date", group_col])[feature_cols].transform(stat)
                renamed = {
                    col: f"{col}_{group_col}_{stat}" for col in feature_cols
                }
                df = df.join(transformed.rename(columns=renamed))

        return df

    def _add_target_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_config = self.fe_config.get("target_lags", {})
        lags = lag_config.get("lags", [])
        if not lags:
            return df

        target_cols = self._target_columns(df)
        if not target_cols:
            return df

        df = df.copy()
        grouped = df.groupby("numerai_ticker", group_keys=False)
        for target in target_cols:
            for lag in lags:
                lag = int(lag)
                df[f"{target}_lag_{lag}"] = grouped[target].shift(lag)

        return df

    @staticmethod
    def _zscore_clip(series: pd.Series, clip: float) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isclose(std, 0.0):
            return pd.Series(0.0, index=series.index)
        zscore = (series - mean) / std
        return zscore.clip(lower=-clip, upper=clip)
