"""Data ingestion utilities for Numerai Signals.

This module handles downloading Numerai Signals datasets, mapping ticker
identifiers to Yahoo Finance symbols, augmenting the data with
market information, and applying embargo rules to prevent look-ahead
bias during training/validation splits.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class MarketDataConfig:
    """Configuration parameters for fetching market data."""

    yahoo_price_features: bool = True
    max_tickers_for_market_data: int = 500


class NumeraiSignalsDataPipeline:
    """Utility class that orchestrates Numerai Signals data ingestion."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get("data", {})
        self.market_config = MarketDataConfig(
            yahoo_price_features=self.data_config.get("yahoo_price_features", True),
            max_tickers_for_market_data=self.data_config.get(
                "max_tickers_for_market_data", 500
            ),
        )
        self.data_dir = Path(self.data_config.get("data_dir", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset download utilities
    # ------------------------------------------------------------------
    def download_numerai_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download (or load) Numerai Signals datasets.

        The pipeline first looks for existing files in ``data_dir``. If files
        are missing it attempts to download them using ``numerapi`` when API
        credentials are available. Users without API credentials can manually
        place the datasets into ``data_dir`` and this method will pick them up.
        """

        dataset_version = self.data_config.get("dataset_version", "v2.1")
        filenames = {
            "train": self.data_config.get("train_filename", "signals_train.parquet"),
            "validation": self.data_config.get(
                "validation_filename", "signals_validation.parquet"
            ),
            "live": self.data_config.get("live_filename", "signals_live.parquet"),
        }

        datasets: Dict[str, pd.DataFrame] = {}
        for split, filename in filenames.items():
            dataset_path = self.data_dir / filename
            if not dataset_path.exists():
                LOGGER.info(
                    "Dataset %s missing locally. Attempting to download %s/%s",
                    filename,
                    dataset_version,
                    filename,
                )
                self._download_dataset(dataset_version, filename, dataset_path)

            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Unable to locate Numerai Signals dataset: {dataset_path}. "
                    "Download the Signals v2.1 data from https://signals.numer.ai/data/v2.1 "
                    "and place the files in the configured data directory."
                )

            datasets[split] = self._load_dataset(dataset_path)
            datasets[split]["data_type"] = split

        # Ensure datetime dtype
        for df in datasets.values():
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        return datasets["train"], datasets["validation"], datasets["live"]

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        """Load parquet or CSV dataset into a DataFrame."""

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    def _download_dataset(self, version: str, filename: str, destination: Path) -> None:
        """Download a dataset using numerapi when credentials are available."""

        public_id = os.getenv("NUMERAI_PUBLIC_ID") or self.data_config.get(
            "numerai_public_id"
        )
        secret_key = os.getenv("NUMERAI_SECRET_KEY") or self.data_config.get(
            "numerai_secret_key"
        )

        try:
            from numerapi import SignalsAPI  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "numerapi is not installed; cannot download %s automatically.", filename
            )
            return

        if not public_id or not secret_key:
            LOGGER.warning(
                "Numerai API credentials not provided. Skipping automatic download of %s.",
                filename,
            )
            return

        api = SignalsAPI(public_id=public_id, secret_key=secret_key)

        try:
            api.download_dataset(
                filename=f"{version}/{filename}",
                dest_path=str(destination),
                chunk_size=1024 * 1024,
            )
            LOGGER.info("Downloaded dataset %s successfully", filename)
        except Exception as exc:  # pragma: no cover - network operation
            LOGGER.error("Failed to download %s: %s", filename, exc)

    # ------------------------------------------------------------------
    # Market data utilities
    # ------------------------------------------------------------------
    def get_ticker_mapping(self, numerai_tickers: Iterable[str]) -> Dict[str, str]:
        """Map Numerai tickers to Yahoo Finance compatible tickers."""

        mapping: Dict[str, str] = {}
        suffix_map = self.data_config.get("exchange_suffix_map", {})

        for ticker in numerai_tickers:
            if ticker is None or ticker in mapping:
                continue

            base_symbol = ticker
            if ":" in ticker:
                exchange, symbol = ticker.split(":", 1)
                suffix = suffix_map.get(exchange.upper(), "")
                base_symbol = symbol.replace("/", "-").replace(".", "-") + suffix
            else:
                base_symbol = ticker.replace("/", "-").replace(".", "-")

            mapping[ticker] = base_symbol

        return mapping

    def fetch_market_data(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data for the provided tickers from Yahoo Finance."""

        if not self.market_config.yahoo_price_features or not tickers:
            return pd.DataFrame()

        capped_tickers = tickers[: self.market_config.max_tickers_for_market_data]

        try:
            import yfinance as yf  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "yfinance is not installed; skipping market data augmentation."
            )
            return pd.DataFrame()

        records: List[pd.DataFrame] = []
        for ticker in capped_tickers:
            try:
                history = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:  # pragma: no cover - network operation
                LOGGER.warning("Failed to download market data for %s: %s", ticker, exc)
                continue

            if history.empty:
                continue

            history = history.reset_index()
            history.rename(columns={"Date": "date"}, inplace=True)
            history["ticker"] = ticker
            records.append(history)

        if not records:
            return pd.DataFrame()

        market_data = pd.concat(records, ignore_index=True)
        market_data["date"] = pd.to_datetime(market_data["date"])

        # Basic price-derived features
        market_data.sort_values(["ticker", "date"], inplace=True)
        market_data["close_to_open"] = (
            market_data["Close"] / market_data["Open"]
        ).replace([np.inf, -np.inf], np.nan)
        market_data["high_to_low"] = (
            market_data["High"] / market_data["Low"]
        ).replace([np.inf, -np.inf], np.nan)
        market_data["return_1d"] = (
            market_data.groupby("ticker")["Close"].pct_change()
        )
        market_data["log_return_1d"] = np.log1p(market_data["return_1d"])
        market_data["volume_log"] = np.log1p(market_data["Volume"])

        for window in (5, 10, 21, 63):
            market_data[f"close_ma_{window}"] = (
                market_data.groupby("ticker")["Close"]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
            )
            market_data[f"return_vol_{window}"] = (
                market_data.groupby("ticker")["return_1d"]
                .transform(lambda s: s.rolling(window, min_periods=2).std())
            )

        return market_data

    # ------------------------------------------------------------------
    # Embargo utilities
    # ------------------------------------------------------------------
    def apply_embargo_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove training rows that fall within the embargo window."""

        split_column = self.data_config.get("split_column", "data_type")
        train_label = self.data_config.get("train_label", "train")
        validation_label = self.data_config.get("validation_label", "validation")
        embargo_days = int(self.data_config.get("embargo_days", 60))

        if split_column not in df.columns or "date" not in df.columns:
            return df

        train_mask = df[split_column] == train_label
        validation_mask = df[split_column] == validation_label

        if not validation_mask.any():
            return df

        validation_start = df.loc[validation_mask, "date"].min()
        if pd.isna(validation_start):
            return df

        embargo_start = validation_start - pd.Timedelta(days=embargo_days)
        embargo_mask = train_mask & (df["date"] >= embargo_start) & (
            df["date"] < validation_start
        )

        if embargo_mask.any():
            LOGGER.info(
                "Applying embargo: removing %s rows within %s days before %s",
                embargo_mask.sum(),
                embargo_days,
                validation_start.date(),
            )
            df = df.loc[~embargo_mask].copy()

        return df


def as_timezone_aware(date_series: pd.Series) -> pd.Series:
    """Convert a date series to timezone-aware UTC timestamps."""

    if pd.api.types.is_datetime64_any_dtype(date_series):
        return date_series.dt.tz_localize("UTC", nonexistent="shift_forward")
    return pd.to_datetime(date_series).dt.tz_localize("UTC")
