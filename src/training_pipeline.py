"""End-to-end training pipeline for Numerai Signals Alpha/MPC optimisation."""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.data.ingestion import NumeraiSignalsDataPipeline
from src.models.feature_engineering import AdvancedFeatureEngineer
from src.models.ensemble import AlphaMPCOptimizedEnsemble

LOGGER = logging.getLogger(__name__)


class NumeraiSignalsTrainingPipeline:
    """Complete pipeline orchestrating data prep, modelling, and inference."""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config: Dict = yaml.safe_load(fh)

        self.logger = self._setup_logging()
        self.data_pipeline = NumeraiSignalsDataPipeline(self.config)
        self.feature_engineer = AdvancedFeatureEngineer(self.config)
        self.model = AlphaMPCOptimizedEnsemble(self.config)
        self.feature_columns: List[str] = []
        self.train_metadata_columns: List[str] = []

    # ------------------------------------------------------------------
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare datasets with consistent feature engineering and embargo."""

        self.logger.info("Downloading Numerai Signals data")
        train_df, validation_df, live_df = self.data_pipeline.download_numerai_data()

        combined_historical = pd.concat([train_df, validation_df], ignore_index=True)
        combined_historical = self.data_pipeline.apply_embargo_periods(combined_historical)

        all_tickers = pd.concat(
            [
                combined_historical["numerai_ticker"],
                live_df.get("numerai_ticker", pd.Series(dtype=str)),
            ]
        ).dropna().unique()
        ticker_mapping = self.data_pipeline.get_ticker_mapping(all_tickers)
        yahoo_tickers = list({v for v in ticker_mapping.values() if v})

        market_data = pd.DataFrame()
        if yahoo_tickers and "date" in combined_historical.columns:
            start_date = combined_historical["date"].min().strftime("%Y-%m-%d")
            end_date_candidates = [combined_historical["date"].max()]
            if "date" in live_df.columns and not live_df.empty:
                end_date_candidates.append(live_df["date"].max())
            end_date = max(end_date_candidates).strftime("%Y-%m-%d")
            market_data = self.data_pipeline.fetch_market_data(
                yahoo_tickers, start_date, end_date
            )

        if not market_data.empty:
            reverse_map = {v: k for k, v in ticker_mapping.items()}
            market_data["numerai_ticker"] = market_data["ticker"].map(reverse_map)
            market_data = market_data.dropna(subset=["numerai_ticker"])
            market_data = market_data.drop(columns=["ticker"], errors="ignore")

            combined_historical = combined_historical.merge(
                market_data,
                on=["numerai_ticker", "date"],
                how="left",
            )
            if not live_df.empty:
                live_df = live_df.merge(
                    market_data,
                    on=["numerai_ticker", "date"],
                    how="left",
                )

        self.logger.info("Applying feature engineering to historical data")
        historical_features = self.feature_engineer.create_all_features(combined_historical)
        historical_features = self.feature_engineer.create_interaction_features(
            historical_features
        )
        historical_features = self.feature_engineer.apply_cross_sectional_normalization(
            historical_features
        )

        live_features = pd.DataFrame()
        if not live_df.empty:
            self.logger.info("Applying feature engineering to live data")
            live_features = self.feature_engineer.create_all_features(live_df, is_live=True)
            live_features = self.feature_engineer.create_interaction_features(live_features)
            live_features = self.feature_engineer.apply_cross_sectional_normalization(
                live_features
            )

        split_col = self.config.get("data", {}).get("split_column", "data_type")
        train_label = self.config.get("data", {}).get("train_label", "train")
        validation_label = self.config.get("data", {}).get("validation_label", "validation")

        if split_col in historical_features.columns:
            train_data = historical_features[historical_features[split_col] == train_label]
            val_data = historical_features[historical_features[split_col] == validation_label]
        else:
            cutoff = train_df["date"].max()
            train_data = historical_features[historical_features["date"] <= cutoff]
            val_data = historical_features[historical_features["date"] > cutoff]

        self.feature_columns = [
            col
            for col in train_data.columns
            if col.startswith(tuple(self.config.get("feature_engineering", {}).get("feature_prefixes", ["feature_"])))
        ]
        self.train_metadata_columns = [
            col
            for col in [
                "numerai_ticker",
                "date",
                split_col,
                self.model.model_config.get("meta_model_column", "meta_model"),
            ]
            if col in train_data.columns
        ]

        self.logger.info(
            "Train samples: %s, Validation samples: %s, Features: %s",
            len(train_data),
            len(val_data),
            len(self.feature_columns),
        )

        return train_data, val_data, live_features

    # ------------------------------------------------------------------
    def train_model(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Train the ensemble and return validation metrics."""

        target_col = self.model.model_config.get("target_column", "target")
        if target_col not in train_data.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in training data."
            )
        target_columns = [col for col in train_data.columns if col.startswith("target")]

        exclude_cols = set(
            [
                "numerai_ticker",
                "date",
                "data_type",
                "era",
                "ticker",
                self.model.meta_model_column,
            ]
            + target_columns
        )

        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        self.feature_columns = feature_cols

        train_mask = ~train_data[target_col].isna()
        val_mask = ~val_data[target_col].isna()

        X_train = train_data.loc[train_mask, feature_cols]
        y_train = train_data.loc[train_mask, target_col]
        train_metadata = train_data.loc[train_mask, self.train_metadata_columns]

        X_val = val_data.loc[val_mask, feature_cols]
        y_val = val_data.loc[val_mask, target_col]
        val_metadata = val_data.loc[val_mask, self.train_metadata_columns]

        self.logger.info("Training ensemble with %s features", len(feature_cols))
        self.model.train_ensemble(X_train, y_train, train_metadata, X_val, y_val, val_metadata)

        val_predictions = self.model.predict(X_val)
        metrics = self.model.calculate_alpha_mpc_proxy(
            val_predictions, y_val.values, val_metadata
        )
        self.logger.info("Validation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    def generate_predictions(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """Generate submission-ready predictions for live data."""

        if live_data is None or live_data.empty:
            raise ValueError("Live data is empty; prepare_data must be run first.")

        target_columns = [col for col in live_data.columns if col.startswith("target")]
        exclude_cols = set(
            [
                "numerai_ticker",
                "date",
                "data_type",
                "ticker",
                self.model.meta_model_column,
            ]
            + target_columns
        )
        feature_cols = [col for col in live_data.columns if col not in exclude_cols]

        if not self.model.selected_features:
            raise ValueError("Model must be trained before generating predictions.")

        X_live = live_data.reindex(columns=self.model.selected_features, fill_value=0.0)
        X_live = X_live.fillna(0)
        predictions = self.model.predict(X_live)

        submission = pd.DataFrame(
            {
                "numerai_ticker": live_data["numerai_ticker"],
                "signal": predictions,
            }
        )

        submission["signal"] = np.clip(submission["signal"], 0.001, 0.999)
        min_val, max_val = submission["signal"].min(), submission["signal"].max()
        if max_val > min_val:
            submission["signal"] = (submission["signal"] - min_val) / (max_val - min_val)
            submission["signal"] = np.clip(submission["signal"], 0.001, 0.999)

        return submission

    # ------------------------------------------------------------------
    def save_model(self, metrics: Dict[str, float]) -> str:
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{self.model.model_config.get('name', 'signals_model')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        self.model.save_model(str(model_path))
        metrics_path = model_path.with_suffix(".metrics.json")
        metrics_path.write_text(pd.Series(metrics).to_json(), encoding="utf-8")
        return str(model_path)


def run_pipeline(config_path: str = "config/model_config.yaml") -> None:
    pipeline = NumeraiSignalsTrainingPipeline(config_path)
    train_data, val_data, live_data = pipeline.prepare_data()
    metrics = pipeline.train_model(train_data, val_data)
    pipeline.save_model(metrics)
    submission = pipeline.generate_predictions(live_data)
    submission_path = Path("submissions")
    submission_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    submission_file = submission_path / f"signals_submission_{timestamp}.csv"
    submission.to_csv(submission_file, index=False)
    LOGGER.info("Saved live predictions to %s", submission_file)


if __name__ == "__main__":
    run_pipeline()
