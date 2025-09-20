"""Alpha/MPC optimised ensemble model for Numerai Signals."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


class AlphaMPCOptimizedEnsemble:
    """An ensemble of diverse regressors tuned for Alpha and MPC."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.random_state = self.model_config.get("random_seed", 42)
        self.meta_model_column = self.model_config.get("meta_model_column", "meta_model")

        self.base_models: Dict[str, object] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.selected_features: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.hgb_params: Dict[str, float] = {}
        self.rng = np.random.default_rng(self.random_state)

    # ------------------------------------------------------------------
    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        train_metadata: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        val_metadata: pd.DataFrame,
    ) -> None:
        """Train ensemble models and learn blending weights."""

        X_train = X_train.copy()
        X_val = X_val.copy()

        # Feature selection
        self.selected_features = self._select_features(X_train, y_train)
        X_train = X_train[self.selected_features].fillna(0)
        X_val = X_val[self.selected_features].fillna(0)

        # Scaling
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=self.selected_features,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=self.selected_features,
            index=X_val.index,
        )

        # Hyper-parameter tuning for HistGradientBoosting
        self._maybe_run_optuna(X_train_scaled, y_train, train_metadata)

        # Fit base models
        self._fit_base_models(X_train_scaled, y_train)

        # Optimise blending weights on validation data
        val_predictions = {
            name: model.predict(X_val_scaled) for name, model in self.base_models.items()
        }
        self.ensemble_weights = self._optimise_weights(
            val_predictions, y_val, val_metadata
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.selected_features:
            raise ValueError("Model has not been trained; no selected features available.")
        if not self.ensemble_weights:
            raise ValueError("Model has not been trained; ensemble weights missing.")

        X = X.copy()
        missing_features = [f for f in self.selected_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features[:10]}")

        X = X[self.selected_features].fillna(0)
        if self.scaler is None:
            raise ValueError("Scaler missing. Did you train the model?")
        X_scaled = self.scaler.transform(X)

        preds = np.zeros(len(X_scaled))
        for name, model in self.base_models.items():
            weight = self.ensemble_weights.get(name, 0)
            if weight == 0:
                continue
            preds += weight * model.predict(X_scaled)
        return preds

    def save_model(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "selected_features": self.selected_features,
            "ensemble_weights": self.ensemble_weights,
            "hgb_params": self.hgb_params,
            "scaler": self.scaler,
            "models": self.base_models,
        }
        joblib.dump(state, path)
        LOGGER.info("Saved model to %s", path)

    # ------------------------------------------------------------------
    def calculate_alpha_mpc_proxy(
        self,
        predictions: np.ndarray,
        target: np.ndarray,
        metadata: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute proxy metrics for Alpha and MPC along with stability penalties."""

        eval_df = metadata.copy()
        eval_df["prediction"] = predictions
        eval_df["target"] = target

        alpha = self._compute_era_correlation(eval_df, "prediction", "target")
        mpc = self._compute_meta_model_corr(eval_df, fallback=alpha)
        turnover = self._compute_turnover(eval_df)
        churn = self._compute_churn(eval_df)

        score = 0.3 * alpha + 0.8 * mpc
        penalties = 0.0
        training_cfg = self.training_config
        turnover_threshold = training_cfg.get("turnover_target")
        churn_threshold = training_cfg.get("churn_target")
        if turnover_threshold is not None and not np.isnan(turnover):
            excess = max(0.0, turnover - turnover_threshold)
            penalties += excess * training_cfg.get("turnover_penalty", 0.1)
        if churn_threshold is not None and not np.isnan(churn):
            excess = max(0.0, churn - churn_threshold)
            penalties += excess * training_cfg.get("churn_penalty", 0.05)

        score -= penalties

        return {
            "alpha": alpha,
            "mpc": mpc,
            "turnover": turnover,
            "churn": churn,
            "penalty": penalties,
            "score": score,
        }

    # ------------------------------------------------------------------
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        top_k = self.model_config.get("feature_selection", {}).get("top_k", "all")
        X = X.fillna(0)
        if top_k == "all" or top_k is None:
            return list(X.columns)

        top_k = min(int(top_k), X.shape[1])
        self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=top_k)
        self.feature_selector.fit(X, y)
        mask = self.feature_selector.get_support()
        selected = X.columns[mask].tolist()
        return selected

    def _fit_base_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.base_models = {}

        # HistGradientBoostingRegressor (primary model)
        hgb_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_iter": 300,
            "l2_regularization": 0.0,
            "random_state": self.random_state,
        }
        hgb_params.update(self.hgb_params)
        self.base_models["hgb"] = HistGradientBoostingRegressor(**hgb_params)
        self.base_models["hgb"].fit(X, y)

        # Ridge regression on scaled features
        ridge = Ridge(alpha=1.0, random_state=self.random_state)
        ridge.fit(X, y)
        self.base_models["ridge"] = ridge

        # ElasticNet for sparse interactions
        elastic = ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=self.random_state,
        )
        elastic.fit(X, y)
        self.base_models["elastic"] = elastic

        # RandomForest as a non-linear alternative
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        self.base_models["rf"] = rf

    def _optimise_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y_val: pd.Series,
        val_metadata: pd.DataFrame,
    ) -> Dict[str, float]:
        model_names = list(predictions.keys())
        if not model_names:
            raise ValueError("No base model predictions available for ensembling.")

        best_score = -np.inf
        best_weights = None
        weight_trials = max(100, 20 * len(model_names))

        for _ in range(weight_trials):
            weights = self.rng.dirichlet(np.ones(len(model_names)))
            blended = np.zeros_like(list(predictions.values())[0])
            for weight, name in zip(weights, model_names):
                blended += weight * predictions[name]

            metrics = self.calculate_alpha_mpc_proxy(blended, y_val.values, val_metadata)
            score = metrics["score"]
            if score > best_score:
                best_score = score
                best_weights = dict(zip(model_names, weights))

        LOGGER.info("Optimised ensemble weights: %s", best_weights)
        return best_weights or {name: 1.0 / len(model_names) for name in model_names}

    # ------------------------------------------------------------------
    def _maybe_run_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
    ) -> None:
        optuna_config = self.model_config.get("optuna", {})
        if not optuna_config.get("enabled", False):
            return

        try:
            import optuna  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning("Optuna is not installed; skipping hyper-parameter tuning.")
            return

        LOGGER.info("Starting Optuna optimisation for HistGradientBoostingRegressor")

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "max_iter": trial.suggest_int("max_iter", 200, 800),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            }
            scores = []
            for train_idx, val_idx in self._generate_time_series_folds(metadata):
                model = HistGradientBoostingRegressor(
                    **params, random_state=self.random_state
                )
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_val_fold)
                fold_metadata = metadata.iloc[val_idx]
                metrics = self.calculate_alpha_mpc_proxy(
                    preds, y_val_fold.values, fold_metadata
                )
                scores.append(metrics["score"])
            return float(np.nanmean(scores))

        study = optuna.create_study(direction="maximize")
        n_trials = optuna_config.get("n_trials", 25)
        timeout = optuna_config.get("timeout")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.hgb_params = study.best_params
        LOGGER.info("Best Optuna params: %s", self.hgb_params)

    # ------------------------------------------------------------------
    def _generate_time_series_folds(
        self, metadata: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        cv_config = self.training_config.get("cv", {})
        n_splits = int(cv_config.get("n_splits", 5))
        min_train = int(cv_config.get("min_train_windows", 252))
        val_window = int(cv_config.get("validation_windows", 63))
        embargo = int(cv_config.get("embargo_days", 60))

        metadata = metadata.copy()
        metadata["date"] = pd.to_datetime(metadata["date"])
        unique_dates = np.sort(metadata["date"].unique())

        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        start_idx = 0
        while len(folds) < n_splits:
            train_end_idx = start_idx + min_train
            val_start_idx = train_end_idx + embargo
            val_end_idx = val_start_idx + val_window
            if val_end_idx >= len(unique_dates):
                break

            train_dates = unique_dates[start_idx:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            train_idx = metadata.index[metadata["date"].isin(train_dates)].to_numpy()
            val_idx = metadata.index[metadata["date"].isin(val_dates)].to_numpy()

            if len(train_idx) == 0 or len(val_idx) == 0:
                break

            folds.append((train_idx, val_idx))
            start_idx += val_window

        if not folds:
            indices = np.arange(len(metadata))
            split = int(len(metadata) * 0.8)
            folds.append((indices[:split], indices[split:]))

        return folds

    # ------------------------------------------------------------------
    def _compute_era_correlation(self, df: pd.DataFrame, pred_col: str, target_col: str) -> float:
        correlations = []
        for _, group in df.groupby("date"):
            if group[target_col].std() == 0 or group[pred_col].std() == 0:
                continue
            corr = group[pred_col].corr(group[target_col], method="spearman")
            if not np.isnan(corr):
                correlations.append(corr)
        if not correlations:
            return float("nan")
        return float(np.nanmean(correlations))

    def _compute_meta_model_corr(self, df: pd.DataFrame, fallback: Optional[float] = None) -> float:
        if self.meta_model_column not in df.columns:
            if fallback is not None and not np.isnan(fallback):
                LOGGER.warning(
                    "Meta model column '%s' missing; using alpha as MPC proxy.",
                    self.meta_model_column,
                )
                return float(fallback)
            return float("nan")
        correlations = []
        for _, group in df.groupby("date"):
            if self.meta_model_column not in group:
                continue
            meta = group[self.meta_model_column]
            if meta.std() == 0:
                continue
            corr = group["prediction"].corr(meta, method="spearman")
            if not np.isnan(corr):
                correlations.append(corr)
        if not correlations:
            return float("nan")
        return float(np.nanmean(correlations))

    def _compute_turnover(self, df: pd.DataFrame) -> float:
        if "numerai_ticker" not in df.columns:
            return float("nan")
        df = df.sort_values("date")
        turnover_values = []
        prev_date = None
        prev_preds: Optional[pd.DataFrame] = None

        for date, group in df.groupby("date"):
            current = group[["numerai_ticker", "prediction"]]
            if prev_preds is not None and prev_date is not None:
                merged = current.merge(
                    prev_preds,
                    on="numerai_ticker",
                    suffixes=("_curr", "_prev"),
                )
                if len(merged) > 5:
                    corr = merged["prediction_curr"].corr(
                        merged["prediction_prev"], method="spearman"
                    )
                    if not np.isnan(corr):
                        turnover_values.append(1 - corr)
            prev_preds = current.rename(columns={"prediction": "prediction_prev"})
            prev_date = date

        if not turnover_values:
            return float("nan")
        return float(np.nanmean(turnover_values))

    def _compute_churn(self, df: pd.DataFrame) -> float:
        if "numerai_ticker" not in df.columns:
            return float("nan")
        df = df.sort_values(["numerai_ticker", "date"])
        diffs = (
            df.groupby("numerai_ticker")["prediction"].diff().abs()
        )
        if diffs.isna().all():
            return float("nan")
        return float(diffs.mean())
