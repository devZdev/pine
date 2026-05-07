"""
regime/model.py
===============
Chronos-T5 loader, inference pipeline, and forecast extraction.

Responsibilities
----------------
- Load ``amazon/chronos-t5-tiny`` (or any variant via ``CHRONOS_MODEL_ID``) on
  the CPU using the HuggingFace ``transformers`` pipeline.
- Accept a 1-D numpy array of close prices and produce quantile forecasts for
  the next ``prediction_length`` steps.
- Return structured ``ForecastResult`` objects that the classifier layer
  consumes without any dependency on the HuggingFace internals.

Notes on Chronos
----------------
Chronos is a probabilistic time-series foundation model fine-tuned from T5.
We use the ``ChronosPipeline`` helper that ships with the
``autogluon.timeseries`` package *or* the standalone ``chronos-forecasting``
package.  Both expose the same ``predict()`` API.  We prefer the standalone
package so that the heavy AutoGluon dependency is not required.

The ``predict()`` call returns a tensor of shape
``(num_series, num_samples, prediction_length)``; we convert it to numpy and
derive empirical quantiles from the sample distribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Structured output from a single Chronos inference call.

    Attributes
    ----------
    q10 : np.ndarray
        10th-percentile quantile forecasts, shape ``(prediction_length,)``.
    q50 : np.ndarray
        Median (50th-percentile) forecasts.
    q90 : np.ndarray
        90th-percentile forecasts.
    samples : np.ndarray
        Raw Monte-Carlo samples, shape ``(num_samples, prediction_length)``.
    prediction_length : int
        Number of future steps forecast.
    """

    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    samples: np.ndarray
    prediction_length: int = 10

    # Derived convenience properties ----------------------------------------

    @property
    def spread(self) -> np.ndarray:
        """Per-step width of the 80 % predictive interval (q90 - q10)."""
        return self.q90 - self.q10

    @property
    def mean_spread(self) -> float:
        """Mean spread across all forecast steps."""
        return float(np.mean(self.spread))

    @property
    def forecast_low(self) -> float:
        """Lowest price in the q10 forecast band."""
        return float(np.min(self.q10))

    @property
    def forecast_high(self) -> float:
        """Highest price in the q90 forecast band."""
        return float(np.max(self.q90))


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ChronosForecaster:
    """Thin wrapper around the Chronos-T5 pipeline.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.  Defaults to the value of the
        ``CHRONOS_MODEL_ID`` environment variable, falling back to
        ``amazon/chronos-t5-tiny``.
    hf_token : str | None
        HuggingFace access token.  Read from ``HUGGINGFACE_TOKEN`` env var
        when not supplied explicitly.
    num_samples : int
        Number of Monte-Carlo samples per prediction (higher → better
        quantile estimates, slower inference).  Default 20.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        num_samples: int = 20,
    ) -> None:
        self._model_id: str = (
            model_id
            or os.environ.get("CHRONOS_MODEL_ID", "amazon/chronos-t5-tiny")
        )
        self._hf_token: Optional[str] = (
            hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        )
        self._num_samples: int = num_samples
        self._pipeline: Optional[object] = None  # lazily set in load()
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Download (first run) or restore from cache, then load the model.

        Raises
        ------
        ImportError
            If ``chronos`` (``chronos-forecasting`` package) is not installed.
        RuntimeError
            If the model fails to load for any other reason.
        """
        try:
            from chronos import ChronosPipeline  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'chronos-forecasting' package is required.  "
                "Install it with: pip install chronos-forecasting"
            ) from exc

        import torch  # type: ignore[import-untyped]

        logger.info("Loading Chronos model '{}' on CPU…", self._model_id)

        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.isdir(os.path.join(cache_dir, "hub")):
            logger.debug("HuggingFace cache directory found — model may load from cache.")

        # ``ChronosPipeline.from_pretrained`` accepts ``use_auth_token`` for
        # gated models and ``torch_dtype`` for precision control.
        self._pipeline = ChronosPipeline.from_pretrained(
            self._model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            token=self._hf_token or None,
        )
        self._loaded = True
        logger.info("Chronos model '{}' ready.", self._model_id)

    def predict(
        self,
        context: np.ndarray,
        prediction_length: int = 10,
    ) -> ForecastResult:
        """Run zero-shot probabilistic forecasting.

        Parameters
        ----------
        context : np.ndarray
            1-D array of close prices (recent history), shape ``(T,)``.
            Values must be positive and finite.
        prediction_length : int
            Number of future steps to forecast.

        Returns
        -------
        ForecastResult
            Quantile forecasts and raw samples.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called first.
        ValueError
            If the context array is empty or contains non-finite values.
        """
        if not self._loaded or self._pipeline is None:
            raise RuntimeError(
                "ChronosForecaster.load() must be called before predict()."
            )

        context = np.asarray(context, dtype=np.float32)
        _validate_context(context)

        import torch  # type: ignore[import-untyped]

        # Chronos expects a list of tensors (one per time-series).
        ctx_tensor = torch.tensor(context).unsqueeze(0)  # shape: (1, T)

        logger.debug(
            "Running Chronos inference: context_len={} prediction_length={} num_samples={}",
            len(context),
            prediction_length,
            self._num_samples,
        )

        # ``forecast_samples`` shape: (num_series=1, num_samples, prediction_length)
        forecast_samples: torch.Tensor = self._pipeline.predict(
            context=ctx_tensor,
            prediction_length=prediction_length,
            num_samples=self._num_samples,
        )

        # Extract the single series → shape: (num_samples, prediction_length)
        samples_np: np.ndarray = forecast_samples[0].numpy().astype(np.float64)

        q10 = np.quantile(samples_np, 0.10, axis=0)
        q50 = np.quantile(samples_np, 0.50, axis=0)
        q90 = np.quantile(samples_np, 0.90, axis=0)

        logger.debug(
            "Forecast: q10[0]={:.4f} q50[0]={:.4f} q90[0]={:.4f}",
            q10[0],
            q50[0],
            q90[0],
        )

        return ForecastResult(
            q10=q10,
            q50=q50,
            q90=q90,
            samples=samples_np,
            prediction_length=prediction_length,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_context(context: np.ndarray) -> None:
    """Raise ``ValueError`` if the context array is unusable."""
    if context.ndim != 1:
        raise ValueError(
            f"context must be a 1-D array; got shape {context.shape}."
        )
    if len(context) == 0:
        raise ValueError("context array is empty.")
    if not np.all(np.isfinite(context)):
        n_bad = int(np.sum(~np.isfinite(context)))
        raise ValueError(
            f"context contains {n_bad} non-finite value(s) (NaN or Inf). "
            "Pre-process the series before calling predict()."
        )
    if np.any(context <= 0):
        n_neg = int(np.sum(context <= 0))
        logger.warning(
            "context contains {} non-positive value(s).  "
            "Chronos expects strictly positive prices; results may be unreliable.",
            n_neg,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (populated by lifespan startup)
# ---------------------------------------------------------------------------

# The global forecaster instance is instantiated in ``main_regime.py`` and
# stored here so that ``router.py`` can import it without circular dependencies.
_forecaster: Optional[ChronosForecaster] = None


def get_forecaster() -> ChronosForecaster:
    """Return the module-level singleton, raising if not yet initialised."""
    if _forecaster is None:
        raise RuntimeError(
            "ChronosForecaster has not been initialised.  "
            "Ensure the FastAPI lifespan startup has completed."
        )
    return _forecaster


def set_forecaster(forecaster: ChronosForecaster) -> None:
    """Set the module-level singleton (called by lifespan startup)."""
    global _forecaster
    _forecaster = forecaster
