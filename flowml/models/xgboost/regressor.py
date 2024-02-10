from typing import Any, Callable
from joblib import dump, load
from xgboost.dask import DaskXGBRegressor
from xgboost import XGBRegressor

from flowml.models.base import BaseMLModel
from flowdapt.compute.artifacts import Artifact
from flowml.tensorboard import TensorBoardCallback
from flowdapt.lib.logger import get_logger
from sklearn.metrics import mean_squared_error
import os
import tempfile

import warnings

# Suppress XGBoost warnings
warnings.filterwarnings(action='ignore', module='xgboost')


logger = get_logger(__name__)

class XGBoostRegressor(BaseMLModel):
    def __init_model__(self, **kwargs) -> "XGBoostRegressor":
        self._model = XGBRegressor(**kwargs)
        return self

    def get_params(self) -> dict:
        return self._model.get_params()

    @classmethod
    def from_artifact(cls, **kwargs) -> Callable[[Artifact], Any]:
        def _inner(artifact: Artifact):
            # Require model artifact
            if not artifact.get_meta("type", None) == "model":
                raise ValueError("Artifact must be of type `model`")

            # Get the meta from the Artifact
            model_params = artifact.get_meta("params", {})
            kwargs.update(model_params)

            file = artifact.get_file("model.joblib")

            with file.open(mode="rb") as f:
                model = load(f)

            # Create the instance and assign the model
            instance = cls(name=artifact.name, **model_params)
            instance._model = model

            return instance
        return _inner

    def to_artifact(self, **kwargs) -> Callable[[Artifact, Any], None]:
        def _inner(artifact: Artifact, value: Any):
            # Add the meta to the artifact
            artifact["type"] = "model"
            artifact["params"] = self.get_params()
            artifact["model_type"] = self.get_type()

            with artifact.new_file("model.joblib", exist_ok=True).open(mode="wb") as f:
                dump(self._model, f)
        return _inner

    def fit(self, *args, **kwargs):
        eval_set = kwargs["eval_set"]
        if self.default_logger:
            self.current_log_dir = os.path.join(tempfile.gettempdir(), self.name_str)
            os.makedirs(self.current_log_dir, exist_ok=True)
            self._model.set_params(callbacks=[
                TensorBoardCallback(self.current_log_dir)
            ])
        self._model.fit(*args, verbose=False, **kwargs)
        if self.default_logger:
            self.logger_to_artifact()
        self._model.set_params(callbacks=[])
        self.best_loss = self.check_eval(eval_set)
        return self

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    def check_eval(self, eval_set: Any):
        ys = self._model.predict(eval_set[0][0])
        valid_loss = mean_squared_error(eval_set[0][1], ys)
        logger.info(f"Validation loss: {valid_loss}")
        return valid_loss


class DaskXGBoostRegressor(XGBoostRegressor):
    def __init_model__(self, name_str, **kwargs):
        return DaskXGBRegressor(**kwargs)
