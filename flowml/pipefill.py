import cloudpickle
import numpy.typing as npt
import numpy as np
from flowdapt.lib.logger import get_logger
from typing import Type, Any
from datasieve.pipeline import Pipeline
from datetime import datetime, timezone

from flowdapt.compute.artifacts import Artifact
from flowml.models.base import BaseMLModel

from flowdapt.lib.utils.misc import import_from_string
from flowdapt.compute.artifacts import get_artifact
# from io import BytesIO
# from flowml.models.pytorch.utils import check_for_gpu


logger = get_logger(__name__)


class PipeFill:
    """
    Class designed to hold all necessary pipeline components to
    more easily move and organize objects at large-scale thousand
    model+ environments.
    It wraps a flowml BaseMLModel object to make cluster-memory
    and distributed storage easier and more efficient.
    """

    def __init__(
        self,
        name_str: str = "",
        namespace: str = "",
        model_str: str = "flowml.XGBoostRegressor",
        model_train_params: dict[str, Any] = {},
        data_split_params: dict[str, Any] = {},
        extras: dict[str, Any] = {}
    ):

        self.model_train_params = model_train_params
        self.data_split_parameters = data_split_params
        self.extras = extras
        self.last_trained_timestamp: int = 0
        self.model_str: str = model_str
        # FIXME: change name from MODELCLASS to something better
        self.MODELCLASS: Type[BaseMLModel] = import_from_string(model_str)
        self.model: BaseMLModel = self.MODELCLASS(
            name_str=name_str,
            namespace=namespace,
            **model_train_params
        )
        self.namespace: str = namespace
        self.model_artifact: Artifact | None = None  # ignore: type
        self.id: str | None = name_str
        self.do_predict: npt.ArrayLike = np.array([])
        self.feature_pipeline: Pipeline = Pipeline()
        self.target_pipeline: Pipeline = Pipeline()
        self.metadata: dict[str, Any] = {}
        self.feature_list: list[str] = []
        self.label_list: list[str] = []
        self.best_loss: float = 0
        self.needs_training: bool = False
        self.update_count: int = 0
        self.update_count_list: list = []

    def set_trained_timestamp(self):
        self.trained_timestamp = datetime.now(tz=timezone.utc).timestamp()

    def clear_data(self):
        """
        Avoid saving too much data. We need to be careful here,
        some outlier detection methods may require some of this
        data. Therefore, we need to recreate the data, or load
        it from disk using the database.
        """
        self.model = None

    def to_dict(self):
        class_dict = vars(self).copy()
        class_dict.pop("MODELCLASS")
        class_dict.pop("model", None)
        return class_dict

    # def __getstate__(self):
    #     buffer = BytesIO()
    #     if self.model:
    #         model_state = self.model.__getstate__()
    #     else:
    #         model_state = None
    #     self.clear_data()
    #     state = self.to_dict()
    #     state["model"] = model_state

    #     buffer.write(cloudpickle.dumps(state))
    #     buffer.seek(0)
    #     return buffer.getvalue()

    # def __setstate__(self, state):
    #     buffer = BytesIO(state)
    #     state = cloudpickle.load(buffer)

    #     for key, value in state.items():
    #         setattr(self, key, value)

    #     self.MODELCLASS = import_from_string(self.model_str)

    #     self.model = self.MODELCLASS()
    #     if state["model"] is not None:
    #         self.model.__setstate__(state["model"])

    @classmethod
    def from_artifact(cls, **kwargs):
        """
        Load a PipeFill from an Artifact.
        """
        # We can make use of any kind of kwargs passed at load time
        # from object_store.get(load_artifact_hook=PipeFill.from_artifact(some_kwarg=some_value))
        # but _inner won't actually be called until the artifact is loaded
        def _inner(artifact: Artifact):
            if not artifact.get_meta("type") == "pipefill":
                raise ValueError("Artifact must be of type `pipefill`")

            instance_pickle = artifact.get_file("pipefill.pkl")

            # Deserialize the pickled model
            instance = cloudpickle.loads(instance_pickle.read())
            model_artifact = get_artifact(
                f"{artifact.name}_model",
                namespace=artifact._namespace
            )
            instance.model = instance.MODELCLASS.from_artifact()(model_artifact)
            return instance
        return _inner

    def to_artifact(self, **kwargs):
        """
        Save a PipeFill to an Artifact.
        """
        # We can make use of any kind of kwargs passed at load time
        # from object_store.get(load_artifact_hook=PipeFill.from_artifact(some_kwarg=some_value))
        # but _inner won't actually be called until the artifact is loaded
        def _inner(artifact: Artifact, _: Any):
            artifact["type"] = "pipefill"
            model_artifact = get_artifact(
                f"{artifact.name}_model",
                namespace=self.namespace,
                create=True
            )
            self.model.to_artifact()(model_artifact, None)
            # we save the model separately above, so now we clear it
            self.clear_data()

            # Create the serialized model object
            serialized_instance = cloudpickle.dumps(self)

            # Save the serialized model to the `model.pkl` file
            pkl = artifact.new_file("pipefill.pkl")
            pkl.write(serialized_instance)
        return _inner
