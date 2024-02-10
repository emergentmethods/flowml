from abc import ABC, abstractmethod
from typing import Any, Callable

from flowdapt.compute.artifacts import Artifact
from flowdapt.lib.utils.misc import generate_name
from flowdapt.compute.artifacts import get_artifact
import glob
import shutil
from flowdapt.lib.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

class BaseMLModel(ABC):
    """
    The base interface a Model subclass must define.
    """
    name: str
    meta: dict[str, Any] = {}

    def __init__(
        self,
        name_str: str | None = None,
        namespace: str | None = None,
        default_logger: bool = False,
        **kwargs
    ) -> None:
        self.default_logger = default_logger
        self.__init_model__(**kwargs)
        self._namespace = namespace
        self._model_train_params = kwargs
        self.name_str = name_str or generate_name()
        self.meta = {
            "params": self.get_params(),
            "namespace": self._namespace,
        }
        self.current_log_dir: str | None = None

    def get_type(self):
        return f"{self.__module__}.{self.__class__.__name__}"

    @abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @abstractmethod
    def predict(self, *args, **kwargs):
        ...

    @classmethod
    @abstractmethod
    def from_artifact(cls, **kwargs) -> Callable[[Artifact], Any]:
        ...

    @abstractmethod
    def to_artifact(self, **kwargs) -> Callable[[Artifact, Any], None]:
        ...

    @abstractmethod
    def __init_model__(self, *args, **kwargs):
        ...

    @abstractmethod
    def get_params(self):
        ...

    def logger_to_artifact(self, **kwargs):
        """
        Copy the local temp tensorboard log to artifact for safe keeping
        Create a unique name for the artifact so that it is easily
        managed by tensorboard
        """
        class_str = self.__class__.__name__
        time_str = int(datetime.now().timestamp())
        artifact_name = (f"{class_str}_{self.name_str}_{time_str}")

        logger.warning("Saving tensorboard logs to artifact")
        events_files = glob.glob(f"{self.current_log_dir}/*.tfevents*")
        if events_files:
            events_file = events_files[0]
            events_file_name = events_file.split("/")[-1]

        artifact = get_artifact(artifact_name, create=True)
        with open(events_file, "rb") as local_file:
            with artifact.new_file(events_file_name).open("wb") as artifact_file:
                artifact_file.write(local_file.read())

        # remove the local directory self.current_log_dir
        shutil.rmtree(self.current_log_dir)
