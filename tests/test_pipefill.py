from flowdapt.compute.artifacts import Artifact
from flowml.pipefill import PipeFill
import pytest

@pytest.fixture(scope="function")
def pipe_fill():
    return PipeFill(
        name_str="pipe_fill",
        namespace="study123",
        model_str="flowml.XGBoostRegressor",
        model_train_params={"param1": "value1", "param2": "value2"},
        data_split_params={"param3": "value3", "param4": "value4"},
        extras={"param5": "value5", "param6": "value6"}
    )

@pytest.fixture(scope="function")
def artifact_params():
    return {"protocol": "memory", "base_path": "test"}


@pytest.fixture(scope="function")
def artifact(artifact_params):
    artifact = Artifact.new_artifact(
        name="test_artifact",
        **artifact_params
    )
    try:
        yield artifact
    finally:
        artifact.delete()



def test_initializer():
    pipe_fill = PipeFill(
            name_str="pipe_fill",
            namespace="study123",
            model_str="flowml.XGBoostRegressor",
            model_train_params={"param1": "value1", "param2": "value2"},
            data_split_params={"param3": "value3", "param4": "value4"},
            extras={"param5": "value5", "param6": "value6"}
        )
    assert pipe_fill.id == "pipe_fill"
    assert pipe_fill.model_str == "flowml.XGBoostRegressor"
    assert pipe_fill.model_train_params == {"param1": "value1", "param2": "value2"}
    assert pipe_fill.data_split_parameters == {"param3": "value3", "param4": "value4"}
    assert pipe_fill.extras == {"param5": "value5", "param6": "value6"}


def test_pipefill_to_artifact(pipe_fill, artifact, mocked_default_values):
    """
    Make an artifact and save the pipefill to that
    """
    pipe_fill.to_artifact()(artifact, pipe_fill)
    assert not artifact.is_empty, artifact.list_files()


def test_pipefill_from_artifact(pipe_fill, artifact, mocked_default_values):
    """
    Load the pipefill from an artifact
    """
    pipe_fill.to_artifact()(artifact, pipe_fill)

    pf = PipeFill.from_artifact()(artifact)

    assert pf.id == "pipe_fill"


def test_pipefill_to_dict(pipe_fill):
    pipe_fill_dict = pipe_fill.to_dict()

    assert pipe_fill_dict["id"] == "pipe_fill"
