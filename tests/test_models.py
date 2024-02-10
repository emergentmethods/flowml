from flowml import XGBoostRegressor, PyTorchLSTMAutoregressive, PyTorchTransformer, PyTorchMLPRegressor, PyTorchLSTMDirect
import numpy
import pytest

@pytest.fixture(scope="function")
def torch_params():
    return {
        "input_features": 200,
        "hidden_dim": 64,
        "output_features": 6,
        "num_layers": 1,
        "dropout": 0.2,
        "batch_size": 10,
        "epochs": 5,
        "lookback": 6
    }

@pytest.fixture(scope="function")
def xs():
    numpy.random.seed(123)
    return numpy.random.rand(100, 200)

@pytest.fixture(scope="function")
def ys():
    numpy.random.seed(123)
    return numpy.random.rand(100, 6)

@pytest.fixture(scope="function")
def xs_test():
    numpy.random.seed(124)
    return numpy.random.rand(6, 200)

@pytest.fixture(scope="function")
def ys_test():
    numpy.random.seed(124)
    return numpy.random.rand(6, 6)


def test_train_loop_XGBoostRegressor(xs, ys, xs_test, ys_test, mocked_default_values):
    model = XGBoostRegressor(n_estimators=10)

    model.fit(xs, ys, eval_set=[(xs_test, ys_test)])

    preds = model.predict(xs_test)

    assert len(preds) == 6

@pytest.mark.parametrize("model_class", [
    PyTorchLSTMDirect,
    PyTorchLSTMAutoregressive,
    PyTorchTransformer,
    PyTorchMLPRegressor
])
def test_fit(model_class, xs, ys, xs_test, ys_test, torch_params, mocked_default_values):
    model = model_class(**torch_params)

    if model_class == PyTorchLSTMAutoregressive:
        eval_set = None
    else:
        eval_set = [(xs_test, ys_test)]

    model.fit(xs, ys, eval_set=eval_set)

    preds = model.predict(xs_test)

    assert len(preds) == 6
