import pandas as pd
import numpy as np
from mlops import load_data, process_data, prepare_data

def test_load_data():
    data = load_data("merged_data.csv")
    assert isinstance(data, pd.DataFrame)

def test_process_data():
    data = pd.DataFrame({"feature": [1, 2, 3], "Churn": [0, 1, 0]})
    x, y = process_data(data, "Churn")
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data("merged_data.csv", "Churn")
    print(f"x_train type: {type(x_train)}")
    print(f"x_test type: {type(x_test)}")
    print(f"y_train type: {type(y_train)}")
    print(f"y_test type: {type(y_test)}")
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
