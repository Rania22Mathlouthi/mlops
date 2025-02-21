from mlops import load_data, process_data, prepare_data

def test_load_data():
    data = load_data("sample.csv")
    assert isinstance(data, pd.DataFrame)

def test_process_data():
    data = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    x, y = process_data(data, "target")
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data("sample.csv", "target")
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
