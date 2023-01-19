from linear_regression import LinRegression
import pytest
import numpy as np

x_testing_data = np.array([1, 2, 3])
y_testing_data = np.array([2, 4, 6])
test_obj1 = LinRegression(0, 0, 2, 1)

obj_attr_error_test_cases = [
    ("hello", 1, 2, 3, TypeError),
    (1, "y", 3, 4, TypeError),
    (1, 2, "x", 4, TypeError),
    (1, 2, 3, "z", TypeError),
    (4, 5, -3, 7, ValueError),
    (3, 1, 5, -9, ValueError)
]

@pytest.mark.parametrize(
    """test_theta0, test_theta1, test_convergence_threshold, test_learning_rate, 
    expected_error""", obj_attr_error_test_cases
    )

def test_error_type_obj_attr(
    test_theta0, test_theta1, test_convergence_threshold, 
    test_learning_rate, expected_error
    ):

    with pytest.raises(expected_error):
        LinRegression(test_theta0, test_theta1, test_convergence_threshold, test_learning_rate)

fit_model_error_test_cases = [
    (np.asarray([]), np.array([1, 2, 3]), ValueError),
    (np.array([1, 2, 3]), np.asarray([]), ValueError),
    ([1, 2, 3], np.array([4, 5, 6]), TypeError),
    (np.array([1, 2, 3]), [1, 2, 3], TypeError),
    (np.asarray([1, 2, "hello"]), np.array([1, 2, 3]), TypeError),
    (np.array([1, 2, 3]), np.asarray([4, 5, "x"]), TypeError)
]

@pytest.mark.parametrize("test_input_x, test_input_y, expected_error", fit_model_error_test_cases)
def test_error_type_fit_model(test_input_x, test_input_y, expected_error):
    with pytest.raises(expected_error):
        test_obj1.fit_model(test_input_x, test_input_y)


# testing calculate_RSS

RSS_value_test_cases = [
    (np.array([1, 1]), 2),
    (np.array([0, 0]), 0),
    (np.array([2, 3]), 13)
]

@pytest.mark.parametrize("test_residuals, expected_RSS", RSS_value_test_cases)
def test_RSS_values(test_residuals, expected_RSS):
    assert test_obj1.calculate_RSS(test_residuals) == expected_RSS

RSS_type_test_cases = [
    (np.array([1, 1]), (np.intc, np.float64)),
    (np.array([1.0, 1.0]), (np.intc, np.float64)),
    (np.zeros(2), (np.intc, np.float64))
]

@pytest.mark.parametrize("test_residuals, expected_type", RSS_type_test_cases)
def test_RSS_type(test_residuals, expected_type):
    assert isinstance(test_obj1.calculate_RSS(test_residuals), expected_type)

#testing calculate_TSS

TSS_value_test_cases = [
    (np.zeros(2), 0),
    (np.array([1, 1]), 0),
    (np.array([1, 2, 3]), 2),
    (np.array([-2, -4, -6]), 8),
    (np.array([-10, 15, -20]), 650),
    (np.array([2.5, 2]), 0.125)
]

@pytest.mark.parametrize("test_input_y, expected_TSS", TSS_value_test_cases)
def test_TSS_values(test_input_y, expected_TSS):
    assert test_obj1.calculate_TSS(test_input_y) == expected_TSS

TSS_type_test_cases = [
    (np.array([1, 1]), (np.intc, np.float64, int, float)),
    (np.array([1.0, 1.0]), (np.intc, np.float64, int, float)),
    (np.zeros(2), (np.intc, np.float64, int, float))
]

@pytest.mark.parametrize("test_input_y, expected_type", TSS_type_test_cases)
def test_TSS_type(test_input_y, expected_type):
    assert isinstance(test_obj1.calculate_TSS(test_input_y), expected_type)


#testing calculate_r_squared

r_squared_value_test_cases = [
    (1, 1, 0),
    (2, 1, -1),
    (3.5, 4, 0.125)
]

@pytest.mark.parametrize("test_RSS, test_TSS, expected_r_squared", r_squared_value_test_cases)
def test_rsquared_values(test_RSS, test_TSS, expected_r_squared):
    assert test_obj1.calculate_r_squared(test_RSS, test_TSS) == expected_r_squared

#testing mean_squared_error

MSE_value_test_cases = [
    (np.array([1, 1]), np.array([1, 1]), 0.5),
    (np.zeros(2), np.zeros(2), 0),
    (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]), 3.75),
    (np.array([1]), 2, 2)
]

@pytest.mark.parametrize("test_input_x, test_residuals, expected_MSE", MSE_value_test_cases)
def test_MSE_values(test_input_x, test_residuals, expected_MSE):
    assert test_obj1.mean_squared_error(test_input_x, test_residuals) == expected_MSE

MSE_type_test_cases = [
    (np.array([1, 1]), np.array([1, 1]), (int, np.intc, float, np.float64)),
    (np.array([1.0, 1.0]), np.array([1.0, 1.0]), (int, np.intc, float, np.float64)),
    (np.zeros(2), np.zeros(2), (int, np.intc, float, np.float64))
]

@pytest.mark.parametrize("test_input_x, test_residuals, expected_type", MSE_type_test_cases)
def test_MSE_type(test_input_x, test_residuals, expected_type):
    assert isinstance(test_obj1.mean_squared_error(test_input_x, test_residuals), expected_type)

