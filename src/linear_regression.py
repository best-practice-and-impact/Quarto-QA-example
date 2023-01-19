import numpy as np
import time
from scipy.stats import shapiro

class LinRegression():

    def __init__(self, theta_0, theta_1, convergence_threshold, learning_rate):

        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.residuals = None
        self.w_stat = None
        self.p_value = None
        self.pipeline_runtime = None
        self.r_squared = None

    @property
    def theta_0(self):
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("theta_0 must be an int or float")
        self._theta_0 = value
        
    @property
    def theta_1(self):
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("theta_1 must be an int or float")
        self._theta_1 = value

    @property
    def convergence_threshold(self):
        return self._convergence_threshold

    @convergence_threshold.setter
    def convergence_threshold(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Convergence threshold must be an int or float")
        elif value <= 0:
            raise ValueError("Convergence threshold must be greater than zero")
        self._convergence_threshold = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Learning rate must be an int or float")
        elif value <= 0:
            raise ValueError("Learning rate must be greater than zero")
        self._learning_rate = value


    def calculate_predicted_values(self, x):
        """
        Calculates predicted values of the dependent variable, y, according to the
        straight-line equation y = mx + c for given x, m (theta_1) and c(theta_0),
        where x is an array of values

        Parameters
        -------
            x : Training data (array-like of floats)

            theta_0 : y-intercept of the model (float)

            theta_1 : gradient of the model (float)

        Returns
        -------
            y_predicted : The predicted y-values at each point x (array of floats)
        """
        y_predicted = (self.theta_1 * x) + self.theta_0

        return y_predicted

    def calculate_residuals_array(self, x, y):
        """
        Calculates the difference between predicted y-values (from the
        hypothesis function) and the actual y-values from the dataset.

        Parameters
        -------
            x : Training data (array-like of floats)

            y : The target/actual y-values at each point x (array-like of floats)

        Returns
        -------
            residuals_array : Residuals, i.e. the difference between predicted and 
            actual y-values (array of floats)
        """
        residuals_array = np.subtract(self.calculate_predicted_values(x), np.asarray(y))
        self.residuals = residuals_array

        return residuals_array

    def calculate_RSS(self, residuals_array):
        """ Calculates the residual sum of squares"""

        residuals_squared = np.square(residuals_array)
        residual_sum_squares = sum(residuals_squared)

        return residual_sum_squares

    def calculate_TSS(self, y):
        """ Calculates the total sum of squares"""

        y_mean = np.mean(y)
        y_diff = y - y_mean
        total_sum_squares = sum(np.square(y_diff))

        return total_sum_squares

    def calculate_r_squared(self, RSS, TSS):
        """ Calculates R^2, the coefficient of determination of the model"""

        r_squared = 1 - (RSS / TSS)

        return r_squared

    

    def mean_squared_error(self, x, residuals_array):
        """
        Calculates the mean squared error between the y-values predicted
        by the model, and the actual values from the dataset

        Parameters
        -------
        x : Training data (array-like of floats)

        residuals_array : Residuals, i.e. the difference between predicted and 
            actual y-values (array-like of floats)

        Returns
        -------
            mean_squared_error : the total error of the predicted
            model against the actual data (float)
        """

        difference_squared = np.square(residuals_array)
        MSE = difference_squared.sum()/(len(x) * 2)

        return MSE

    def gradient_descent(self, x, y):
        """
        Performs gradient descent on the loss function, i.e. calculates
        the partial derivatives of the loss function with respect to
        theta_0 and theta_1 and updates the values of theta_0 and theta_1
        with the aim of minimising the loss function

        Parameters
        -------
        x : Training data (array-like of floats)

        y : The target/actual y-values at each point x (array-like of floats)

        Returns
        -------
            (new_theta_0, new_theta_1) : updated model parameters (tuple 
            of floats)
        """
        # Calculating the partial derivatives of the loss function w.r.t theta_0
        # and theta_1
        residuals_array = self.calculate_residuals_array(x, y)
        dtheta_0 = residuals_array.sum() / len(x)
        dtheta_1 = np.matmul(residuals_array, x) / len(x)

        #Updating the parameter values
        new_theta_0 = self.theta_0 - (self.learning_rate * dtheta_0)
        new_theta_1 = self.theta_1 - (self.learning_rate * dtheta_1)

        return (new_theta_0, new_theta_1)

    def fit_model(self, x, y):
        """
        Performs the linear regression on the provided dataset, attempting to fit
        a straight line model to the data

        Parameters
        -------
        x : Training data (array-like of floats)

        y : The target/actual y-values at each point x (array-like of floats)
        
        """

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError(f"The input data should be numpy arrays")

        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays must not be empty")

        if not all(isinstance(number, (int, float)) for number in x):
            raise TypeError("One or more values in input array x is neither an int nor a float")

        if not all(isinstance(number, (int, float)) for number in y):
            raise TypeError("One or more values in input array y is neither an int nor a float")

        start_time = time.time()

        initial_residuals = self.calculate_residuals_array(x, y)
        diff = self.mean_squared_error(x, initial_residuals)
        print(f"Initial loss: {diff}")

        iter_count = 0

        while diff >= self.convergence_threshold:
            
            residuals = self.calculate_residuals_array(x, y)
            initial_loss = self.mean_squared_error(x, residuals)
            
            self.theta_0, self.theta_1 = self.gradient_descent(x, y)
            
            new_residuals = self.calculate_residuals_array(x, y)
            new_loss = self.mean_squared_error(x, new_residuals)
            
            diff = initial_loss - new_loss
            
            iter_count += 1

            if iter_count % 1000 == 0:
                print()
                print(f"Iteration: {iter_count}")
                print(f"Loss diff: {diff}")
                print(f"theta 0: {self.theta_0}")
                print(f"theta 1: {self.theta_1}")

        self.w_stat, self.p_value = shapiro(self.residuals)
        self.r_squared = self.calculate_r_squared(self.calculate_RSS(self.calculate_residuals_array(x, y)), self.calculate_TSS(y))

        # Converting from numpy floats to native python floats since quarto's
        # render function uses yaml.dump() when overriding params with
        # the execute_params arg, which doesn't play well with NumPy dtypes
        self.theta_0 = self.theta_0.item()
        self.theta_1 = self.theta_1.item()
        self.r_squared = self.r_squared.item()

        self.pipeline_runtime = time.time() - start_time
