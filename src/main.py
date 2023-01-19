import yaml
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from linear_regression import LinRegression
from quarto import render

with open('config.yaml') as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

# out_filename = str(datetime.today().date()) + "regression_QA"

df = pd.read_csv(config["file_path"])
sns.pairplot(df)
plt.show()

x_data, y_data = make_regression(n_samples = 100, n_features = 1, noise = 0.4, bias = 50)
x_data = x_data.flatten()

# Create a heatmap showing correlation of different variables/columns in the dataset
# correlation = df.corr()
# sns.heatmap(correlation, cmap = "Wistia", annot = True)
# plt.show()

# humidity_data = pd.DataFrame(data = df["Humidity"])
# temperature_data = pd.DataFrame(data = df["Temperature (C)"])

# x_data = temperature_data.to_numpy().flatten()
# y_data = humidity_data.to_numpy().flatten()


learning_rate = config["learning_rate"]
convergence_threshold = config["convergence_threshold"]
theta_0 = config["theta_0"]
theta_1 = config["theta_1"]

linreg = LinRegression(theta_0, theta_1, convergence_threshold, learning_rate)
linreg.fit_model(x_data, y_data)

y_predicted = linreg.calculate_predicted_values(x_data)


# Quarto runs its own Python environment, so below we are passing any data and
# model parameters that we want to put in the report using the execture_params argument
# Alternatively we could save the data in a temporary location instead and load
# it in Quarto.
# We could also Pickle our model and load it into Quarto that way

# NumPy dtypes (arrays, numpy floats) are converted back to native types as numpy
# doesn't play well with yaml.dump() (used in quarto.render's logic)
render(
    input = "report.qmd", 
    output_format = "html",
    # output_file = out_filename, 
    execute_params = {
        "theta_0" : linreg.theta_0, 
        "theta_1" : linreg.theta_1, 
        "pipeline_runtime" : linreg.pipeline_runtime, 
        "x_data" : x_data.tolist(), 
        "y_data" : y_data.tolist(), 
        "y_predicted" : y_predicted.tolist(), 
        "residuals" : linreg.residuals.tolist(), 
        "w_stat" : linreg.w_stat, 
        "p_value" : linreg.p_value, 
        "config" : config, 
        "r_squared" : linreg.r_squared
    }
    )