import os
from datetime import datetime
import re

import yaml
from quarto import render
from sklearn.datasets import make_regression
from src.linear_regression import LinRegression

with open('config.yaml') as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

x_data, y_data = make_regression(n_samples = 100, n_features = 1, noise = 0.4, bias = 50)
x_data = x_data.flatten()

learning_rate = config["learning_rate"]
convergence_threshold = config["convergence_threshold"]
theta_0 = config["theta_0"]
theta_1 = config["theta_1"]

linreg = LinRegression(theta_0, theta_1, convergence_threshold, learning_rate)
linreg.fit_model(x_data, y_data)

y_predicted = linreg.calculate_predicted_values(x_data)

out_filename = str(datetime.today().date()) + "_regression_QA.html"

def clean_report(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

        os.chdir("./docs")

        with open('report.html', 'r', encoding='utf-8') as f:
            html = f.read()

        # Use regex to find and replace the Unicode characters
        html = re.sub('[\u2018\u2019\u201c\u201d]', lambda x: {'\u2018': "'", '\u2019': "'",
                                                               '\u201c': '"', '\u201d': '"'}[x.group()], html)

        # Open the HTML file for writing and overwrite the original file
        with open('report.html', 'w', encoding='utf-8') as f:
            f.write(html)

        if os.path.exists(out_filename):
            os.remove(out_filename)
        os.rename("report.html", out_filename)
    return wrapper

render = clean_report(render)


# Quarto runs its own Python environment, so below we are passing any data and
# model parameters that we want in the report using the execture_params argument
# Alternatively we could save the data in a temporary location instead and load
# it in Quarto.
# We could also Pickle our model and load it into Quarto that way

# NumPy dtypes (arrays, numpy floats) are converted back to native types as numpy
# doesn't play well with yaml.dump() (used in quarto.render's logic)

render(
    input = "quarto_files/report.qmd", 
    output_format = "html",
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