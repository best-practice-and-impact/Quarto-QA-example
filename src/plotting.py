import plotly.express as px
import numpy as np
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt

def residuals_plot(y_predicted, residuals):
    """ Creates a scatter plot of the residuals against the predicted y-values"""

    fig = px.scatter(x = y_predicted, y = residuals, labels = dict(x = "Predicted y-values", y = "Residuals"), 
                    title = "Residual plot")
    fig.update_layout(paper_bgcolor = "white")
    fig.update_layout(plot_bgcolor = "white")
    fig.update_yaxes(zeroline=True, linewidth=1, zerolinecolor="Red")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="Black")

    fig.show()

def qq_plot(residuals):
    """ Creates a QQ plot to compare the distribution of the residuals to a Gaussian distribution"""
    qqplot_data = qqplot(residuals, line='s').gca().lines

    # Residuals
    fig = px.scatter(x = qqplot_data[0].get_xdata(), y = qqplot_data[0].get_ydata(), title = "Quantile-Quantile Plot")

    # Normally distributed data
    fig2 = px.line(x = qqplot_data[1].get_xdata(), y = qqplot_data[1].get_ydata())
    fig2.update_traces(line_color = "red")
    fig.add_trace(fig2.data[0])

    fig.update_layout(paper_bgcolor = "white", plot_bgcolor = "white")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="Black")

    return fig

def scatter_plot(theta_0, theta_1, x_data, y_data):
    """Creates a scatter plot of the data and plots the best fit line of the model onto it"""
    fig = px.scatter(x = x_data, y = y_data)
    my_line = np.linspace(x_data.min(), x_data.max())
    z = theta_0 + (theta_1 * my_line)
    fig2 = px.line(x = my_line, y = z)
    fig2.update_traces(line_color="red")
    fig.add_trace(fig2.data[0])

    fig.update_layout(paper_bgcolor = "white")
    fig.update_layout(plot_bgcolor = "white")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="Black")

    fig.show()