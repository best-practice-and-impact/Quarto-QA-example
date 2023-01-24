# Quarto-QA-example

Demonstrating automated quality assurance (QA) reporting using Quarto.

## Full project description

This project is a linear regression pipeline built to demonstrate how Quarto can be used in Python to automate the production of QA reports, minimising unnecessary 
hassle when it comes to storing and checking QA outputs. This makes it easier for analysis to be peer reviewed, and guarantees a clearer audit trail, helping to bring 
new code in line with the
ONS' [RAP Minimum Viable Product criteria](https://github.com/best-practice-and-impact/rap_mvp_maturity_guidance/blob/master/Reproducible-Analytical-Pipelines-MVP.md).

The report outputs the results of a handful of QA processes relating to the assumptions made by the model, namely a residual plot, a quantile-quantile (QQ) plot, a scatter plot of the data and the model itself, the p-value of the residuals, the Shapiro-Wilk test statistic, and the coefficient of correlation, $R^2$. Also output are the model's final values for the intercept and gradient, and the pipeline runtime.

At data ingest there are some basic input validation checks performed both on the dataset (type checks of the arrays and the data they contain) and on the model hyperparameters (type checks and value checks where necessary).

## Installation

1. Install the Quarto command-line interface (CLI) from the [Quarto website](https://quarto.org/docs/get-started/) or using `pip install quarto` from the command line.
If using Visual Studio, you should additionally install the Quarto VSCode extension.

2. After cloning the repository to your machine, navigate to the project directory within the terminal and install/update all package dependencies 
using `pip install -r requirements.txt`

## Usage

After cloning the repo and downloading all the dependencies, the pipeline should run unmodified by executing the main script. If desired, you may alter parameters from within the config file, where you have control over the model's initial values for the intercept and gradient, labelled $\theta_0$ and $\theta_1$ according to $y=\theta_0 + \theta_1 x$.

The convergence threshold determines when the model will stop training. At every iteration of the training loop the value of the loss function is subtracted from its previous value, and training stops when this difference reaches the convergence threshold.

The learning rate determines the size of the steps taken by the model when updating $\theta_0$ and $\theta_1$. Higher values will train faster but may reach a less precise final model.

Finally the config file allows you to select which data to run the model on. If you wish to add your own dataset, you will have to put a csv file into the data folder, and alter the code to select the specific rows/columns of the dataframe that you want to model.

Quarto currently has no easy way to specify output directories, and as a result you will find the output file (called report.html) in the src folder.

Unit tests may be run by navigating to the project's test directory in the terminal and executing the command `pytest`.

## Useful links

1. [The Duck Book](https://best-practice-and-impact.github.io/qa-of-code-guidance/intro.html)
2. [The RAP MVP criteria](https://github.com/best-practice-and-impact/rap_mvp_maturity_guidance/blob/master/Reproducible-Analytical-Pipelines-MVP.md)
3. [Quarto Getting Started page](https://quarto.org/docs/get-started/)
4. [Quarto FAQ](https://quarto.org/docs/faq/)
