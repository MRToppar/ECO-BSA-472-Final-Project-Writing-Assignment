# ECO-BSA-472-Final-Project-Writing-Assignment
ECO/BSA 472 Final Project Writing Assignment

Today, you will use the products.csv dataset, which is a simplified version of Nevo's (2000) fake cereal data with less information and fewer derived columns. The data were motivated by real grocery store scanner data, but due to the proprietary nature of this type of data, the provided data are not entirely real. This dataset has been used as a standard example in much of the literature on BLP estimation.

The data contains information about 24 breakfast cereals across 94 markets. Each row is a product-market pair. Each market has the same set of breakfast cereals, although with different prices and quantities. The columns in the data are as follows.

Compared to typical datasets, the number of observations in this example dataset is quite small. This helps with making these exercises run very fast, but in practice one would want more data than just a couple thousand data points to estimate a flexible model of demand. Typical datasets will also include many more product characteristics. This one only includes a couple to keep the length of the exercises manageable.

## Part 1 Download Anaconda and Python

You should install PyBLP on top of the [Anaconda Distribution](https://www.anaconda.com/). Anaconda comes pre-packaged with all of PyBLP's dependencies and many more Python packages that are useful for statistical computing. Steps:

1. [Install Anaconda](https://docs.anaconda.com/free/anaconda/install/) if you haven't already. You may wish to [create a new environment](https://docs.anaconda.com/free/anacondaorg/user-guide/work-with-environments/) for just these exercises, but this isn't strictly necessary.
2. [Install PyBLP](https://github.com/jeffgortmaker/pyblp#installation). On the Anaconda command line, you can run the command `pip install pyblp`.

If you're using Python, you have two broad options for how to do the coding exercises.

- Use a [Jupyter Notebook](https://jupyter.org/install#jupyter-notebook). The solutions to each exercise will be in a notebook. In general, notebooks are a good way to weave text and code for short exercises, and to distribute quick snippets of code with others.



## Part 2 Open Anaconda, Open JupyterLabs, Open Python3 Notebook, Install BLP, 

You can install the current release of PyBLP with `pip <https://pip.pypa.io/en/latest/>`_::

    pip install pyblp
    
    import pyblp
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    pyblp.options.digits = 2
    pyblp.options.verbose = False
    pyblp.__version__

## Part 3 Upload Data, Define Data and View 5 observations::

Click the file products.csv under README.md and download the file to your computer.

To upload data into the Anaconda environment, click the upload icon (second to the right-side of the blue plus tab).

To define the data and view the first five observations run the following code:

    product_data = pd.read_csv('products.csv')
    product_data.head()

    product_data.sample()
    product_data.describe()
    
To view a sample observation of the data and get a quick summary of the data, run the following code:

    product_data.sample()
    product_data.describe()


    
