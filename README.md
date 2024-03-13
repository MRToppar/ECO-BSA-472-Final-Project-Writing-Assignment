# ECO-BSA-472-Final-Project-Writing-Assignment
ECO/BSA 472 Final Project Writing Assignment


## Part 1 Download Anaconda and Python
`Click here for instructions on how to download Anaconda and Python:<https://saas.berkeley.edu/education/installing-python-and-anaconda#:~:text=Visit%20the%20Anaconda%20website%20and%20click%20the%20Windows%20icon.,for%20me%20only%20before%20continuing.>`


## Part 2 Open Anaconda, Open JupyterLabs, Open Python3 Notebook, Install BLP, 

You can install the current release of PyBLP with `pip <https://pip.pypa.io/en/latest/>`_::

    pip install pyblp
    
    import pyblp
    import numpy as np
    import pandas as pd

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


    
