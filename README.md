# ECO-BSA-472-Final-Project-Writing-Assignment
ECO/BSA 472 Final Project Writing Assignment

Today, you will use the products.csv dataset, which is a simplified version of Nevo's (2000) fake cereal data with less information and fewer derived columns. The data were motivated by real grocery store scanner data, but due to the proprietary nature of this type of data, the provided data are not entirely real. This dataset has been used as a standard example in much of the literature on BLP estimation.

The data contains information about 24 breakfast cereals across 94 markets. Each row is a product-market pair. Each market has the same set of breakfast cereals, although with different prices and quantities. The columns in the data are as follows.

Compared to typical datasets, the number of observations in this example dataset is quite small. This helps with making these exercises run very fast, but in practice one would want more data than just a couple thousand data points to estimate a flexible model of demand. Typical datasets will also include many more product characteristics. This one only includes a couple to keep the length of the exercises manageable.

## Part 1 Download Anaconda and Python

You should install PyBLP on top of the [Anaconda Distribution](https://www.anaconda.com/). Anaconda comes pre-packaged with all of PyBLP's dependencies and many more Python packages that are useful for statistical computing. Steps:

1. [Install Anaconda](https://docs.anaconda.com/free/anaconda/install/) if you haven't already. You may wish to [create a new environment](https://docs.anaconda.com/free/anacondaorg/user-guide/work-with-environments/) .
2. [Install PyBLP](https://github.com/jeffgortmaker/pyblp#installation). 

If you're using Python, you have two broad options for how to do the coding exercises.

- Use a [Jupyter Notebook](https://jupyter.org/install#jupyter-notebook). In general, notebooks are a good way to weave text and code for short exercises, and to distribute quick snippets of code with others.



## Part 2 Open Anaconda, Open JupyterLabs, Open Python3 Notebook, Install BLP, 

Open Anaconda, select the Home button on the left menu, launch JupyterLab. DO NOT connect to a cloud!

Click the blue tab plus sign located beneath file. Next, click on Python 3 under Notebook. You are ready to type codes into the command line.

You can install the current release of PyBLP with ::

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

To define the data and view the first five observations run the following code::

    product_data = pd.read_csv('products.csv')
    product_data.head()

    product_data.sample()
    product_data.describe()
    
To view a sample observation of the data and get a quick summary of the data, run the following code::

    product_data.sample()
    product_data.describe()


## Part 4 Market Shares Computation
We want to transform observed quantities $q_{jt}$ into market shares $s_{jt} = q_{jt} / M_t$.
We first need to define a market size $M_t$. We'll assume that the potential number of servings sold in a market is the city's total population multiplied by 90 days in the quarter. Create [a new column] called `market_size` equal to `city_population` times `90`. Note that this assumption is somewhat reasonable but also somewhat arbitrary. Perhaps a sizable portion of the population in a city would never even consider purchasing cereal. Or perhaps those who do tend to want more than one serving per day.  ::

    product_data["market_size"] = product_data["city_population"] * 90

    product_data.head()
    
Next, compute a new column `market_share` equal to `servings_sold` divided by `market_size`. This gives our market shares $s_{jt}$. We'll also need the outside share $s_{0t} = 1 - \sum_{j \in J_t} s_{jt}$. Create a new column `outside_share` equal to this expression.  ::

    product_data["market_share"] = product_data["servings_sold"] / product_data["market_size"]
    product_data.head()
    
Compute summary statistics for your inside and outside shares. If you computed market shares correctly, the smallest outside share should be $s_{0t} \approx 0.305$ and the largest should be $s_{0t} \approx 0.815$.::

    grouped_market_shares = product_data.groupby("market")["market_share"]
    within_market_sum_inside_shares = grouped_market_shares.transform("sum")
    product_data["outside_share"] = 1 - within_market_sum_inside_shares
    product_data.head()

    product_data.describe()

    
## Part 5 Estimate the pure logit model with OLS::
Recall the pure logit estimating equation: $\log(s_{jt} / s_{0t}) = \delta_{jt} = \alpha p_{jt} + x_{jt}' \beta + \xi_{jt}$. 
First, create a new column `logit_delta` equal to the left-hand side of this expression. I used [`np.log`] to compute the log.

        product_data["logit_delta"] = np.log(product_data["market_share"] /product_data["outside_share"])
        product_data.head()


Then, run an OLS regression of `logit_delta` on a constant, `mushy`, and `price_per_serving`. T To use robust standard errors, you can specify `cov_type='HC0'` in [`OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html).

        from statsmodels.formula.api import ols

        mdlols = ols("logit_delta ~ 1 + mushy + price_per_serving", data=product_data) ## model object
        mdlols = mdlols.fit(cov_type="HC0") ## model fitting
        print(mdlols.params)  ## model parameters
        
        

Interpret your estimates. Your coefficient on `price_per_serving` should be around `-7.48`. In particular, can you re-express your estimate on `mushy` in terms of how much consumers are willing to pay for `mushy`, using your estimated price coefficient?


## Part 6 Estimate the pure logit model with PyBLP::      
For the rest of the exercises, we'll use PyBLP do to our demand estimation. This isn't necessary for estimating the pure logit model, which can be done with linear regressions, but using PyBLP allows us to easily run our price cut counterfactual.

PyBLP requires that some key columns have specific names so that they can be understood by PyBLP.

        product_data_renamed = product_data.rename(columns={"market": "market_ids", "product": "product_ids", "market_share":"shares", "price_per_serving":"prices"})
        product_data_renamed.head()


By default, PyBLP treats `prices` as endogenous, so it won't include them in its matrix of instruments. But the "instruments" for running an OLS regression are the same as the full set of regressors. So when running an OLS regression and not account for price endogeneity, we'll "instrument" for `prices` with `prices` themselves. We can do this by creating a new column `demand_instruments0` equal to `prices`. PyBLP will recognize all columns that start with `demand_instruments` and end with `0`, `1`, `2`, etc., as "excluded" instruments to be stacked with the exogenous characteristics to create the full set of instruments.

With the correct columns in hand, we can initialize our [`pyblp.Problem`]. 

        product_data_renamed["demand_instruments0"] = product_data_renamed["prices"]
        product_data_renamed.head()

        ols_problem = pyblp.Problem(pyblp.Formulation('1 + mushy + prices'), product_data_renamed)
        print(ols_problem)

If you `print(ols_problem)`, you'll get information about the configured problem. There should be 94 markets (`T`), 2256 observations (`N`), 3 product characteristics (`K1`), and 3 total instruments (`MD`).

To estimate the configured problem, use [`.solve`]. Use `method='1s'` to just do 1-step GMM instead of the default 2-step GMM. In this case, this will just run a simple linear OLS regression. The full code should look like the following.
        
        ols_results = ols_problem.solve(method='1s') # 1 step gmm
        print(ols_results)

Again, if you `print(ols_results)`, you'll get estimates from the logit model. Make sure that your estimates are the same as those you got from your OLS regression. If you used `'HC0'` standard errors like suggested above, you standard errors should also be the same.

      

## Part 7 Add market and product fixed effects

Since we expect price $p_{jt}$ to be correlated with unobserved product quality $\xi_{jt}$, we should be worried that our estimated $\hat{\alpha}$ on price is biased. Since we have multiple observations per market and product, and prices vary both across and within markets, it is feasible for us to add both market and product fixed effects. If $\xi_{jt} = \xi_j + \xi_t + \Delta\xi_{jt}$ and most of the correlation between $p_{jt}$ and $\xi_{jt}$ is due to correlation between $p_{jt}$ and either $\xi_j$ (product fixed effects) or $\xi_t$ (market fixed effects), then explicitly accounting for these fixed effects during estimation should help reduce the bias of our $\hat{\alpha}$.

The simplest way to add fixed effects is as dummy variables. Since there are only 24 products and 94 markets for a total of 118 fixed effects, this approach is actually feasible in this case. But in a more realistic dataset with hundreds or thousands of products and markets, running an OLS regression with this many dummy variables starts to become computationally infeasible.

The alternative, which we'll do today, is to "absorb" the fixed effects. For a single fixed effect, we could just de-mean our outcome variable and each of our regressors within the fixed effects levels, and then run our regression. For multiple fixed effects, we need to *iteratively* de-mean. PyBLP does this automatically if you specify `absorb='C(market_ids) + C(product_ids)'` in your formulation instead of adding these as dummy variables.

Since `mushy` is always either 1 or 0 for the same product across different markets, it's collinear with product fixed effects, and you can drop `mushy` from your formula. Similarly, you can drop the constant. After dropping these, re-create your problem with absorbed fixed effects and re-solve it. Compare the new $\hat{\alpha}$ estimate with the last one. You should now be getting a coefficient on price of around `-28.6`. Does its change suggest that price was positively or negatively correlated with unobserved product-level/market-level quality?

        ols_problem_absfixedeff = pyblp.Problem(pyblp.Formulation('prices', absorb='C(market_ids) + C(product_ids)'), product_data_renamed)  ## add fixed effects 
        ols_results_absfixedeff = ols_problem_absfixedeff.solve(method='1s')
        print(ols_results_absfixedeff)


## Part 8 Add an instrument for price
Adding market and product fixed effects can be helpful, but since unobserved quality typically varies by both product *and* market, we really want to instrument for prices. The data comes with a column `price_instrument` that we should interpret as a valid instrument for price that satisfies the needed exclusion restriction. It could be a cost-shifter, a valid Hausman instrument, or similar.

Before using it, we should first run a first-stage regression to make sure that it's a relevant instrument for price. To do so, use the same package you used above to run an OLS regression to run a second OLS regression of prices on `price_instrument` and your market and product fixed effects. Does `price_instrument` seem like a relevant instrument for `prices`?


        mdlolsfirststage = ols("prices ~ price_instrument + C(market_ids) + C(product_ids)", data=product_data_renamed) ## model object
        mdlolsfirststage = mdlolsfirststage.fit(cov_type="HC0") ## model fitting
        print(mdlolsfirststage.params)  ## model parameters

        print(mdlolsfirststage.summary())

Now that we've checked relevance, we can set our `demand_instruments0` column equal to `price_instrument`, re-create the problem, and re-solve it. You should get a new coefficient on price of around `-30.6`. Does the change in $\hat{\alpha}$ suggest that price was positively or negatively correlated with $\Delta\xi_{jt}$ in $\xi_{jt} = \xi_j + \xi_t + \Delta\xi_{jt}$?        

        product_data_renamed["demand_instruments0"] = product_data_renamed["price_instrument"]
        product_data_renamed.head()

        problem_costshiftinstr = pyblp.Problem(pyblp.Formulation('prices', absorb='C(market_ids) + C(product_ids)'), product_data_renamed)
        problem_costshiftinstr = problem_costshiftinstr.solve(method='1s')
        print(problem_costshiftinstr)

## Part 9 Counterfacutal: Cut a price in half and see what happens
Now that we have our pure logit model estimated, we can run our counterfactual of interest: what if we halved an important product's price? We'll select a single market, the most recent quarter in the first city: `C01Q2`. Create a new dataframe called `counterfactual_data` data for just that market and inspect the data. We'll pretend that we're firm one, and deciding whether we want to cut the price of our brand four's product `F1B04`. In particular, we might be worried about *cannibalization*, i.e. how much this price cut will result in consumers of our other 8 brands of cereal in this market just substituting from their old choice to the new, cheaper cereal. Alternatively, we could be a regulator or academic interested in how taxing that product would affect demand in the market.

In your new dataframe with just data from `C01Q2`, create a `new_prices` column that is the same as `prices` but with the price of `F1B04` cut in half. To do this, you could use [`DataFrame.loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html). Then, use [`.compute_shares`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_shares.html) on your results from the last question, passing `market_id='C01Q2'` to only compute new market shares for our market of interest, and passing `prices=counterfactual_data['new_prices']` to specify that prices should be set to the new prices. This function will re-compute market shares at the changed prices implied by the model's estimates. Store them in a `new_shares` column.
    
        counterfactual_data = product_data_renamed[product_data_renamed["market_ids"] =="C01Q2"]
        counterfactual_data.head()
        
        counterfactual_data["new_prices"] = counterfactual_data["prices"]
        rule = counterfactual_data["prices"] * 0.5
        mask = counterfactual_data['product_ids'] == "F1B04"
        counterfactual_data.loc[mask, 'new_prices'] = rule
        counterfactual_data.head(20)


Compute the percent change in shares for each product in the market. From firm one's perspective, do the estimates of cannibalization make sense. That is, do the signs on the percent changes for product `F1B04` and for other products make sense? Would you normally expect percent changes for other products to be different depending on how other products compare to the one whose price is being changed?
        
        new_shares = problem_costshiftinstr.compute_shares(market_id='C01Q2', prices=counterfactual_data['new_prices'])
        new_shares

        counterfactual_data["new_shares"] = new_shares
        counterfactual_data.head()

        new_minus_old = counterfactual_data["new_shares"] - counterfactual_data["shares"]
        counterfactual_data["percentage_change_in_shares"]= new_minus_old / counterfactual_data["shares"]


        counterfactual_data.head()


## Part 10 Compute demand elasticities

To better understand what's going on, use [`.compute_elasticities`) again specifying `market_id='C01Q2'`, to compute price elasticities for our market of interest. These measure what the model predicts will happen to demand in percentage terms when there's a 1% change in price of a product. The diagonal elements are own-price elasticities and the off-diagonal elements are cross-price elasticities. Does demand seem very elastic? Do the cross-price elasticities seem particularly reasonable?

        elasticities = problem_costshiftinstr.compute_elasticities(market_id='C01Q2')
        elasticities

        import matplotlib as plt
        plt.colorbar(plt.matshow(elasticities))

        
        
        
    





