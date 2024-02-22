import pandas as pd
import numpy as np
import math

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import scipy.optimize as optimization
from scipy.stats import norm

import seaborn as sns

from datetime import date

import warnings
warnings.filterwarnings("ignore")

def StocksAnalyze(ticker, start = None, end = None, market = None):
    """
    Parameters
    ----------
    tickers : str
        Single or multiple tickers selected from Yahoo Finance, separated by spaces.
    start_date : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2018-01-01'.
    end_date : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.
    market : str, optional
        Market identifier. The default is None. Set to US or TR market.

    Returns
    -------
    None
        Will plot the percentage movements and risk/return, return the correlation matrix,
        along with the chart for the Sharpe Ratio of the portfolio.
    """
    ticker = ticker
    if start == None:
        start = "2018-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
    
    stocks = yf.download(ticker, start, end)
    stocks.to_csv("ES_stocks.csv")

    stocks = pd.read_csv("ES_stocks.csv", header = [0,1], index_col = [0], parse_dates = [0])
    stocks.columns = stocks.columns.to_flat_index()
    stocks.columns = pd.MultiIndex.from_tuples(stocks.columns)
    stocks.swaplevel(axis = 1).sort_index(axis = 1)
    close = stocks.loc[:, "Close"].copy()


    close.plot(figsize = (15, 8), linewidth = 3, fontsize = 13)
    plt.legend(fontsize = 13)
    plt.show()

    close.iloc[0,0]
    norm = close.div(close.iloc[0]).mul(1)
    norm["Portfolio"] = norm.mean(axis=1)
    norm.plot(figsize = (15, 8), linewidth= 3, fontsize = 13)
    plt.legend(fontsize = 13)
    plt.title("Percentage Movements", fontsize = 24)
    plt.show()

    ret = close.pct_change().dropna()
    summary = ret.describe().T.loc[:, ["mean", "std"]]
    summary["mean"] = summary["mean"] * 252
    summary["std"] = summary["std"] * np.sqrt(252)

    summary.plot.scatter(x = "std", y = "mean", figsize = (12, 8), s = 50, fontsize = 15)
    for i in summary.index:
        plt.annotate(i, xy=(summary.loc[i, "std"]+0.0005, summary.loc[i, "mean"]+0.0005), size = 12)

    plt.xlabel("ann. Risk(std)", fontsize = 18)
    plt.ylabel("ann. Return", fontsize = 18)
    plt.title("Risk/Return", fontsize = 24)
    plt.show()

    ret.corr()
    plt.figure(figsize = (12 , 8))
    sns.set(font_scale = 1.4)
    sns.heatmap(ret.corr(), cmap = "coolwarm", annot = True)
    plt.title("Correlation Matrix", fontsize = 24)
    plt.show()

    N = 255
    if market == None or market == "US":
        rf = yf.Ticker("^TNX").fast_info['last_price'] / 100
    if market == "TR":
        rf = pd.read_html("https://www.bloomberght.com/tahvil/tr-10-yillik-tahvil")[0]['SON'][10] / 10000
    sr_mean =norm.mean()*N-rf
    sr_sigma = norm.std()*np.sqrt(255)
    sharpe_ratio = sr_mean / sr_sigma / 100
    sharpe_ratio.plot.bar(figsize = (15,8), fontsize = 15)
    plt.title("Sharpe Ratio", fontsize = 24)
    plt.show()
    sns.reset_orig()
    
def MonteCarlo(ticker, start=None, end=None, decimal=None):
    """
    Parameters
    ----------
    ticker : str
        Single ticker selected from Yahoo Finance.
    start_date : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2020-01-01'.
    end_date : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.
    decimal_places : int, optional
        Digits after the decimal point. The default is None for 2 decimal places.

    Returns
    -------
    None
        Performs Monte Carlo simulation on a single ticker from Yahoo Finance.
    """
    ticker = ticker
    if start == None:
        start = "2020-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
    if decimal == None:
        decimal = 2
    else:
        decimal = decimal
    
    stock_data = yf.download(ticker, start, end)
    
    current_price = yf.Ticker(ticker).fast_info['lastPrice']
    returns = stock_data["Adj Close"].pct_change()
    daily_vol = returns.std()

    T = 252
    count = 0
    price_list = []
    last_price = stock_data["Adj Close"][-1].copy()

    price = last_price * (1 + np.random.normal(0, daily_vol))
    price_list.append(price)

    for y in range(T):
        if count == 251:
            break
        price = price_list[count] * (1+ np.random.normal(0, daily_vol))
        price_list.append(price)
        count += 1

    NUM_SIMULATIONS = 1000
    df = pd.DataFrame()
    last_price_list = []
    for x in range(NUM_SIMULATIONS):
        count = 0
        price_list = []
        price = last_price * (1 + np.random.normal(0, daily_vol))
        price_list.append(price)
        
        for y in range(T):
            if count == 251:
                break
            price = price_list[count] * (1 + np.random.normal(0, daily_vol))
            price_list.append(price)
            count += 1
            
        df[x] = price_list
        last_price_list.append(price_list[-1])
        

    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0,
                                      alpha=0)]*4
    labels = []
    labels.append("Last Price: ${}".format(round(current_price,decimal)))
    labels.append("Expected Price: ${}".format(round(np.mean(last_price_list),decimal)))
    labels.append("Quantile (5%): {}".format(round(np.percentile(last_price_list, 5),decimal)))
    labels.append("Quantile (95%): {}".format(round(np.percentile(last_price_list, 95),decimal)))             

    plt.hist(last_price_list, bins=100)
    plt.suptitle("Histogram: {}".format(ticker))
    plt.axvline(np.percentile(last_price_list, 5), color="r", linestyle="dashed", linewidth=2)
    plt.axvline(np.percentile(last_price_list, 95), color="r", linestyle="dashed", linewidth=2)
    plt.legend(handles, labels, loc="best", fontsize="small", fancybox=True, framealpha=0.7, handlelength=0, handletextpad=0)
    plt.show()
    
from tabulate import tabulate    
def VaR(ticker, start=None, end=None):
    """
    Parameters
    ----------
    ticker : str
        Single ticker selected from Yahoo Finance.
    start_date : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2021-01-01'.
    end_date : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.

    Returns
    -------
    None
        Performs Value at Risk (VaR) on a single ticker from Yahoo Finance.
    """
    ticker = ticker
    if start == None:
        start = '2021-01-01'
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
    
    asset  = yf.download(ticker, start, end)
    asset.head()

    asset_close = asset['Adj Close'].pct_change()

    asset_close.sort_values(inplace=True, ascending=True)
    # Use quantile method
    VaR_90 = asset_close.quantile(0.1).round(4) # for 90%
    VaR_95 = asset_close.quantile(0.05).round(4)# for 95%
    VaR_99 = asset_close.quantile(0.01).round(4)# for 99%

    PR_90 = (asset['Adj Close'][-1] + (asset['Adj Close'][-1] * VaR_90)).round(4) #Price Range at 90%
    PR_95 = (asset['Adj Close'][-1] + (asset['Adj Close'][-1] * VaR_95)).round(4) #Price Range at 95%
    PR_99 = (asset['Adj Close'][-1] + (asset['Adj Close'][-1] * VaR_99)).round(4) #Price Range at 99%

    print(tabulate([['90%', VaR_90, PR_90], ['95%', VaR_95, PR_95], ['99%', VaR_99, PR_99]], 
                   headers=['Confidence Level', 'Value at Risk', 'Price Range'],tablefmt='pretty'))
    
import scipy.optimize as optimization
def markowitz(ticker, start_date=None, end_date=None):
    """
    Parameters
    ----------
    tickers : str
        Single or multiple tickers selected from Yahoo Finance, separated by spaces.
    start_date : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2021-01-01'.
    end_date : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.

    Returns
    -------
    None
        Creates an Efficient Frontier based on tickers selected from Yahoo Finance.
        Only for Call options.
    """
    # on average there are 252 trading days in a year
    NUM_TRADING_DAYS = 252
    # we will generate random w (different portfolios)
    NUM_PORTFOLIOS = 10000
    
    ticker = ticker
    ticker = "".join(str(ticker).split(','))
    stocks = []
    stocks = [item for item in ticker.split()]
    if start_date == None:
        start_date = '2021-01-01'
    else:
        start_date = start_date
    if end_date == None:
        end_date = date.today()
    else:
        end_date = end_date
        
    def download_data():
        # name of the stock (key) - stock values (2010-1017) as the values
        stock_data = {}

        for stock in stocks:
            # closing prices
            ticker = yf.Ticker(stock)
            stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

        return pd.DataFrame(stock_data)


    def show_data(data):
        data.plot(figsize=(10, 5))
        plt.show()


    def calculate_return(data):
        # NORMALIZATION - to measure all variables in comparable metric
        log_return = np.log(data / data.shift(1))
        return log_return[1:]


    def show_statistics(returns):
        # instead of daily metrics we are after annual metrics
        # mean of annual return
        print(returns.mean() * NUM_TRADING_DAYS)
        print(returns.cov() * NUM_TRADING_DAYS)


    def show_mean_variance(returns, weights):
        # we are after the annual return
        portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()
                                                                * NUM_TRADING_DAYS, weights)))
        print("Expected portfolio mean (return): ", portfolio_return)
        print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)


    def show_portfolios(returns, volatilities):
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()


    def generate_portfolios(returns):
        portfolio_means = []
        portfolio_risks = []
        portfolio_weights = []

        for _ in range(NUM_PORTFOLIOS):
            w = np.random.random(len(stocks))
            w /= np.sum(w)
            portfolio_weights.append(w)
            portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
            portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()
                                                              * NUM_TRADING_DAYS, w))))

        return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


    def statistics(weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()
                                                                * NUM_TRADING_DAYS, weights)))
        return np.array([portfolio_return, portfolio_volatility,
                         portfolio_return / portfolio_volatility])


    # scipy optimize module can find the minimum of a given function
    # the maximum of a f(x) is the minimum of -f(x)
    def min_function_sharpe(weights, returns):
        return -statistics(weights, returns)[2]


    # what are the constraints? The sum of weights = 1 !!!
    # f(x)=0 this is the function to minimize
    def optimize_portfolio(weights, returns):
        # the sum of weights is 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # the weights can be 1 at most: 1 when 100% of money is invested into a single stock
        bounds = tuple((0, 1) for _ in range(len(stocks)))
        return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns
                                     , method='SLSQP', bounds=bounds, constraints=constraints)


    def print_optimal_portfolio(optimum, returns):
        print("Optimal portfolio: ", optimum['x'].round(3))
        print("Expected return, volatility and Sharpe ratio: ",
              statistics(optimum['x'].round(3), returns))


    def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
        plt.show()
    #Call functions to execute
    dataset = download_data()
    show_data(dataset)
    log_daily_returns = calculate_return(dataset)
    # show_statistics(log_daily_returns)

    pweights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)
    optimum = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
 
def CAPM(ticker, start_date=None, end_date=None, market=None):
    """
    Parameters
    ----------
    ticker : str
        Single or multiple tickers selected from Yahoo Finance, separated by spaces.
    start_date : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2021-01-01'.
    end_date : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.
    market : str, optional
        Market identifier. The default is "US". It can also be selected as "TR".

    Returns
    -------
    None
        Plots the Capital Asset Pricing Model (CAPM).
        Based on both the formula and linear regression.
    """
    
    ticker = ticker
    ticker = "".join(str(ticker).split(','))
    stocks = []
    stocks = [item for item in ticker.split()]
    if start_date == None:
        start_date = '2021-01-01'
    else:
        start_date = start_date
    if end_date == None:
        end_date = date.today()
    else:
        end_date = end_date
    def download_data():
        data = {}
        
        for stock in stocks:
            ticker = yf.download(stock, start_date, end_date)
            data[stock] = ticker["Adj Close"]
            
        return pd.DataFrame(data)
    
    def initialize():
        stock_data = download_data()
        stock_data = stock_data.resample("M").last()
        
        data = pd.DataFrame({'s_adjclose': stock_data[stocks[0]],
                                  'm_adjclose': stock_data[stocks[1]]})
        
        #logarithmic monthly returns
        data[['s_returns', "m_returns"]] = np.log(data[['s_adjclose', 'm_adjclose']] /
                                                       data[['s_adjclose', 'm_adjclose']].shift(1))
        
        data = data[1:]
        return data
    
    def calculate_beta():
        
        data = initialize()
        covariance_matrix = np.cov(data['s_returns'], data['m_returns'])
        #calculate beta based on the formula
        beta = covariance_matrix[0,1] / covariance_matrix[1,1]
        print("Beta from formula: ", beta)
    
    def regression():
        
        #using linear regression to fit a line to the data, where beta is our slope
        data = initialize()
        beta, alpha = np.polyfit(data['m_returns'], data['s_returns'], deg=1) #degree is one for linear line
        print("Beta from regression: ", beta)
        expected_return = RISK_FREE_RATE + beta * (data['m_returns'].mean() * MONTHS_IN_YEAR - RISK_FREE_RATE)
      
        print("Expected Return(Intercept): ", expected_return)
        plot_regression(data, alpha, beta)
    
    def plot_regression(data, alpha,beta):
        fig, axis = plt.subplots(1, figsize=(8,5))
        axis.scatter(data['m_returns'], data['s_returns'], label='Data Points')
        axis.plot(data['m_returns'], beta * data['m_returns'] + alpha, color='red', label="CAPM Line")
        plt.title("Capital Asset Pricing Model")
        plt.xlabel('Market return $R_m$', fontsize=14)
        plt.ylabel('Stock return $R_a$')
        plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def get_risk_free_rate(start_date, end_date, market):
        if market == "US":
            tnx_data = yf.download("^TNX", start=start_date, end=end_date)
            avg_yield = tnx_data['Close'].mean() / 100
        elif market == "TR":
            avg_yield = pd.read_html("https://www.bloomberght.com/tahvil/tr-10-yillik-tahvil")[0]['%'][10] / 10000
        else:
            raise ValueError("Invalid market identifier")
        return avg_yield
       
    
    RISK_FREE_RATE = get_risk_free_rate(start_date, end_date, market if market else "US")   
    MONTHS_IN_YEAR = 12
    calculate_beta()
    regression()  
    
def TFEF(ticker, start_date=None, end_date=None):
    """
    Parameters
    ----------
    tickers : str
        Single or multiple tickers selected from Yahoo Finance, separated by spaces.
    start_date : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2021-01-01'.
    end_date : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.

    Returns
    -------
    None
        Plots the Two-Funds Efficient Frontier.
        Two different optimizations for 20% and 28% returns.
        Available for both call and put options.
    """
    ticker = ticker
    ticker = "".join(str(ticker).split(','))
    stocks = []
    stocks = [item for item in ticker.split()]
    if start_date == None:
        start_date = '2021-01-01'
    else:
        start_date = start_date
    if end_date == None:
        end_date = date.today()
    else:
        end_date = end_date
    
   
    def download_data():
        # name of the stock (key) - stock values (2010-1017) as the values
        stock_data = {}

        for stock in stocks:
            # closing prices
            ticker = yf.Ticker(stock)
            stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

        return pd.DataFrame(stock_data)
    
    df = download_data()
    df = np.log(df).diff()
    df = df.dropna()
    
    def expected_return(df):
        expected_returns = []
        for i in df.columns:
            expected_returns.append(df[i].mean()*252)
        return np.array(expected_returns)
        
    df_cov = df.cov()*252
    #correlation matrix

    #Inverse matrix of the variance-covariance matrix
    df_cov_inv = pd.DataFrame(np.linalg.pinv(df_cov.values), df_cov.columns, df_cov.index)

    #plotting the correlation matrix
    correlation_mat = df.corr()

    def plot_corr_matrix(correlation_mat):
        plt.figure(figsize=(12.2, 4.5))
        sns.heatmap(correlation_mat, cmap = "coolwarm", annot = True)
        plt.title('Correlation Matrix', fontsize = 18)
        plt.xlabel('Stocks', fontsize = 14)
        plt.ylabel('Stocks', fontsize = 14)
        plt.show()
    plot_corr_matrix(correlation_mat)
    
    returns = expected_return(df)
    #Create a vector of 1 with a length equal to the quantity of stocks
    vector_ones = np.ones(len(stocks))
    #Then calculate a vector(@ is the dot multiplication)
    a = vector_ones@df_cov_inv@returns
    b = returns.T@df_cov_inv@returns
    c = vector_ones.T@df_cov_inv@vector_ones
    d = b*c - a**2

    #Finally calculate g and h to have all the optimal weights
    g = 1/d * (b*df_cov_inv@vector_ones - a*df_cov_inv@returns)
    h = 1/d * (c*df_cov_inv@returns - a*df_cov_inv@vector_ones)

    #Optimal weights for minimal variance, expecting 20% returns, shortselling allowed
    weights1 = g + h*0.2
    #Optimal weights for minimal variance, expecting 28% returns, shortselling allowed
    weights2 = g + h*0.28
    
    # This function gives us a list with values between any given value inside a list:
    def alpha(list):
      for i in range(1000):
          list.append(list[i] + 1/100)
      return list
    # Let's calculate a list with values between -5 and 5:
    alpha = alpha([-5])

    # Let's use this function to generate the optimal solution from the two solutions calculated before:
    def w3(weight1,weight2,alpha):
      weight3 = []
      for i in alpha:
        weight3.append(i*weight1 + (1-i)*weight2)
      return np.array(weight3)
    # Now we have obtained the entire efficient frontier just from the two original optimal solutions:
    weight3 = w3(weights1,weights2,alpha)

    # Let's calculate the variance for each solution
    def var_portfolio(weights,covar_matrix):
      empty_list = []
      for i in weights:
        # We use the variance of a portfolio formula:
        empty_list.append(i.T@covar_matrix@i)
      return np.array(empty_list)

    # Now we have a list with the variance for each solution for the entire efficient frontier
    var_portafolio = var_portfolio(weight3,df_cov)

    # Now we use this function for calculating the expected return of each portfolio

    def exp_return(weights,retornos_esperados):
      empty_list = []
      for i in weights:
        # We multiply each weight by it's expected return:
        empty_list.append(i.T@retornos_esperados)
      return np.array(empty_list)

    r = exp_return(weight3,returns)

    # Plotting the efficient Frontier:

    def scatterplot(df, x_dim, y_dim):
      x = np.sqrt(var_portafolio)
      y = r
      fig, ax = plt.subplots(figsize=(10, 5))
      ax.scatter(x, y, alpha=0.70, marker ='o')
      ax.set_title('Efficient Frontier')
      ax.set_xlabel('Volatility')
      ax.set_ylabel('Returns')
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.grid(color='blue', linestyle='-', linewidth=0.25, alpha=0.5)
      plt.show()
    scatterplot(df, "Volatility", "Expected Return")
    
    # We import the libraries:
    from pypfopt.efficient_frontier import EfficientFrontier

    # Portfolio Optimization for a given return: 20% return.

    ef1 = EfficientFrontier(pd.Series(returns), df_cov,weight_bounds=(-1,1))
    weights = ef1.efficient_return(0.2, market_neutral=False)
    cleaned_weights = ef1.clean_weights() 
    print(cleaned_weights) 
    ef1.portfolio_performance(verbose=True)

    # Portfolio Optimization for 28% return:

    ef1 = EfficientFrontier(pd.Series(returns), df_cov,weight_bounds=(-1,1))
    weights = ef1.efficient_return(0.28, market_neutral=False)
    cleaned_weights = ef1.clean_weights() 
    print(cleaned_weights) 
    ef1.portfolio_performance(verbose=True)
    
def PortSharpe(ticker_weights, start = None, end = None, market = None):
    """
    Parameters
    ----------
    ticker_weights : dict
        Dictionary of tickers and their corresponding weights in the portfolio.
    start : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2018-01-01'.
    end : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.
    market : str, optional
        Market identifier. Set to 'US' or 'TR'. The default is None.

    Returns
    -------
    None
        Calculates the Sharpe ratio of a portfolio based on given weights.
        Then creates a pie chart.
    """
        
    if start == None:
        start = "2018-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
        
    if market == None or market == "US":
        rf = yf.Ticker("^TNX").fast_info['last_price'] / 100
    if market == "TR":
        rf = pd.read_html("https://www.bloomberght.com/tahvil/tr-10-yillik-tahvil")[0]['SON'][10] / 10000
    
    # Obtain the tickers and weights from the ticker_weights dictionary
    tickers = list(ticker_weights.keys())
    weights = np.array(list(ticker_weights.values()))
    
    # Download the historical data
    data = yf.download(tickers, start, end)
    
    # Calculate daily returns
    close = data['Adj Close']
    daily_rets = close.pct_change()
    
    # Calculate portfolio's returns
    port_returns = daily_rets.dot(weights)
       
    # Calculate Sharpe ratio
    sr_mean = port_returns.mean()*255-rf
    sr_sigma = port_returns.std()*np.sqrt(255)
    sharpe_ratio = sr_mean / sr_sigma
    
    print(' Sharpe Ratio for Portfolio:', sharpe_ratio.round(2), "\n",
         'Annual Volatility:', (sr_sigma*100).round(2),'%', "\n",
         'Expected Annual Returns:', (sr_mean*100).round(2),'%')
    
    # Data for the pie chart
    labels = ticker_weights.keys()
    sizes = ticker_weights.values()

    # Plotting the pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')

    # Aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Display the chart
    plt.show()
    
from statsmodels.tsa.filters.hp_filter import hpfilter
def hpfma(ticker, start=None, end=None):
    """
    Parameters
    ----------
    ticker : str
        Single ticker selected from Yahoo Finance.
    start : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2003-01-01'.
    end : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.

    Returns
    -------
    None
        Uses the Hodrick-Prescott filter on daily price to reduce noise
        and 50 to 200 moving averages to check trend.
        [Harris and Yilmaz, 2009]
    """
    ticker = ticker
    if start == None:
        start = "2003-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
    #Download the data
    data = yf.download(ticker, start, end)
    #Extract trend and cyclicity using hpfilter
    df_cycle, df_trend = hpfilter(data['Close'], lamb = 1600*3**4)
    data['Trend'] = df_trend
    data['MA50'] = df_trend.rolling(window=50).mean()
    data['MA200'] = df_trend.rolling(window=200).mean()
    data['Cycle'] = df_cycle
    
    data[['Trend', 'MA50', 'MA200']].plot(figsize=(12,5)).autoscale(axis='x',tight=True)
    data[['Cycle']].plot(figsize=(12,2)).autoscale(axis='x',tight=True)
    
import arch
def garch(ticker, start = None, end = None):
    """
    Parameters
    ----------
    ticker : str
        Single ticker selected from Yahoo Finance.
    start : str, optional
        Start Date. The default is None, and if not provided, it defaults to '2021-01-01'.
    end : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.

    Returns
    -------
    None
        Performs and plots the GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model
        on a single ticker from Yahoo Finance.
    """
    ticker = ticker
    if start == None:
        start = '2021-01-01'
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
        
    Returns = yf.download(ticker, start, end)['Adj Close'].pct_change().dropna()
    
    # Create a GARCH model and fit to data
    model = arch.arch_model(Returns, vol='GARCH', p=1, q=1, rescale=False)
    results = model.fit()
    print(results.summary())
    
    # Plot the standardized residuals
    results.plot(annualize='D')
    plt.show()
    
def PortGarch(ticker_weights, start = None, end = None):
    """
    Parameters
    ----------
    ticker_weights : dict
        Dictionary of tickers and their corresponding weights in the portfolio.
    start : str, optional
        Start Date. The default is None, and if not provided, it defaults to "2018-01-01".
    end : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.
    market : str, optional
        Market identifier. Set to 'US' or 'TR'. The default is None.

    Returns
    -------
    None
        Calculates and plots the GARCH for a portfolio based on given weights.
    """
        
    if start == None:
        start = "2018-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
        
    # Obtain the tickers and weights from the ticker_weights dictionary
    tickers = list(ticker_weights.keys())
    weights = np.array(list(ticker_weights.values()))
    
    # Download the historical data
    data = yf.download(tickers, start, end)
    
    # Calculate daily returns
    close = data['Adj Close']
    daily_rets = close.pct_change().dropna()
    
    # Calculate portfolio's returns
    port_returns = daily_rets.dot(weights)
    
    # Create a GARCH model and fit to data
    model = arch.arch_model(port_returns, vol='GARCH', p=1, q=1, rescale=False)
    results = model.fit()
    print(results.summary())
    
    # Plot the standardized residuals
    results.plot(annualize='D')
    plt.show()

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
    
def beta_hedge(s1, W1, s2, index, start = None, end = None):    
    """
    Parameters
    ----------
    s1 : str
        Ticker for the stock in Long position.
    W1 : float
        Weight of the stock in Long position.
    s2 : str
        Ticker for the stock to be used to hedge as a Short position.
    index : str
        Ticker for the index to calculate the beta.
    start : str, optional
        Start Date. The default is None, and if not provided, it defaults to "2020-01-01".
    end : str, optional
        End Date. The default is None, and if not provided, it defaults to today's date.

    Returns
    -------
    None
        Calculates the weight for the second stock to be used as a short position to hedge the beta of the first stock.
    """
        
    if start == None:
        start = "2020-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
    
    stock1 = yf.download(s1, start, end)['Close']
    stock2 = yf.download(s2, start, end)['Close']
    index = yf.download(index, start, end)['Close']
    
    stock1_n = (stock1.div(stock1.iloc[0]).mul(100) - 100)
    stock2_n = (stock2.div(stock2.iloc[0]).mul(100) - 100)
    cum_ret = stock1_n - stock2_n
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing= 0.05, row_heights=[2,1])

    fig.add_trace(go.Scatter(x=stock1_n.index, y=stock1_n.round(2), name=s1), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock2_n.index, y=stock2_n.round(2), name=s2), row=1, col=1)

    fig.update_layout(title='Percentage Change')

    fig.add_shape(
        type="line",
        x0=stock1_n.index[0],
        y0=0,
        x1=stock1_n.index[-1],
        y1=0,
        line=dict(color="red", dash="dash")
    )

    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='% Returns', row=1, col=1, title_standoff=20)

    fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.round(2), name='Portfolio', line=dict(color='green')), row=2, col=1)

    fig.update_layout(title='Portfolio Performance', height=600, width=800, template='plotly_dark', hovermode = "x")

    fig.show()
    
    def calculate_beta(stock, index = index, start = start, end = end):
        stock_returns = stock.pct_change()[1:]
        index_returns = index.pct_change()[1:]
    
        data = pd.concat([stock_returns, index_returns], axis = 1)
        data.columns = ['Stock Returns', 'Index Returns']
    
        cov_matrix = data.cov()
    
        beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
    
        return beta
    
    def market_days(start = start, end = end):
        start = datetime.strptime(start, "%Y-%m-%d").date()
        end = end
        mdays = 0
        current_date = start
        
        while current_date <= end:
            if current_date.weekday() < 5:
                mdays += 1
            current_date += timedelta(days=1)
        return mdays
    
    W2 = (W1 * calculate_beta(stock1, index = index, start = start, end = end)
            /calculate_beta(stock2, index = index, start = start, end = end) 
            * stock1/stock2)[-1].round(2)

    print(" Beta Hedge Optimization Results \n",
          "-------------------------------------------- \n",
          f"Range of Timeframe: {start} - {end} \n",
          f"Market Days: {market_days(start, end)} \n",
          "-------------------------------------------- \n",
          f"Weight for the long stock {s1}: {W1} \n",
          f"Weight for the short stock {s2}: {W2} \n",
          f"Cumulative return: {cum_ret[-1].round(2)}%")
    
def Seasonality(ticker, start=None, end=None):
    """
    Parameters
    ----------
    ticker : str
        Single ticker selected from Yahoo Finance.
    start : str or None, optional
        Start Date. The default is None, and if not provided, it defaults to "2020-01-01".
    end : str or None, optional
        End Date. The default is None, and if not provided, it defaults to today's date.

    Returns
    -------
    pd.DataFrame
        Returns a DataFrame containing monthly data summary.

    Calculates the monthly returns and plots the seasonality chart.

    """
    if start == None:
        start = "2020-01-01"
    else:
        start = start
    if end == None:
        end = date.today()
    else:
        end = end
        
    price = yf.download(ticker, start, end)
    df = pd.DataFrame({'return': price['Close'].pct_change().fillna(0)})

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = df[df.index >= df[df.index.month == 1].index[0]]
    df = df[df.index <= df[df.index.month == 12].index[-1]]
    
    # Seasonal data
    seasonal_data = {}
    for year in df.index.year.unique():
        seasonal_data[year] = df[df.index.year == year].reset_index()['return']
    seasonal_data = pd.DataFrame(seasonal_data)
    
   # Monthly Cumulative Returns
    year_long = seasonal_data[-1:].T.dropna().index[0]
    seasonal_data.index = df[df.index.year == year_long].index.strftime('%Y%m')
    seasonal_returns = seasonal_data.dropna(how='all').groupby(seasonal_data.index).cumsum()
    seasonal_returns.reset_index(drop=True, inplace=True)
    seasonal_returns = seasonal_returns.dropna(how='all').mean(axis=1) 
    
    # Monthly Data Summary
    monthly = {}
    for year in df.index.year.unique():
        yeardf = df[df.index.year == year]
        monthly[year] = yeardf.groupby(yeardf.index.month).sum() * 100

    data = pd.concat(monthly, axis=1)
    data.columns = [col[0] for col in data.columns]
    data.index = months

    summary = pd.DataFrame(data.mean(axis=1))
    summary.columns = ['Return %']
    
    # Create a line plot using plotly.graph_objs
    fig = go.Figure()

    # Add a line trace for the summary data
    fig.add_trace(go.Scatter(
        x=summary.index,
        y=summary['Return %'].round(2),
        mode='lines+markers',
        name='Monthly Returns',
        line=dict(color='green'),
        marker=dict(size=8, color='green')
    ))

    # Set plot title and axis labels
    fig.update_layout(
        title=f'Seasonal Chart : {ticker}',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Return %'),
        template='plotly_dark'
    )

    fig.add_shape(
        type="line",
        x0=summary.index[0],
        y0=0,
        x1=summary.index[-1],
        y1=0,
        line=dict(color="red", dash="dash")
    )

    # Show the plot
    fig.show()
    
    data_df = pd.DataFrame(data.T)
    return data_df

def dollarspread(start, end=None):
    """
    Parameters
    ----------
    start : str or datetime
        Start date for fetching data.

    end : str or datetime, optional
        End date for fetching data. If not provided, the default is set to today's date.

    Returns
    -------
    None

    Calculates and plots the spread between the Dollar Index and Dollar Futures.

    """
    if end is None:
        end = date.today()

    # Download both the dollar index and futures
    index = yf.download("DX-Y.NYB", start, end)
    futures = yf.download("DX=F", start, end)

    # Check if the dataframes are empty
    if index.empty or futures.empty:
        print("Error: Data not available.")
        return

    # Create subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Dollar Index vs Dollar Futures', 'Spread between Dollar Index and Dollar Futures'],
                        row_heights=[2, 1],  # Adjust the row heights
                        vertical_spacing=0.05)  # Adjust the vertical spacing

    # Plot Dollar Index and Futures prices
    trace_index = go.Scatter(x=index.index, y=index['Close'], mode='lines', name='Dollar Index')
    trace_futures = go.Scatter(x=futures.index, y=futures['Close'], mode='lines', name='Dollar Futures')

    fig.add_trace(trace_index, row=1, col=1)
    fig.add_trace(trace_futures, row=1, col=1)

    # Set subplot title and axis labels
    fig.update_yaxes(title_text='Price', row=1, col=1)

    # Calculate and plot the spread between dollar index and futures
    spread = index['Close'] - futures['Close']
    trace_spread = go.Scatter(x=spread.index, y=spread, mode='lines', name='Spread')
    zero_line = go.Scatter(x=spread.index, y=[0] * len(spread), mode='lines', name='Zero Spread', line=dict(dash='dash'))

    fig.add_trace(trace_spread, row=2, col=1)
    fig.add_trace(zero_line, row=2, col=1)

    # Set subplot title and axis labels for the spread plot
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Spread', row=2, col=1)

    # Update layout for better display
    fig.update_layout(height=600, width=800, title_text='Dollar Index and Futures Interactive Subplots', template='plotly_dark')

    # Show interactive plot
    fig.show()

def BlackScholes(r, S, K, T, sigma, option_type="C", div=0):
    """
Calculate the price of a European Call or Put option using the Black-Scholes formula, including adjustments for continuous dividend yield.

Parameters
----------
r : float
    Risk-free interest rate, expressed as a decimal (e.g., 0.05 for 5%).
S : float
    Current stock price.
K : float
    Strike price of the option.
T : float
    Time to expiration of the option, expressed in years (e.g., 0.5 for six months).
sigma : float
    Volatility of the underlying stock, expressed as a decimal.
option_type : str, optional
    Type of the option: 'C' for Call option or 'P' for Put option. The default is 'C'.
div : float, optional
    Continuous dividend yield of the underlying stock, expressed as a decimal. The default is 0.

Returns
-------
None
    This function prints the option price, Delta, Gamma, Vega, Theta, and Rho for the specified Call or Put option.
    - Option Price: The theoretical price of the option using the Black-Scholes formula.
    - Delta: Measures the rate of change of the option's price with respect to changes in the underlying asset's price.
    - Gamma: Measures the rate of change in Delta with respect to changes in the underlying asset's price.
    - Vega: Measures sensitivity of the option's price to changes in the volatility of the underlying asset. Note that Vega is not a Greek letter; the symbol ν (nu) is sometimes used.
    - Theta: Measures the rate of time decay of the option's price, expressed per day. Used 252 trading days.
    - Rho: Measures sensitivity of the option's price to changes in the risk-free interest rate.

Note
----
This function assumes European options, which can only be exercised at expiration. It does not apply to American options, which can be exercised at any time before expiration.
----
"""
    
    # Define d1 and d2 for Black-Scholes, including the dividend yield
    d1 = (np.log(S / K) + (r - div + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Subfunction to calculate delta
    def calc_delta():
        if option_type == "C":
            return norm.cdf(d1, 0, 1)  # Delta for Call option
        elif option_type == "P":
            return -norm.cdf(-d1, 0, 1)  # Delta for Put option
        
    # Subfunction to calculate Gamma
    def calc_gamma():
        return norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
    
    # Subfunction to calculate Vega
    def calc_vega():
        return (S * norm.pdf(d1, 0, 1) * np.sqrt(T)) * 0.01
    
    # Subfunction to calculate Theta, including dividend yield
    def calc_theta():
        if option_type == "C":
            return (-S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T)) + div * S * norm.cdf(d1, 0, 1) - r * K * np.exp(-r * T) * norm.cdf(d2, 0, 1)) / 252 # Theta for Call option
        elif option_type == "P":
            return (-S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T)) - div * S * norm.cdf(-d1, 0, 1) + r * K * np.exp(-r * T) * norm.cdf(-d2, 0, 1)) / 252 # Theta for Put option

    # Subfunction to calculate Rho
    def calc_rho():
        if option_type == "C":
            return (K * T * np.exp(-r * T) * norm.cdf(d2, 0, 1)) * 0.01  # Rho for Call option
        elif option_type == "P":
            return (-K * T * np.exp(-r * T) * norm.cdf(-d2, 0, 1)) * 0.01  # Rho for Put option

    try:
        delta = calc_delta()  # Call the subfunction to calculate delta
        gamma = calc_gamma()  # Call the subfunction to calculate gamma
        vega = calc_vega()    # Call the subfunction to calculate vega
        theta = calc_theta()  # Call the subfunction to calculate theta
        rho = calc_rho()      # Call the subfunction to calculate rho
        if option_type == "C":
            price = S * np.exp(-div * T) * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
            print(f"Call Option Price: {price:.5f}")
            print("-----------------")
            print(f"Call Delta: {delta:.5f}")
            print(f"Gamma: {gamma:.5f}")
            print(f"Vega: {vega:.5f}")
            print(f"Call Theta: {theta:.5f}")
            print(f"Call Rho: {rho:.5f}")
        elif option_type == "P":
            price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * np.exp(-div * T) * norm.cdf(-d1, 0, 1)
            print(f"Put Option Price: {price:.5f}")
            print("-----------------")
            print(f"Put Delta: {delta:.5f}")
            print(f"Gamma: {gamma:.5f}")
            print(f"Vega: {vega:.5f}")
            print(f"Put Theta: {theta:.5f}")
            print(f"Put Rho: {rho:.5f}")
        else:
            raise ValueError("Invalid option type! Please confirm option type either 'C' for Call option or 'P' for Put option")
    except ValueError as e:
        print(e)
