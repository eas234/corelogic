import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math
import numpy as np
import os
import pandas as pd
import statistics as stats
import statsmodels.api as sm

import yaml
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
from scipy.stats import kstest


from typing import Union


def set_working_dir():

    '''
    Sets working directory as directory of script.

    inputs: None
    outputs: None
    '''

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    return None


def binned_dot_plot(df,
                    binVar='MARKET_TOTAL_VALUE_WINS',
                    nBins=20,
                    yVar='SALE_AMOUNT_WINS',
                    thousands=True, # convert axis varbs to units of thousands
                    center='mean', # options: median, mean
                    xLabel='Assessed Market Value ($, Thousands)',
                    yLabel='Sale Price ($, Thousands)',
                    axLim=800):
    
    copy=df.copy()

    if thousands==True:
        copy[binVar] = copy[binVar]/1000
        copy[yVar] = copy[yVar]/1000
    
    bins = pd.qcut(copy[binVar], q=nBins, duplicates='drop')
    binMidpoints = bins.apply(lambda interval: (interval.left + interval.right) / 2)
    copy['midpoints'] = binMidpoints

    grouped = copy.groupby('midpoints')[yVar].agg(median='median', 
                                                    q1=lambda x: np.percentile(x, 25), 
                                                    q3=lambda x: np.percentile(x, 75),
                                                    mean='mean',
                                                    std='std',
                                                    count='count').reset_index()

    if center=='median':
        grouped['lowerErr'] = grouped['median'] - grouped['q1']
        grouped['upperErr'] = grouped['q3'] - grouped['median']

    elif center=='mean':
        grouped['se'] = grouped['std']/np.sqrt(grouped['count'])
        grouped['lowerErr'] = grouped['mean'] - 1.96*grouped['se']
        grouped['upperErr'] = grouped['mean'] + 1.96*grouped['se']

    formatter = FuncFormatter(lambda x, _: f"{int(x)}")

    plt.figure(figsize=(8,5))
    plt.errorbar(
    grouped['midpoints'],
    grouped[center],
    yerr=[grouped['lowerErr'], grouped['upperErr']],
    fmt='o',
    capsize=5,
    color='black')

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(0, axLim)
    plt.ylim(0, axLim)

    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], linestyle='--', color='red')

    plt.savefig('../figs/dot_plot_' + binVar + '_' + yVar + '_' + center + '_' + str(axLim) + '.png')

    return None

def binned_dot_plot_multi_y(df,
                        binVar='MARKET_TOTAL_VALUE_WINS',
                        yVars=['SALE_AMOUNT_WINS'],
                        line_labels=None,
                        colors=None,
                        alphas=None,
                        nBins=20,
                        thousands=True,
                        center='mean',
                        xLabel='Assessed Market Value ($, Thousands)',
                        yLabel='Sale Price ($, Thousands)',
                        axLim=800,
                        censor_top=False):

    copy = df.copy()

    if thousands:
        copy[binVar] = copy[binVar] / 1000
        for yVar in yVars:
            copy[yVar] = copy[yVar] / 1000

    # Create quantile bins and midpoints
    bins = pd.qcut(copy[binVar], q=nBins, duplicates='drop')
    copy['midpoints'] = bins.apply(lambda interval: (interval.left + interval.right) / 2)

    # Begin plot
    plt.figure(figsize=(8, 5))

    for idx, yVar in enumerate(yVars):
        grouped = copy.groupby('midpoints')[yVar].agg(
            median='median',
            q1=lambda x: np.percentile(x, 25),
            q3=lambda x: np.percentile(x, 75),
            mean='mean',
            std='std',
            count='count'
        ).reset_index()

        if center == 'median':
            grouped['lowerErr'] = grouped['median'] - grouped['q1']
            grouped['upperErr'] = grouped['q3'] - grouped['median']
        elif center == 'mean':
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
            grouped['lowerErr'] = 1.96 * grouped['se']
            grouped['upperErr'] = 1.96 * grouped['se']

        # Pick color and label
        color = colors[idx] if colors and idx < len(colors) else 'black'
        label = line_labels[idx] if line_labels and idx < len(line_labels) else yVar
        alpha = alphas[idx] if alphas and idx < len(line_labels) else 1.0
        
        if censor_top==True:
            plt.errorbar(
                grouped['midpoints'][:-1],
                grouped[center][:-1],
                yerr=[grouped['lowerErr'][:-1], grouped['upperErr'][:-1]],
                fmt='o-',
                fillstyle='none',
                capsize=5,
                label=label,
                color=color,
                alpha=alpha
            )
        else:
            plt.errorbar(
            grouped['midpoints'],
            grouped[center],
            yerr=[grouped['lowerErr'], grouped['upperErr']],
            fmt='o-',
            fillstyle='none',
            capsize=5,
            label=label,
            color=color,
            alpha=alpha
        )

    # Final plot settings
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(0, axLim)
    plt.ylim(0, axLim)

    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], linestyle='--', color='red')

    plt.legend()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

    joined_yvars = '_'.join(yVars)
    plt.savefig(f'../figs/dot_plot_{binVar}_{joined_yvars}_{center}_{axLim}.png')
    plt.show()

    return None

def cod(assessed, sale, format=False):
    
    """
    Calculate coefficient of dispersion for a given array or dataframe column of
    sales ratios.

    input:
    - ratios. Can be np.array or pd.DataFrame

    output:
    - cod, coefficeint of dispersion
    """
    ratios = [a/s for a, s in zip(assessed, sale)]
    median = stats.median(ratios)
    diff = [abs(r-median) for r in ratios]

    cod = (100/(len(diff)*median))*(sum(diff))

    if format:
        return f"{cod:,.2f}%"
    else: 
        return cod

def prd(assessed, sale, format=False):
    """

    """

    ratios = [a/s for a, s in zip(assessed, sale)]
    mean_ratio = np.mean(ratios)
    mean_assessed = np.mean(assessed)
    mean_sale = np.mean(sale)

    prd = mean_ratio/(mean_assessed/mean_sale)

    if format:
        return f"{prd:,.4f}"
    else:
        return prd

def log_coef(assessed, sale, format=False):
    
    """

    """
    X = [math.log(s) for s in sale]
    X = sm.add_constant(X)
    y = [math.log(a) - math.log(s) for a, s in zip(assessed,sale)]
    
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC0')

    beta = results.params[1]

    if format:
        return f"{beta:,.4f}"
    else:
        return beta

def mse(assessed, sale, format=False):
    """
    """

    se = [(a-s)**2 for a, s in zip(assessed, sale)]
    mse = np.mean(se)

    if format:
        return f"{mse:,.4f}"
    else:
        return mse

def rmse(assessed, sale, format=False):
    """
    """

    se = [(a-s)**2 for a, s in zip(assessed, sale)]
    mse = np.mean(se)
    rmse = np.sqrt(mse)

    if format:
        return f"${rmse:,.0f}"
    else:
        return rmse

def mae(assessed, sale, format=False):
    
    ae = [abs(a-s) for a, s, in zip(assessed, sale)]
    mae = np.mean(ae)
    
    if format:
        return f"${mae:,.0f}"
    else:
        return mae

def mpe(assessed, sale, format=False):

    pe = [(s-a)/s for a,s in zip(assessed, sale)]
    mpe = 100*np.mean(pe)

    if format:
        return f"{mpe:,.2f}"
    else:
        return mpe

def mape(assessed, sale, format=False):
    
    ape = [abs((s-a)/s) for a, s in zip(assessed, sale)]
    mape = 100*np.mean(ape)
    
    if format:
        return f"{mape:,.2f}%"
    else:
        return mape

def r_squared(assessed, sale, format=False):
    
    resids = assessed - sale
    resids_squared = resids**2
    sum_squared_resids = resids_squared.sum()
    
    mean_sale = sale.mean()
    dif = sale - mean_sale
    dif_squared = dif**2
    total_sum_squares = dif_squared.sum()
    
    r_squared = 1 - sum_squared_resids/total_sum_squares
    
    if format:
        return f"{r_squared:,.4f}"
    else:
        return r_squared

def gini_ratio(assessed, sale):
    """
    Gini_sale > Gini_assessed means assessment ratios are higher for low-priced than high-priced properties, i.e., regressive
    """
    return gini(assessed) / gini(sale)

def gini(vector):
    """
    1. Order data from smallest to largest
    2. G = 2(Sum_i=1^n i*x_i) / (n * Sum_i=1^n x_i) - (n + 1) / n 
    """

    vector = vector.sort_values(ascending=True).reset_index(drop=True)
    n = len(vector)
    gini = 2 *  np.sum(vector.index * vector) / (n * np.sum(vector)) - (n + 1) / n

    return gini

def suits_index(assessed, sale):
    
    """
    1 - (L/K) where:
    K = 5000
    L approx. Sum_i=1^n (1/2) * [a(p_i) + a(p_{i - 1})] * (p_i - p_{i - 1})
    """

    ## Create DF and sort on sale price from lowest to highest
    suits_df = pd.DataFrame({'assessed' : assessed, 'sale' : sale})
    suits_df = suits_df.sort_values('sale', ascending=True).reset_index(drop=True)

    ## Calculate cumulative amount of sale and tax
    suits_df_cum = np.cumsum(suits_df)

    ## Rescale to be in 0-100 range
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(suits_df_cum)
    suits_df_cum = scaler.transform(suits_df_cum)
    suits_df_cum = pd.DataFrame(suits_df_cum)
    suits_df_cum.columns = ['assessed', 'sale']

    ## Lag each column
    suits_df_cum['assessed_lag'] = suits_df_cum['assessed'].shift(1)
    suits_df_cum['sale_lag'] = suits_df_cum['sale'].shift(1)
    suits_df_cum = suits_df_cum.dropna()

    ## Approximate integral
    L = np.sum(0.5 * (suits_df_cum['assessed'] + suits_df_cum['assessed_lag']) * (suits_df_cum['sale'] - suits_df_cum['sale_lag']))

    ## K is the area under the 45-degree line - since data is scaled 0-100 that is:
    ## (1/2) * 100 * 100 = 5000
    K = 5000

    return 1 - (L/K)

def prb(assessed, sale):
    """
    Calculate coefficient of price-related bias
    Formula from: https://gitlab.com/ccao-data-science---modeling/packages/assessr/-/blob/master/R/formulas.R?ref_type=heads

    Indicates the percentage by which assessment ratios change whenever values are doubled or halved. 
    e.g. a PRB of −0.03 means assessment levels decrease by 3 percent when value doubles. 

    Per CCAO: 
    - The PRB should range between −0.05 and +0.05.
    - PRBs outside the range of −0.10 to +0.10 are considered unacceptable. 
    """

    ratio = assessed / sale
    median_ratio = np.median(ratio)

    y = (ratio - median_ratio) / median_ratio
    X = np.log(((assessed / median_ratio) + sale) * 0.5) / np.log(2)

    X = sm.add_constant(X)
    
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC0')

    beta = results.params.iloc[1]

    return beta

def ks_test(assessed, sale):
    """
    KS test: max distance between distributions
    standard (fast) test for whether data was drawn from different distributions
    """

    assessed = np.asarray(assessed)
    sale = np.asarray(sale)

    kde_assessed = gaussian_kde(assessed)
    kde_sale = gaussian_kde(sale)

    min_val = min(assessed.min(), sale.min())
    max_val = max(assessed.max(), sale.max())

    curve_assessed =  kde_assessed.evaluate(np.linspace(min_val, max_val, 1000))
    curve_sale =  kde_sale.evaluate(np.linspace(min_val, max_val, 1000))

    plt.plot(np.linspace(min_val, max_val, 1000), curve_assessed, label='assessed')
    plt.plot(np.linspace(min_val, max_val, 1000), curve_sale, label='sale')
    plt.xlabel("value")
    plt.legend()
    plt.show()
    plt.close()

    return kstest(curve_assessed, curve_sale).pvalue

def kde_test(assessed, sale, n_bootstrap=1000):

    """
    Tests whether the distance of two density functions f(x) and g(x) as given by: 
    S_v [f(v) - g(v)]^2 
    is statistically significant. v represents "the x-axis, i.e., sale price or assessed value" (per McMillen & Singh)
    """

    assessed = np.asarray(assessed)
    sale = np.asarray(sale)
    
    kde_assessed = gaussian_kde(assessed)
    kde_sale = gaussian_kde(sale)
    
    min_val = min(assessed.min(), sale.min()) ## TODO: not sure if these should be across both sets of data or pick one
    max_val = max(assessed.max(), sale.max())
    grid = np.linspace(min_val, max_val, 1000) ## TODO: 1000 may be too large of a step size

    statistic = np.trapezoid((kde_assessed(grid) - kde_sale(grid)) ** 2, grid)
    
    p_value = kde_test_pvalue(assessed, sale, n_bootstrap, statistic)
    
    return p_value

def kde_test_pvalue(x, y, n_bootstrap, observed_stat):

    n = len(x)
    combined = np.concatenate([x, y])
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        perm = np.random.permutation(combined)
        x_boot = perm[:n]
        y_boot = perm[n:]
        
        kde_xboot = gaussian_kde(x_boot)
        kde_yboot = gaussian_kde(y_boot)
        
        min_val = min(x_boot.min(), y_boot.min())
        max_val = max(x_boot.max(), y_boot.max())
        grid = np.linspace(min_val, max_val, 1000)

        stat_boot = np.trapezoid((kde_xboot(grid) - kde_yboot(grid)) ** 2, grid)

        bootstrap_stats.append(stat_boot)
    
    p_value = np.mean(np.array(bootstrap_stats) >= observed_stat)
    return p_value

def add_labels(x_positions, heights, offset=0.01, fontsize=10):

    """
    Helper function for over_under_bars(). adds value labels formatted
    as integers at the top of each bar, centered over the midpoint.
    """
    for x, h in zip(x_positions, heights):
        plt.text(x, h + offset, f"{int(round(h))}", ha='center', va='bottom', fontsize=fontsize)

def over_under_bars(df: pd.DataFrame=None,
                     model_id: str='model',
                     n_bins: int=10,
                     thousands: bool=True,
                     figsize: tuple=(10,6),
                     x_label: str='Sale price ($, thousands)',
                     y_label: str='Share (percentage points)',
                     fig_title: str=None,
                     axis_label_fontsize: int=18,
                     bar_label_fontsize: int=10,
                     title_fontsize: int=20,
                     tick_fontsize: int=16,
                     fig_dir: str=None):
    """
    Grouped bar plot showing share of homes over/underassessed by bins of sale price
    
    inputs:
    - df: DataFrame containing assessed values and sale prices
    - model_id: string indicating model which generated assessed values
    - nbins: number of bins to put on x axis of histogram
    - thousands: boolean indicating whether to divide x axis variable by 1000
    for readability
    - figsize: size of figure
    - x_label, y_label: axis labels
    - fig_title: title of figure
    - axis_label_fontsize: fontsize of axis labels
    - bar_label_fontsize: fontside of value labels that sit atop each bar
    - title_fontsize: fontsize of title
    - tick_fontsize: fontsize of axis tick labels
    - fig_dir: directory where figure gets written. if None, function does not write.
    """
    
    # copy dataframe
    copy=df.copy()

    # bin observations by sale price
    bins = pd.qcut(copy['y_true_' + model_id], q=n_bins, duplicates='drop')
    binMidpoints = bins.apply(lambda interval: (interval.left + interval.right) / 2)
    copy['midpoints'] = binMidpoints
    
    # gen varbs for plotting
    copy['assessed_below'] = [1 if x <= y else 0 for x, y in zip(copy['y_pred_' + model_id], copy['y_true_' + model_id])]
    copy['assessed_above'] = [1 if x > y else 0 for x, y in zip(copy['y_pred_' + model_id], copy['y_true_' + model_id])]

    # group observations in copy by bins of sale price
    grouped = copy[['midpoints', 'assessed_below', 'assessed_above']].groupby('midpoints').mean().reset_index()
    grouped.sort_values('midpoints', ascending=True, inplace=True)
                         
    if thousands==True:
        # divide x-axis (sale price) by 1000 for readability
        grouped['midpoints'] = grouped['midpoints'].astype(float)*(1/1000)
        grouped['midpoints'] = grouped['midpoints'].apply(lambda x: f"{x:,.0f}")
    
    x = np.arange(n_bins)
    bar_width = 0.35

    # set figsize
    plt.figure(figsize=figsize)
    
    # set font
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    # Plot bars centered over each bin label
    plt.bar(x - bar_width/2, grouped['assessed_below']*100, width=bar_width, label='Share Underassessed', color='#ef8a62', alpha=0.8)
    plt.bar(x + bar_width/2, grouped['assessed_above']*100, width=bar_width, label='Share Overassessed', color='#67a9cf', alpha=0.8)

    # add text labels atop each bar displaying value of bar
    add_labels(x - bar_width/2, grouped['assessed_below']*100, fontsize=bar_label_fontsize)
    add_labels(x + bar_width/2, grouped['assessed_above']*100, fontsize=bar_label_fontsize)
    
    # Set tick labels
    plt.xticks(x, grouped['midpoints'], fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Set axis labels
    plt.xlabel(x_label, fontsize=axis_label_fontsize)
    plt.ylabel(y_label, fontsize=axis_label_fontsize)

    if fig_title:
        # set figure title
        plt.title(fig_title, fontsize=title_fontsize)

    # set tick fontsize
    plt.legend(fontsize=tick_fontsize)

    # various plot formatting
    ax = plt.gca()
    # get rid of top and right plot bars
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add some horizontal lines for easy reading
    for y in [20, 40, 60, 80]:
        ax.axhline(y=y, linestyle='--', color='gray', alpha=0.2)
    plt.tight_layout()

    if fig_dir:
        plt.save_fig(os.path.join(fig_dir, model_id + '_over_under_bars.png'))
    
    return grouped


def binned_dot_plot_multi_x(df,
                        binVars=['MARKET_TOTAL_VALUE_WINS'],
                        yVar='SALE_AMOUNT_WINS',
                        line_labels=None,
                        colors=None,
                        alphas=None,
                        nBins=20,
                        thousands=True,
                        center='mean',
                        xLabel='Sale Price ($, Thousands)',
                        yLabel='Assessed Market Value ($, Thousands)',
                        axLim=800,
                        censor_top=False):

    copy = df.copy()

    if thousands:
        for var in binVars:
            copy[var] = copy[var] / 1000
        
        copy[yVar] = copy[yVar] / 1000

    # Create quantile bins and midpoints
    for var in binVars:
        bins = pd.qcut(copy[var], q=nBins, duplicates='drop')
        copy['midpoints_' + var] = bins.apply(lambda interval: (interval.left + interval.right) / 2)

    # Begin plot
    plt.figure(figsize=(8, 5))

    for idx, var in enumerate(binVars):
        grouped = copy.groupby('midpoints_'+var)[yVar].agg(
            median='median',
            q1=lambda x: np.percentile(x, 25),
            q3=lambda x: np.percentile(x, 75),
            mean='mean',
            std='std',
            count='count'
        ).reset_index()

        if center == 'median':
            grouped['lowerErr'] = grouped['median'] - grouped['q1']
            grouped['upperErr'] = grouped['q3'] - grouped['median']
        elif center == 'mean':
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
            grouped['lowerErr'] = 1.96 * grouped['se']
            grouped['upperErr'] = 1.96 * grouped['se']

        # Pick color and label
        color = colors[idx] if colors and idx < len(colors) else 'black'
        label = line_labels[idx] if line_labels and idx < len(line_labels) else yVar
        alpha = alphas[idx] if alphas and idx < len(line_labels) else 1.0
        
        if censor_top==True:
            plt.errorbar(
                grouped['midpoints_'+var][:-1],
                grouped[center][:-1],
                yerr=[grouped['lowerErr'][:-1], grouped['upperErr'][:-1]],
                fmt='o-',
                fillstyle='none',
                capsize=5,
                label=label,
                color=color,
                alpha=alpha
            )
        else:
            plt.errorbar(
            grouped['midpoints_'+var],
            grouped[center],
            yerr=[grouped['lowerErr'], grouped['upperErr']],
            fmt='o-',
            fillstyle='none',
            capsize=5,
            label=label,
            color=color,
            alpha=alpha
        )

    # Final plot settings
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(0, axLim)
    plt.ylim(0, axLim)

    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], linestyle='--', color='red')

    plt.legend()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

    joined_binVars = '_'.join(binVars)
    plt.savefig(f'../figs/dot_plot_{yVar}_{joined_binVars}_{center}_{axLim}.png')
    plt.show()

    return None

def get_metrics(df: pd.DataFrame=None, 
                true: str='sale', 
                pred: str='assessed', 
                model_id: str='model_id'):

    """
    Helper function for calibration_by_ntile() to generate table of evaluation metrics.
    Function is applied individually to each group of a grouped dataframe.

    Inputs:
    - df: DataFrame with true and predicted values
    - true: string indicating name of true value column
    - pred: string indicating name of predicted value column
    - model_id: string indicating name of column containing the ID of the model that generated the predictions

    Returns:
    pd.Series object with model ids, median sale price within the dataframe, and suite of eval metrics
    """
    
    return pd.Series({
        'Model ID': df['model_id'].min(),
        'Median Sale Price': f"${df[true].median():,.0f}",
        'RMSE': rmse(df[pred], df[true]),
        'MAE': mae(df[pred], df[true]),
        'MAPE': mape(df[pred], df[true]),
        'Log Coef': log_coef(df[pred], df[true]),
        'COD': cod(df[pred], df[true]),
        'PRD': prd(df[pred], df[true])
    })


def calibration_by_ntile(dfs: list=None,
                        ntile_var: str='assessed',
                        model_ids: Union[str,list]='model_id',
                        n: int=4,
                        id_colname: str=None
                        ):

    """
    function which creates a table of evaluation metrics separately by ntile and model type. 

    Inputs: 
    - dfs: list of dataframes containing individual model outputs
    - model_ids: can be string or list. if string, indicates column across dfs that contains the model id.
    if list, contains list of custom model ids.
    - n: number of ntiles. E.g., if n=4, function produces metrics by quartile.
    - id_colname: if not None, function renames 'model_id' to whatever value is in id_colname. Useful in instances
    where model ids indicate e.g. number of features in model, and you want the output table to reflect that

    output:
    dataframe whose horizontal axis is the metrics computed in get_metrics, and whose vertical axis is quartiles-by-model
    """
    
    # generate dataframe to store output
    output = pd.DataFrame(columns=['ntile', 'Model ID', 'Median Sale Price', 'RMSE', 'MAE', 'MAPE', 'Log Coef', 'COD', 'PRD'])
    
    i = 0 
    # looping through dataframes of model outputs
    for df in dfs:
        # get the index of the column containing model predictions
        idx = [i for i in range(len(df.columns)) if 'y_pred' in df.columns[i]]
        
        # define pd.Series 'assessed' which contains model predictions
        assessed = df[df.columns[idx[0]]]
        
        # if 'assessed' is passed as ntile_var, then ntiles will be defined by model predictions/assessments
        if ntile_var == 'assessed':
            ntile = pd.qcut(assessed, q=n, duplicates='drop', labels=['Q' + str(x+1) for x in range(n)])
        
        # next, get index of dataframe column containing sale prices/true values
        idx = [i for i in range(len(df.columns)) if 'y_true' in df.columns[i]]
        sale = df[df.columns[idx[0]]]
        
        # if 'sale' is passsed as ntile_var, then define ntile vars by sale prices/true values
        if ntile_var == 'sale':
            ntile = pd.qcut(sale, q=n, duplicates='drop', labels=['Q' + str(x+1) for x in range(n)])
            
            
        # model_ids can either be a string, indicating the column in each dataframe containing the model id
        if isinstance(model_ids, str):
            model_id = df[model_ids]
        
        # or they can be a list, indicating that for each model we want to set the id as whatever is in the list
        else:
            model_id = [model_ids[i] for x in ntile]
            i+=1
        
        # make a temporary dataframe containing ntiles, assessed values, sale values, and model ids
        temp = pd.DataFrame({'ntile': ntile, 'assessed': assessed, 'sale': sale, 'model_id': model_id})
        
        # group temp and get metrics by quartile
        grouped = temp.groupby('ntile').apply(get_metrics).reset_index()
        
        # append grouped to output dataframe
        output = pd.concat([output, grouped], axis=0)
        
    if id_colname:
        output.rename(columns={'Model ID': id_colname}, inplace=True)
        return output.groupby(by=['ntile', id_colname]).min().transpose()
    
    else:
        return output.groupby(by=['ntile', 'Model ID']).min().transpose()

def r_squared_coef_dot_plot(dfs: list=None,
                           figsize: tuple=(10,8),
                           labels: list=None,
                           x_label: str='R-squared',
                           y_label: str='Log coefficient (lower is more regressive)',
                           axis_label_fontsize: int=20,
                           tick_fontsize: int=16):
    
    '''
    Function to generate dot plot, where each dot corresponds to a model. X-axis
    is the r-squared of the model and y-axis is the coefficient on a regression
    of log ratios to log sale price.
    
    Plot captures tradeoff between fairness/accuracy for a given model
    '''
    
    r_squareds = []
    coefs = []
    
    for df in dfs:
        idx = [i for i in range(len(df.columns)) if 'y_pred' in df.columns[i]]
        assessed = df[df.columns[idx[0]]]
        
        idx = [i for i in range(len(df.columns)) if 'y_true' in df.columns[i]]
        sale = df[df.columns[idx[0]]]
        
        r_squareds.append(r_squared(assessed,sale))
        coefs.append(log_coef(assessed, sale))
        
    # correct types
    r_squareds = [float(x) for x in r_squareds]
    coefs = [float(x) for x in coefs]
    
    # create figure
    fig,ax = plt.subplots(figsize=figsize)
    
    # set font
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    # gen array of alphas
    
    alphas = np.linspace(0.1,1,len(dfs))
    ax.scatter(r_squareds, coefs, alpha=alphas, edgecolors='k', s=100)
    
    if labels:
        for i, txt in enumerate(labels):
            ax.annotate(txt, (r_squareds[i]+0.0005, coefs[i]+0.0005), fontsize=tick_fontsize-2)
            
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    
    ax.tick_params(axis='both', labelsize=tick_fontsize) 
    plt.show()
    plt.close()
    
    return r_squareds, coefs

def mae_coef_dot_plot(dfs=None, #: Optional[List[pd.DataFrame]] = None,
                           figsize: tuple = (5,4),
                           labels=None, #: Optional[List[str]] = None,
                           x_label: str = 'MAE ($)',
                           y_label: str = 'Log coefficient (lower is more regressive)',
                           axis_label_fontsize: int = 20,
                           tick_fontsize: int = 16):
    
    if dfs is None:
        raise ValueError("List of DataFrames cannot be None")
        
    #set_serif_font()  # Set serif font before creating plot
    maes = []
    coefs = []
    
    for df in dfs:
        idx = [i for i in range(len(df.columns)) if 'y_pred' in df.columns[i]]
        assessed = df[df.columns[idx[0]]]
        
        idx = [i for i in range(len(df.columns)) if 'y_true' in df.columns[i]]
        sale = df[df.columns[idx[0]]]
        
        maes.append(mae(assessed,sale,format_output=False))
        coefs.append(log_coef(assessed, sale, format_output=False))
        
    # correct types
    maes = [float(x) for x in maes]
    coefs = [float(x) for x in coefs]
    
    # create figure
    fig,ax = plt.subplots(figsize=figsize)
    
    # set font
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    # gen array of alphas
    
    alphas = np.linspace(0.1,1,len(dfs))
    ax.scatter(maes, coefs, alpha=alphas, edgecolors='k', s=100)
    
    if labels:
        for i, txt in enumerate(labels):
            ax.annotate(txt, (maes[i]+100, coefs[i]+0.0005), fontsize=tick_fontsize-2)
            
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    
    ax.tick_params(axis='both', labelsize=tick_fontsize) 
    plt.show()
    plt.close()
    
    return maes, coefs

def error_coef_relative_dot_plot(dfs: list = None,
                                 error = mae,
                                 best_fit_line = False,
                                 set_alphas = False,
                                 baseline_type = 'max error',
                                 figsize: tuple = (5,4),
                                 x_label: str = u'Δ MAE \n(lower is more accurate)',
                                 y_label: str = u'Δ log coefficient \n(higher is less regressive)',
                                 title: str = '',
                                 axis_label_fontsize: int = 10,
                                 tick_fontsize: int = 8):
    
    with open('../config/county_dict.yaml', 'r') as stream:
        county_dict_yaml = yaml.safe_load(stream)

    county_dict = pd.DataFrame({'fips' : list(county_dict_yaml.keys()), 'county' : list(county_dict_yaml.values())})

    if dfs is None:
        raise ValueError("List of DataFrames cannot be None")
        
    errs = []
    coefs = []
    ids = []
    n_features = []

    for df in dfs:
        idx = [i for i in range(len(df.columns)) if 'y_pred' in df.columns[i]]
        assessed = df[df.columns[idx[0]]]
        
        idx = [i for i in range(len(df.columns)) if 'y_true' in df.columns[i]]
        sale = df[df.columns[idx[0]]]

        idx = [i for i in range(len(df.columns)) if 'fips' in df.columns[i]]
        fips = df[df.columns[idx[0]]][0]

        idx = [i for i in range(len(df.columns)) if 'model_id' in df.columns[i]]
        nf = df[df.columns[idx[0]]]

        n = nf.str.extract(r'_(\d+)_features').astype(int)[0][0]
        
        errs.append(error(assessed, sale))
        coefs.append(log_coef(assessed, sale))
        ids.append(fips)
        n_features.append(n)

    errs = [float(x) for x in errs]
    coefs = [float(x) for x in coefs]

    out = pd.DataFrame({'fips' : ids, 'n_features' : n_features, 'errs' : errs, 'coefs' : coefs})

    if baseline_type == 'max error':
        baseline = out[out.groupby(['fips'])['errs'].transform('max') == out['errs']]

    elif baseline_type == 'fewest features':
        baseline = out[out.groupby(['fips'])['n_features'].transform('min') == out['n_features']]

    else:
        raise Exception('unrecognized baseline type')

    ## Either way, results are then compared to the baseline
    baseline.columns = ['fips', 'n_features', 'baseline_err', 'baseline_coef']
    baseline = baseline.drop(columns=['n_features'])
    baseline = baseline.drop_duplicates(subset=['fips'])

    out = out.merge(baseline, on='fips')
    out['error_delta'] =  out['errs'] - out['baseline_err']
    out['coef_delta'] = out['coefs'] - out['baseline_coef']
    out['fips'] = out['fips'].astype(str).str.pad(width=5, side='left', fillchar='0')

    out = out.merge(county_dict, on='fips')

    sns.set_style('white')

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['font.family'] = 'DejaVu Serif'
    ax.axvline(x=0, color='lightgray', linestyle='--', linewidth=0.8) 
    ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=0.8)

    colors = sns.color_palette(n_colors=out['fips'].nunique())

    i = 0
    legend = []
    for fips in out['fips'].unique():
        tmp = out[out['fips'] == fips].reset_index(drop=True)

        if set_alphas:
            alpha = tmp['n_features'] / max(tmp['n_features'])
        else: 
            alpha = np.repeat(1, tmp.shape[0])

        label = tmp['county'][0]

        ax.scatter(tmp['error_delta'], tmp['coef_delta'], alpha=alpha, edgecolors='k', s=50, color=colors[i])
        legend.append(Line2D([0], [0], marker='o', color='k', label=label,
                            markerfacecolor=colors[i], markersize=8, ls = ''))

        if best_fit_line: 
            m, b = np.polyfit(tmp['error_delta'], tmp['coef_delta'], deg=1)
            x = np.array([min(tmp['error_delta']), max(tmp['error_delta'])])
            ax.plot(x, b + m * x, color=colors[i], lw=1)

        i += 1
            
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    ax.set_title(title)

    ax.tick_params(axis='both', labelsize=tick_fontsize) 
    ax.tick_params(axis='x', rotation=30)

    ax.legend(handles=legend, bbox_to_anchor=(1.1, 1.05))

    plt.show()
    plt.close()

    return out[['fips', 'n_features', 'errs', 'coefs']]

