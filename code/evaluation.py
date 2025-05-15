
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math
import numpy as np
import os
import pandas as pd
import statistics as stats
import statsmodels.api as sm

from scipy.stats.mstats import winsorize

from sklearn.linear_model import LinearRegression


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


def binnedDotPlot(df,
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

def binnedDotPlotMultiY(df,
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

def cod(ratios):
    
    """
    Calculate coefficient of dispersion for a given array or dataframe column of
    sales ratios.

    input:
    - ratios. Can be np.array or pd.DataFrame

    output:
    - cod, coefficeint of dispersion
    """
    
    median = stats.median(ratios)
    diff = [abs(r-median) for r in ratios]

    cod = (100/(len(diff)*median))*(sum(diff))

    return cod

def prd(ratio, assessed, sale):
    """

    """

    median_ratio = stats.median(ratio)
    median_assessed = stats.median(assessed)
    median_sale = stats.median(sale)

    prd = median_ratio/(median_assessed/median_sale)

    return prd

def log_coef(assessed, sale):
    
    """

    """
    X = [math.log(s) for s in sale]
    X = sm.add_constant(X)
    y = [math.log(a) - math.log(s) for a, s in zip(assessed,sale)]
    
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC0')

    beta = results.params[1]

    return beta

def mse(assessed, sale):
    """
    """

    se = [(a-s)**2 for a, s in zip(assessed, sale)]
    mse = np.mean(se)

    return mse

def mpe(assessed, sale):

    pe = [(s-a)/s for a,s in zip(assessed, sale)]
    mpe = np.mean(se)

    return mpe
