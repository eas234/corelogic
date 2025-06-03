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

    return f"{cod:,.4f}"

def prd(ratio, assessed, sale):
    """

    """

    median_ratio = stats.median(ratio)
    median_assessed = stats.median(assessed)
    median_sale = stats.median(sale)

    prd = median_ratio/(median_assessed/median_sale)

    return f"{prd:,.4f}"

def log_coef(assessed, sale):
    
    """

    """
    X = [math.log(s) for s in sale]
    X = sm.add_constant(X)
    y = [math.log(a) - math.log(s) for a, s in zip(assessed,sale)]
    
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC0')

    beta = results.params[1]
    print(results.params)

    return f"{beta:,.4f}"

def mse(assessed, sale):
    """
    """

    se = [(a-s)**2 for a, s in zip(assessed, sale)]
    mse = np.mean(se)

    return f"{mse:,.4f}"

def rmse(assessed, sale):
    """
    """

    se = [(a-s)**2 for a, s in zip(assessed, sale)]
    mse = np.mean(se)
    rmse = np.sqrt(mse)

    return f"{rmse:,.4f}"

def mpe(assessed, sale):

    pe = [(s-a)/s for a,s in zip(assessed, sale)]
    mpe = np.mean(pe)

    return f"{mpe:,.4f}"

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
