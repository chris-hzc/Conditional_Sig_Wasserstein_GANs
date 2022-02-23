import datetime as dt
from random import gauss
import sys
import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import pandas as pd
import seaborn as sns
from pylab import rcParams

from arch import arch_model
from numpy.linalg import LinAlgError
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, q_stat, adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot, moment
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
from sklearn.model_selection import TimeSeriesSplit

# Plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Data preprocessing
from sklearn import preprocessing

# Stats Test
from arch.unitroot import VarianceRatio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# plot
import seaborn as sns

# model fit
from scipy.optimize import curve_fit

import math
import random
import warnings

def func_powerlaw_full(x, a, b, c):
    return a*np.power(x, b) + c

def auto_correlation(data, maxLags=10, title="Auto correlation", plot=False, ax=None):
    auto_corr = []
    for i in range(1, maxLags):
        corr = data.autocorr(lag=i)
        #         corr = np.corrcoef(np.array([data[:-i], data[i:]]))[0,1]
        auto_corr.append(corr)

    if plot:
        if ax is None:
            ax = plt.gca()
        ax.scatter(x=range(1, maxLags), y=auto_corr, s=4)
        ax.set_ylim(-1, 1)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("lag K", fontsize=12)
        ax.set_ylabel("Auto-Correlation", fontsize=12)
        stats = f'Auto-Corr: {np.mean(auto_corr[:10]):.4f}\n'
        ax.text(x=1, y=.65, s=stats, color="red")

        return auto_corr, ax
    return auto_corr


def fat_tail_dist_cdf(data):
    dataMean, dataStd = np.mean(data), np.std(data)
    standardized_data = np.abs((data - dataMean) / dataStd)

    x = np.sort(standardized_data)
    x = x[x > np.mean(x) + 2 * np.std(x)]
    cdf = [(x <= value).sum() / len(x) for value in x]

    popt, pcov = curve_fit(func_powerlaw_full, x, cdf, maxfev=10000, p0=[0.5, -3, 0],
                           bounds=([-10, -10, -10], [10., 10., 10.]))
    return 1 - popt[1]

def leverage_effect_testing(data, maxLag=10, plot=False, title="Leverage Effect (True)", ax=None, ylim=(-15, 12)):
    leverage_value = []
    for shiftValue in range(1, maxLag + 1):
        origData = data.values[:len(data) - shiftValue]
        shiftedData = data.shift(-shiftValue).values[:len(data) - shiftValue]

        firstNumerator = np.mean(np.square(shiftedData) * origData)
        secondNumerator = np.mean(data) * np.mean(np.square(data))
        denominator = np.square(np.mean(np.square(data)))

        value = (firstNumerator - secondNumerator) / denominator
        leverage_value.append(value)

    if plot:
        if not ax:
            ax = plt.gca()
        ax.plot(leverage_value)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("lag k", fontsize=12)
        ax.axhline(y=0, color="black", linestyle="dashed")

        stats = f'Auto-Corr: {np.mean(leverage_value[:10]):.4f}\n'
        ax.text(x=1, y=8, s=stats, color="red")
        ax.set_ylim(ylim[0], ylim[1])
        return leverage_value, ax

    return leverage_value

def partial_acf(data, nlags=100, plot=False, title="PACF", ax=None):
    pacf_list = pacf(data, nlags=nlags)[1:]
    if plot:
        if ax is None:
            ax = plt.gca()
        ax.scatter(x=range(1, nlags + 1), y=pacf_list, s=4)
        ax.set_ylim(-1, 1)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("lag K", fontsize=12)
        ax.set_ylabel("Partial Auto-Correlation", fontsize=12)
        stats = f'PACf: {np.mean(pacf_list[:5]):.4f}\n'
        ax.text(x=1, y=.65, s=stats, color="red")
        return pacf_list, ax

    return pacf_list


def hurst(ts):
    ts = list(ts)
    N = len(ts)
    if N < 20:
        raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

    max_k = int(np.floor(N / 2))
    R_S_dict = []
    for k in range(10, max_k + 1):
        R, S = 0, 0

        # split ts into subsets
        subset_list = [ts[i:i + k] for i in range(0, N, k)]
        if np.mod(N, k) > 0:
            subset_list.pop()

        # calc mean of every subset
        mean_list = [np.mean(x) for x in subset_list]
        for i in range(len(subset_list)):
            cumsum_list = pd.Series(subset_list[i] - mean_list[i]).cumsum()
            R += max(cumsum_list) - min(cumsum_list)
            S += np.std(subset_list[i])
        R_S_dict.append({"R": R / len(subset_list), "S": S / len(subset_list), "n": k})

    log_R_S = []
    log_n = []

    for i in range(len(R_S_dict)):
        R_S = (R_S_dict[i]["R"] + np.spacing(1)) / (R_S_dict[i]["S"] + np.spacing(1))
        log_R_S.append(np.log(R_S))
        log_n.append(np.log(R_S_dict[i]["n"]))

    Hurst_exponent = np.polyfit(log_n, log_R_S, 1)[0]
    return Hurst_exponent

class Model_Evaluation():

    def __init__(self, data):
        self.data = data

    def auto_correlation(self, absolute=False, maxLags=10, title="Auto correlation", plot=False, ax=None):
        """Returns the auto correlation of a single price returns

        INPUTS
        =======
        absolute: boolean, optional, default value is False
           whether take the absolute value of time series
        maxLags: integer, optional, default value is 10
           the maximum number of lags to be calculated
        title: str, optional, default value is Auto correlation
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the auto correlation
        ax: ax, optional, default value is Auto correlation
           title of the plot

        RETURNS
        ========
        auto_corr: list of floats
        ax: ax of the plot
        """
        auto_corr = []

        data = self.data.copy()
        if absolute:
            for i in range(1, maxLags):
                corr = np.corrcoef(np.array([np.abs(data[:-i]), np.abs(data[i:])]))[0, 1]
                auto_corr.append(corr)
        else:
            for i in range(1, maxLags):
                corr = np.corrcoef(np.array([data[:-i], data[i:]]))[0, 1]
                auto_corr.append(corr)

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(x=range(1, maxLags), y=auto_corr, s=4)
            ax.set_ylim(-1, 1)
            ax.set_xscale('log')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Auto-Correlation", fontsize=12)
            stats = f'Auto-Corr: {np.mean(auto_corr[:10]):.4f}\n'
            ax.text(x=1, y=.65, s=stats, color="red")
            return auto_corr, ax

        return auto_corr

    def auto_correlation_avg(self, maxLags=10, title="Auto correlation", plot=False, ax=None):
        """Returns the auto correlation of multiple price returns

        INPUTS
        =======
        maxLags: integer, optional, default value is 10
           the maximum number of lags to be calculated
        title: str, optional, default value is Auto correlation
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the auto correlation
        ax: ax, optional, default value is Auto correlation
           title of the plot

        RETURNS
        ========
        auto_corr_mean: list of floats
            average value of multiple time series at each lag
        ax: ax of the plot
        """
        df = self.data.apply(auto_correlation, maxLags=maxLags, title=title, plot=False, ax=ax)
        auto_corr_mean = df.mean(axis=1)

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(x=range(1, maxLags), y=auto_corr_mean, s=4)
            ax.set_ylim(-1, 1)
            ax.set_xscale('log')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Auto-Correlation", fontsize=12)
            stats = f'Auto-Corr: {np.mean(auto_corr_mean[:10]):.4f}\n'
            ax.text(x=1, y=.65, s=stats, color="red")

            return auto_corr_mean, ax

        return auto_corr_mean

    # Fat-tail distribution
    def fat_tail_dist_cdf(self):
        """Returns the alpha of the function fitted to the probability density function of a single price returns

        RETURNS
        ========
        1 - popt[1]: float
            alpha of the best fitted function
        """

        dataMean, dataStd = np.mean(self.data), np.std(self.data)
        standardized_data = np.abs((self.data - dataMean) / dataStd)

        x = np.sort(standardized_data)
        x = x[x > np.mean(x) + 2 * np.std(x)]
        cdf = [(x <= value).sum() / len(x) for value in x]

        popt, pcov = curve_fit(func_powerlaw_full, x, cdf, maxfev=10000, p0=[0.5, -3, 0],
                               bounds=([-10, -10, -10], [10., 10., 10.]))
        return 1 - popt[1]

    # Fat-tail distribution average
    def fat_tail_dist_cdf_avg(self):
        """Returns the average alpha value of the function fitted to the probability density function of multiple price returns

        RETURNS
        ========
         avg_alpha: float
            average alpha of the best fitted function for multiple price returns
        """
        avg_alpha = self.data.apply(fat_tail_dist_cdf, axis=0).mean()
        return avg_alpha

    def vol_cluster_test(self, title="Volatility Clustering", plot=False, ax=None):
        """Returns the slope of the best fitted line between auto-correlation (y) and lags (x)

        INPUTS
        =======
        title: str, optional, default value is Volatility Clustering
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the auto correlation
        ax: ax, optional, default value is None
           ax of the plot

        RETURNS
        ========
        -popt[1]: floats
            slope of the best fitted line between auto-correlation and lag
        ax: ax of the plot
        """

        y = self.auto_correlation(absolute=True, maxLags=len(self.data) - 1, plot=False)
        x = list(range(1, len(self.data) - 1))

        try:
            popt, pcov = curve_fit(func_powerlaw_full, x, y, maxfev=1000, p0=[0.5, 0.5, 0],
                                   bounds=([-10, -10, -10], [10., 10., 10.]))
            stats = f'Auto-Corr: {-popt[1]:.4f}\n'
        except RuntimeError:
            popt = [0, 0]

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(np.log10(x[:100]), y[:100], s=2, alpha=1)
            #     ax.set_ylim(-0.5, 0.5)
            #     ax.set_xscale('log')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Auto-Correlation", fontsize=12)
            return -popt[1], ax

        return -popt[1]

    def vol_cluster_test_avg(self, title="Volatility Clustering", plot=False, ax=None):
        """Returns the slope of the best fitted line between auto-correlation (y) and lags (x)

        INPUTS
        =======
        title: str, optional, default value is Volatility Clustering
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the auto correlation (y) against lags (x)
        ax: ax, optional, default value is None
           ax of the plot

        RETURNS
        ========
        -popt[1]: floats
            slope of the best fitted line between auto-correlation and lag
        ax: ax of the plot
        """

        df = np.abs(self.data)
        y = df.apply(auto_correlation, maxLags=df.shape[0] - 1)
        y = y.dropna(axis=0)
        x = list(range(1, df.shape[0] - 1))

        coeffs = []

        for i in range(y.shape[1]):
            try:
                popt, pcov = curve_fit(func_powerlaw_full, x, y.iloc[:, i], maxfev=1000, p0=[0.5, 0.5, 0],
                                       bounds=([-10, -10, -10], [10., 10., 10.]))
                stats = f'Auto-Corr: {-popt[1]:.4f}\n'

            except RuntimeError:
                popt = [0, 0]

            coeffs.append(-popt[1])

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter((x[:100]), y.mean(axis=1)[:100], s=2, alpha=1)
            #     ax.set_ylim(-0.5, 0.5)
            ax.set_xscale('log')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Auto-Correlation", fontsize=12)

            return np.mean(coeffs), ax

        return np.mean(coeffs)

    def leverage_effect_testing(self, maxLag=10, plot=False, title="Leverage Effect (True)", ax=None):
        """Returns the leverage value at multiple lags of a single time series

        INPUTS
        =======
        maxLag: integer, optional, default value is 10
           the maximum number of lags to be calculated
        plot: boolean, optional, default value is False
           whether plot the auto correlation
        title: str, optional, default value is Leverage Effect (True)
           title of the plot
        ax: ax, optional, default value is Auto correlation
           title of the plot

        RETURNS
        ========
        leverage_value: list of floats
            leverage value of a single time series at each lag
        ax: ax of the plot
        """
        leverage_value = []
        for shiftValue in range(1, maxLag + 1):
            origData = self.data.values[:len(self.data) - shiftValue]
            shiftedData = self.data.shift(-shiftValue).values[:len(self.data) - shiftValue]

            firstNumerator = np.mean(np.square(shiftedData) * origData)
            secondNumerator = np.mean(self.data) * np.mean(np.square(self.data))
            denominator = np.square(np.mean(np.square(self.data)))

            value = (firstNumerator - secondNumerator) / denominator
            leverage_value.append(value)

        if plot:
            if not ax:
                ax = plt.gca()
            ax.plot(leverage_value)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag k", fontsize=12)
            ax.axhline(y=0, color="black", linestyle="dashed")

            stats = f'Auto-Corr: {np.mean(leverage_value[:10]):.4f}\n'
            ax.text(x=1, y=8, s=stats, color="red")
            ax.set_ylim(-15, 12)
            return leverage_value, ax

        return leverage_value

    def leverage_effect_testing_avg(self, maxLag=10, plot=False, title="Leverage Effect (True)", ax=None,
                                    ylim=(-15, 12)):
        """Returns the leverage value at multiple lags of multiple time series

        INPUTS
        =======
        maxLag: integer, optional, default value is 10
           the maximum number of lags to be calculated
        plot: boolean, optional, default value is False
           whether plot the auto correlation
        title: str, optional, default value is Leverage Effect (True)
           title of the plot
        ax: ax, optional, default value is Auto correlation
           title of the plot

        RETURNS
        ========
        leverage_value_mean: list of floats
            average leverage value of multiple time series at each lag
        ax: ax of the plot
        """
        leverage_value_mean = self.data.apply(leverage_effect_testing, maxLag=maxLag).mean(axis=1)

        if plot:
            if not ax:
                ax = plt.gca()
            ax.plot(leverage_value_mean)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag k", fontsize=12)
            ax.axhline(y=0, color="black", linestyle="dashed")

            stats = f'Auto-Corr: {np.mean(leverage_value_mean[:10]):.4f}\n'
            ax.text(x=1, y=8, s=stats, color="red")
            ax.set_ylim(ylim[0], ylim[1])
            return np.mean(leverage_value_mean[:10]), ax

        return np.mean(leverage_value_mean[:10])


    def coarse_fine_vol_corr_test(self, lags=10, title="Coarse-fine Volatility Correlation", plot=False, ax=None):
        """Returns the lead_lag_diff at lag 1 of a single time series

        INPUTS
        =======
        lags: integer, optional, default value is 10
           the maximum number of lags to be calculated
        title: str, optional, default value is Leverage Effect (True)
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the auto correlation
        ax: ax, optional, default value is Auto correlation
           title of the plot

        RETURNS
        ========
        lead_lag_diff: floats
            coarse volatility - fine volatility at lag 1
        ax: ax of the plot
        """

        coarse_volatility = np.abs(self.data.rolling(window=5).sum().shift(1))
        coarse_volatility = coarse_volatility.dropna().values
        fine_volatility = np.abs(self.data).rolling(window=5).sum().shift(1)
        fine_volatility = fine_volatility.dropna().values

        positive_k = []
        negative_k = []
        self_corr = []
        for i in range(1, lags + 1):
            # k > 0
            lead_corr = np.corrcoef(np.array([coarse_volatility[i:], fine_volatility[:-i]]))[0, 1]
            # k < 0
            lag_corr = np.corrcoef(np.array([coarse_volatility[:-i], fine_volatility[i:]]))[0, 1]

            positive_k.append(lead_corr)
            negative_k.append(lag_corr)

        self_corr.append(np.corrcoef(np.array([coarse_volatility, fine_volatility]))[0, 1])

        x_corr = range(-lags, lags + 1)
        x_diff = range(1, lags + 1)
        lead_lag_corr = negative_k[::-1] + self_corr + positive_k
        lead_lag_diff = np.array(positive_k) - np.array(negative_k)

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(x=x_corr, y=lead_lag_corr)
            ax.scatter(x=x_diff, y=lead_lag_diff)
            ax.axhline(y=0, color="black", linestyle="dashed")
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Auto-Correlation", fontsize=12)
            return lead_lag_diff[0], ax

        return lead_lag_diff[0]

    def coarse_fine_vol_corr_test_avg(self, lags=10, title="Coarse-fine Volatility Correlation", plot=False, ax=None):
        """Returns the lead_lag_diff at lag 1 of multiple time series

        INPUTS
        =======
        lags: integer, optional, default value is 10
           the maximum number of lags to be calculated
        title: str, optional, default value is Leverage Effect (True)
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the auto correlation
        ax: ax, optional, default value is None
           ax of the plot

        RETURNS
        ========
        lead_lag_diff: floats
            average value of (coarse volatility - fine volatility) at lag 1
        ax: ax of the plot
        """
        allData = self.data
        lead_lag_corrs = []
        lead_lag_diffs = []
        for i in range(allData.shape[1]):
            data = allData.iloc[:, i]
            coarse_volatility = np.abs(data.rolling(window=5).sum().shift(1))
            coarse_volatility = coarse_volatility.dropna().values
            fine_volatility = np.abs(data).rolling(window=5).sum().shift(1)
            fine_volatility = fine_volatility.dropna().values

            positive_k = []
            negative_k = []
            self_corr = []
            for i in range(1, lags + 1):
                # k > 0
                lead_corr = np.corrcoef(np.array([coarse_volatility[i:], fine_volatility[:-i]]))[0, 1]
                # k < 0
                lag_corr = np.corrcoef(np.array([coarse_volatility[:-i], fine_volatility[i:]]))[0, 1]

                positive_k.append(lead_corr)
                negative_k.append(lag_corr)

            self_corr.append(np.corrcoef(np.array([coarse_volatility, fine_volatility]))[0, 1])

            x_corr = range(-lags, lags + 1)
            x_diff = range(1, lags + 1)
            lead_lag_corr = negative_k[::-1] + self_corr + positive_k
            lead_lag_diff = np.array(positive_k) - np.array(negative_k)

            lead_lag_corrs.append(lead_lag_corr)
            lead_lag_diffs.append(lead_lag_diff)

        lead_lag_diffs_avg = np.array(lead_lag_diffs).mean(axis=0)
        lead_lag_corrs_avg = np.array(lead_lag_corrs).mean(axis=0)

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(x=x_corr, y=lead_lag_corrs_avg)
            ax.scatter(x=x_diff, y=lead_lag_diffs_avg)
            ax.axhline(y=0, color="black", linestyle="dashed")

            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Auto-Correlation", fontsize=12)

            return lead_lag_diffs_avg[0], ax

        return lead_lag_diffs_avg[0]

    def gain_loss_asymmetry(self, title="Gain/loss (True)", plot=False, ax=None):
        df = pd.DataFrame(self.data)

        pos_time_ticks = []
        neg_time_ticks = []

        for i in range(df.shape[1]):
            data = pd.Series([math.exp(num) for num in df.iloc[:, i].values])
            for i in range(0, len(data)):
                array = data.values[i:].cumprod()

                try:
                    pos_time_tick = np.where(array > 1.1)[0][0]
                    pos_time_ticks.append(pos_time_tick)
                except IndexError:
                    pass

                try:
                    neg_time_tick = np.where(array < 0.9)[0][0]
                    neg_time_ticks.append(neg_time_tick)
                except IndexError:
                    pass

        pos_time_ticks = np.array(pos_time_ticks)
        pos_time_counts = np.bincount(pos_time_ticks)
        pos_array = np.nonzero(pos_time_counts)[0]
        pos_max_index = pos_array[np.argmax(pos_time_counts[pos_array])]

        neg_time_ticks = np.array(neg_time_ticks)
        neg_time_counts = np.bincount(neg_time_ticks)
        neg_array = np.nonzero(neg_time_counts)[0]
        neg_max_index = neg_array[np.argmax(neg_time_counts[neg_array])]

        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(x=pos_array, y=pos_time_counts[pos_array] / len(pos_time_ticks), s=3, alpha=0.7, label="Pos")
            ax.scatter(x=neg_array, y=neg_time_counts[neg_array] / len(neg_time_ticks), s=3, alpha=0.7, label="Neg",
                       color="red")
            ax.set_ylim(0, 0.025)
            ax.set_xscale('log')
            ax.set_xlabel("lag K", fontsize=12)
            ax.legend()
            ax.set_title(title)

            return pos_max_index - neg_max_index, ax

        return pos_max_index - neg_max_index

    def variance_ratio_test(self, truePriceData, lags = [2, 3, 4, 5, 10, 20, 21, 30, 40, 50, 100], title="Statistics of Variance Ratio Test", ax=None):
        """Returns the boxplot of VRT at different lags for multiple time series

        INPUTS
        =======
        truePriceData: pandas Series, required
            True price data to compare with
        lags: list of integer, optional, default value is [2, 3, 4, 5, 10, 20, 21, 30, 40, 50, 100]
            list of lags to be calculated
        title: str, optional, default value is Statistics of Variance Ratio Test
           title of the plot
        ax: ax, optional, default value is None
           ax of the plot

        RETURNS
        ========
        lead_lag_diff: floats
            average value of (coarse volatility - fine volatility) at lag 1
        ax: ax of the plot
        """

        flierprops = dict(marker="o", markerfacecolor='red', markersize=5,
                          markeredgecolor='none')

        trueData = np.log(truePriceData)
        true_test_stats = []
        for lag in lags:
            vr = VarianceRatio(trueData, lags=lag)
            true_test_stats.append(vr.stat)

        data = self.data.applymap(lambda x: math.exp(x)).cumprod()
        lag_test_statistics_dict = {}

        for lag in lags:
            test_stats = []
            for i in range(len(data.columns)):
                vr = VarianceRatio(data.iloc[:, i], lags=lag)
                test_stats.append(vr.stat)

            lag_test_statistics_dict[lag] = test_stats

        if ax is None:
            ax = plt.gca()

        box_plot_VR_df = pd.DataFrame(lag_test_statistics_dict)
        ax.boxplot(box_plot_VR_df.T.reset_index(drop=True), flierprops=flierprops)
        ticks = ax.get_xticks()
        line_plot = ax.plot(ticks, true_test_stats)
        ax.set_xticks(range(len(lags) + 1))
        ax.set_xticklabels([str(lag) for lag in [0] + lags])
        ax.set_title(title)
        #     ax.set_ylim(-3,4)
        return ax

    def hurst(self):
        """Returns the Hurst Exponent of a single time series

        RETURNS
        ========
        Hurst_exponent : floats
            Hurst_exponent
        """
        ts = list(self.data)
        N = len(ts)
        if N < 20:
            raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

        max_k = int(np.floor(N / 2))
        R_S_dict = []
        for k in range(10, max_k + 1):
            R, S = 0, 0

            # split ts into subsets
            subset_list = [ts[i:i + k] for i in range(0, N, k)]
            if np.mod(N, k) > 0:
                subset_list.pop()

            # calc mean of every subset
            mean_list = [np.mean(x) for x in subset_list]
            for i in range(len(subset_list)):
                cumsum_list = pd.Series(subset_list[i] - mean_list[i]).cumsum()
                R += max(cumsum_list) - min(cumsum_list)
                S += np.std(subset_list[i])
            R_S_dict.append({"R": R / len(subset_list), "S": S / len(subset_list), "n": k})

        log_R_S = []
        log_n = []

        for i in range(len(R_S_dict)):
            R_S = (R_S_dict[i]["R"] + np.spacing(1)) / (R_S_dict[i]["S"] + np.spacing(1))
            log_R_S.append(np.log(R_S))
            log_n.append(np.log(R_S_dict[i]["n"]))

        Hurst_exponent = np.polyfit(log_n, log_R_S, 1)[0]
        return Hurst_exponent

    def hurst_all(self, title="Hurst distirbution", plot=False, ax=None):
        """Returns the Hurst Exponent value of multiple time series

        INPUTS
        =======
        title: str, optional, default value is Hurst distribution
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the histogram of Hurst exponent
        ax: ax, optional, default value is None
           ax of the plot

        RETURNS
        ========
        hurstAll: list of floats
            list of hurst value for multiple time series
        ax: ax of the plot
        """

        hurstAll = (self.data).apply(hurst, axis=0)

        cut_bins = np.linspace(hurstAll.min(), hurstAll.max(), 30)
        hurstBins = pd.cut(hurstAll, bins=cut_bins)

        if plot:
            if ax is None:
                ax = plt.gca()
            hurstBins.value_counts(sort=False).plot.bar(rot=0, color="royalblue", ax=ax)
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels=labels, rotation=45)
            ax.set_title(label=title)
            return hurstAll.median(), ax

        return hurstAll

    def DTW_value(self, pairCounts=100):
        """Returns the similarity measure (DTW value) of multiple time series

        INPUTS
        =======
        pairCounts: integer, optional, default value is 100
           the number of pairs of time series to be calculated

        RETURNS
        ========
        pacf_list: list of floats
            DTW distance between two time series
        """
        random.seed(616)
        distances = []

        for _ in range(pairCounts):
            random1, random2 = random.sample(range(0, self.data.shape[1]), 2)
            distance, path = fastdtw(self.data.iloc[:, random1].values, self.data.iloc[:, random2].values,
                                     dist=euclidean)
            distances.append(distance)
        return distances

    def partial_acf(self, nlags=100, plot=False, title="PACF", ax=None):
        """Returns the partial auto correlation of a single time series

        INPUTS
        =======
        nlags: integer, optional, default value is 10
           the maximum number of lags to be calculated
        plot: boolean, optional, default value is False
           whether plot the partial auto correlation of nlags
        title: str, optional, default value is Auto correlation
           title of the plot
        ax: ax, optional, default value is Auto correlation
           title of the plot

        RETURNS
        ========
        pacf_list: list of floats
            partial auto correlation from lag 1 to nlags
        ax: ax of the plot
        """

        pacf_list = pacf(self.data, nlags=nlags)[1:]
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(x=range(1, nlags + 1), y=pacf_list, s=4)
            ax.set_ylim(-1, 1)
            ax.set_xscale('log')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Partial Auto-Correlation", fontsize=12)
            stats = f'PACf: {np.mean(pacf_list[:5]):.4f}\n'
            ax.text(x=1, y=.65, s=stats, color="red")
            return pacf_list, ax

        return pacf_list

    def partial_acf_avg(self, nlags=100, title="PACF", plot=False, ax=None):
        """Returns the average value of partial auto correlation of multiple time series from lag 1 to nlags

        INPUTS
        =======
        nlags: integer, optional, default value is 10
           the maximum number of lags to be calculated
        title: str, optional, default value is Auto correlation
           title of the plot
        plot: boolean, optional, default value is False
           whether plot the partial auto correlation of nlags
        ax: ax, optional, default value is Auto correlation
           title of the plot

        RETURNS
        ========
        pacf_mean: floats
            average partial auto correlation from lag 1 to lag 5
        ax: ax of the plot
        """

        df = self.data.apply(partial_acf, nlags=nlags, plot=False, title="PACF")
        pacf_mean = df.mean(axis=1)
        print(pacf_mean)
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.scatter(x=range(1, nlags + 1), y=pacf_mean, s=4)
            ax.set_ylim(-1, 1)
            ax.set_xscale('log')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("lag K", fontsize=12)
            ax.set_ylabel("Auto-Correlation", fontsize=12)
            stats = f'Auto-Corr: {np.mean(pacf_mean[:5]):.4f}\n'
            ax.text(x=1, y=.65, s=stats, color="red")

            return np.mean(pacf_mean[:5]), ax

        return np.mean(pacf_mean[:5])


# garch_constant = pd.read_csv("./data/garch_constant.csv")
#
# fig, axes = plt.subplots(2, 3, figsize=(14,10))
# me = Model_Evaluation(garch_constant.iloc[:, :10])
# me.variance_ratio_test(SSE_data["close"], lags = [2, 3, 4, 5, 10, 20, 21, 30, 40, 50, 100], title="Statistics of Variance Ratio Test", ax=None)

# auto_corr, ax = me.auto_correlation_avg(maxLags = 100, title = "Auto correlation Avg (GARCH)", plot = True, ax = axes[0,1])
# pacf, ax = me.partial_acf_avg(nlags = 100, title = "PACF Avg (GARCH)", plot = True, ax = axes[0,2])
# fat_tail_coeff = me.fat_tail_dist_cdf()
# garchHurst = me.hurst_all()
#
# plt.show()
# print(auto_corr)
# print(fat_tail_coeff)
# print(garchHurst)
