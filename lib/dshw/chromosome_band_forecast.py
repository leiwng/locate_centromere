import numpy as np
from .dshw import double_seasonal


def chromosome_band_forecast(s, x):
    """
    Forecasting chromosome band sequence using double seasonal Holt-Winters method.
    :param s: season number, for how many data points in a season
    :param x: the array of multiple seasonal chromosome band sequence data. the length of x must be longer than s*2.
    :return: forecasted one season size values
    """
    # insorted_x = list(np.array(running_median_insort(x, s)))
    insorted_x = list(np.array(x)) # 不做平滑
    return double_seasonal(
        x=insorted_x,   # 要求输入序列的长度大于下面m2实参值的两倍,如果m2是300,那么输入序列的长度至少要大于600
        m=s,
        m2=2*s,         # 第二个周期为第一个周期的两倍
        forecast=s,     # 预测一条染色体序列的长度
        alpha=None,
        beta=None,
        gamma=None,
        delta=None,
        autocorrelation=None,
        initial_values_optimization=None,
        optimization_type="MSE",
    )[0][:]
