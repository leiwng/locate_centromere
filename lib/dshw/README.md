# dshw_4_chromosome_band_forecast

使用Double Seasonal Holt Winter's(dshw)算法预测染色体序列.

## 文件说明

chromosome_band_forecast.py : 主程序

./common_lib/dshw.py : dshw算法实现

## 函数接口

```python
def chromosome_band_forecast(s, x):
    """
    Forecasting chromosome band sequence using double seasonal Holt-Winters method.
    :param s: season number, for how many data points in a season
    :param x: the array of multiple seasonal chromosome band sequence data. the length of x must be longer than s*2.
    :return: forecasted one season size values
    """
```
