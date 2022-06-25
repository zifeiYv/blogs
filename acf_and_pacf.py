# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 数据读入
price_data = pd.read_csv('data/price-data.csv')

# 绘制acf图
plot_acf(price_data)
plt.show()

# 通过设置不同的nlags,来验证自相关系数中的均值是根据全序列求得的
nlags = 5
print(tsa.acf(price_data)[nlags])

y_mean = np.mean(price_data.values)
print(np.sum((price_data.values[:-nlags] - y_mean) * (price_data.values[nlags:] - y_mean)
             / np.sum((price_data.values - y_mean)**2)))
