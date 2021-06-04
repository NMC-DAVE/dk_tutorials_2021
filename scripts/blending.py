# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:00:40 2021

@author: MJW
"""
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt

latest_err = nc.Dataset('last_err.nc','r')['data'][:] #上个时次的滑动误差  4月1-30日，alpha=0.2
latest_mae = nc.Dataset('last_mae.nc','r')['data'][:] #上个时次的滑动mae   4月1-30日, 误差绝对值的滑动

(f0, fc0), (f1, fc1), (f2, fc2), ...., (fn, ) mae_a
...                                           mae_b

#下列所用数据已经过预处理，相应的实况和预报场进行了时间对齐和空间插值处理，长度为30天，空间范围包含【110-115E,25-30N】
#######################################################################################
obs_his = nc.Dataset('observation_history.nc','r')['data'][:] #训练所用实况场（CLDAS）  5月份
grd_tru = nc.Dataset('ground_truth.nc','r')['data'][:] #验证所用实况场（CLDAS）   与forecast.nc对应
fore_his = nc.Dataset('forecast_history.nc','r')['data'][:] #训练所用预报场   5月预报场
forecast = nc.Dataset('forecast.nc','r')['data'][:] #用于进行订正和融合的预报场   错开一个起报时间
#######################################################################################

alpha=0.2
revised_mae=[];blend_mae=[];forecast_mae=[]
for i in range(30):
    err_update=fore_his[i]-obs_his[i]
    latest_err = (1-alpha)*latest_err + alpha*err_update
    latest_mae = (1-alpha)*latest_mae + alpha*abs(err_update)
    fore_tmp = forecast[i] - latest_err
    wgt=[]
    for j in range(3):
        wgt.append((1/latest_mae[j])/(np.sum((1/latest_mae),axis=0)))
    wgt=np.array(wgt)
    forecast_mae.append(np.abs(forecast[i]-grd_tru[i]))     # 订正前的预报MAE
    revised_mae.append(np.abs(fore_tmp-grd_tru[i]))         # 订正后的预报MAE
    blend_mae.append(np.abs(np.sum((fore_tmp*wgt),axis=0)-grd_tru[i]))   # 融合后的MAE

forecast_mae=np.mean(np.array(forecast_mae),axis=(0,3,4))
revised_mae=np.mean(np.array(revised_mae),axis=(0,3,4))
blend_mae=np.expand_dims(np.mean(np.array(blend_mae),axis=(0,2,3)),axis=0)

xticks = np.arange(6,240+1,6)
fig,ax = plt.subplots()
pd.DataFrame(np.transpose(np.concatenate((forecast_mae,revised_mae,blend_mae),axis=0),(1,0)),index=xticks).plot(ax=ax)
ax.legend(["EC_origin", "NCEP_origin",'Grapes_origin',"EC_mos","NCEP_mos",'Grapes_mos','Blending'])
ax.set_xlabel("forecast time(in hours)")
ax.set_ylabel("errors")
ax.figure.savefig("plot.pdf")



