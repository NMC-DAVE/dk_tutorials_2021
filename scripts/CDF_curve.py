# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

def construct_CDFs(obs_mat, fst_mat, b_show=False): 
    """
    基于 实况和预报数据 制作 CDF 曲线
    并显示

    Parameters: 
    -----------
    obs_mat:        实况数据
    fst_mat:        预报数据

    b_show:         是否显示图像

    Returns:
    --------
    fst_CDF:        预报 CDF 曲线，实际上是1维数组，分别对应 idxs 各分位的值
    obs_CDF:        实况 CDF 曲线，同上
    """

    # CDF 曲线的分位
    idxs = np.array([0.0001, 0.0005, 0.001, 0.005] + [i*0.01 for i in range(1, 100)] + [0.995, 0.999, 0.9995, 0.9999])

    idx_fst = (idxs*fst_mat.size+0.5).astype(np.int32)
    idx_obs = (idxs*obs_mat.size+0.5).astype(np.int32)
    
    idx_fst[idx_fst == fst_mat.size] = fst_mat.size-1
    idx_obs[idx_obs == obs_mat.size] = obs_mat.size-1

    fst_CDF = np.sort(fst_mat.reshape(-1))[idx_fst]
    obs_CDF = np.sort(obs_mat.reshape(-1))[idx_obs]

    if b_show: 
        plt.plot(idxs, fst_CDF, 'r-', label='Forecast')
        plt.plot(idxs, obs_CDF, 'b-', label='Observation')
        plt.legend()
        plt.show()

    return fst_CDF, obs_CDF


def do_qm(fst_CDF, obs_CDF, fst): 
    """
    进行分位映射订正

    Parameters:
    -----------
    fst_CDF:        预报的CDF曲线 
    obs_CDF:        实况的CDF曲线

    fst:            预报值

    Returns:
    --------
    val:            经分位映射订正后的预报
    """
    # fst 在 fst_CDF 中的位置
    idx = np.sum(fst_CDF < fst)
    idx = (idx - 1) if (idx==fst_CDF.size) else idx
    return obs_CDF[idx]


if __name__ == "__main__":
    a1 = np.random.rand(200)
    b1 = np.random.rand(500)

    fst_CDF, obs_CDF = construct_CDFs(a1, b1, True)
    val = do_qm(fst_CDF, obs_CDF, 0.7888)
    print(val)
