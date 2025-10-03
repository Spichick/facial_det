# models/lbp_features.py
import numpy as np
from skimage.feature import local_binary_pattern

# 典型参数：uniform LBP
LBP_P = 8
LBP_R = 2
LBP_METHOD = "uniform"
LBP_NBINS = LBP_P + 2    # uniform 模式下的直方图桶数

def lbp_hist(gray_48x48):
    """
    输入：灰度图 (48x48 或任意灰度，但建议先 resize 到 48x48)
    输出：归一化后的 LBP 直方图 (长度=LBP_NBINS, dtype=float32)
    """
    lbp = local_binary_pattern(gray_48x48, LBP_P, LBP_R, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_NBINS, range=(0, LBP_NBINS))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-8)
    return hist
