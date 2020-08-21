import numpy as np

def hyperNormalize(M):
    minVal = np.min(M)
    maxVal = np.max(M)

    normalizeM = M - minVal
    if maxVal == minVal:
        normalizeM = np.zeros(M.shape)
    else:
        normalizeM = normalizeM / (maxVal - minVal)

    return normalizeM