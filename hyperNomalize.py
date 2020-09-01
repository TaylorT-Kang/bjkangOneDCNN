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

def spectralNormalize(S):
    minVal = np.min(S)
    maxVal = np.max(S)

    normalizeS = S - minVal
    if maxVal == minVal:
        normalizeS = np.zeros(S.shape)
    else:
        normalizeS = normalizeS / (maxVal - minVal)

    return normalizeS