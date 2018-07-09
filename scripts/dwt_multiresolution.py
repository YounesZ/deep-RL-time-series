import pywt
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy



def dwt_multiresolution(data, wtype='sym2', nlevels=5):
    # Init
    w   =   pywt.Wavelet(wtype)
    a   =   deepcopy(data)
    ca  =   []
    cd  =   []
    for i in range(nlevels):
        (a, d)  =   pywt.dwt(a, w, mode='per', axis=0)
        ca.append(a)
        cd.append(d)
    # iWavelet transform: wavelet coefficients
    rec_d   =   []
    for i, coeff in enumerate(cd):
        coeff_list  =   [None, coeff.T] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w)[:len(data)])
    # iWavelet transform: scaling coefficients
    coeff_list      =   [ca[-1].T, None] + [None] * (nlevels - 1)
    rec_d.append(pywt.waverec(coeff_list, w)[:len(data)])
    return rec_d


def test_dwt_multiresolution():
    sig = np.random.random(1000)
    dwt = dwt_multiresolution(sig, nlevels=10)
    plt.figure();
    plt.plot(sig, label='original signal')
    plt.plot( np.sum(dwt, axis=0), '--r', label='reconstructed')
    #[plt.plot(y, label='psi_{}'.format(x)) for x,y in enumerate(dwt[:-1])]
    #plt.plot(dwt[-1], label='phi_{}'.format(10))
    plt.legend()


# LAUNCHER
if __name__=='__main__':
    test_dwt_multiresolution()
