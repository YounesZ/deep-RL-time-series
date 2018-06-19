import pywt
import numpy as np
import pandas as pd
from copy import deepcopy
import pywt


def get_chunks(data_length, chunk_size=2**16):
    start   =   list( range(0, data_length, chunk_size) )
    return start, chunk_size


def chunk_file(data, start, chunk_size):
    # Get end
    finish  =   min( len(data), start+chunk_size )
    return data.loc[start:finish], finish


def to_series(data):
    # Test for data type
    if not type(data) is pd.Series:
        return pd.Series(data=data, name='unlabelled_time_series')
    return data


def do_transform(data, wtype='sym2', nlevels=5):
    # Make sure input is a time series
    data    =   to_series(data)
    # Get chunks
    data_l  =   len(data)
    st, chs =   get_chunks(data_l)
    # Prep containers
    colNm   =   ['psi_{}'.format(x) for x in range(nlevels)] + ['phi_{}'.format(nlevels-1)]
    dwt     =   pd.DataFrame(data=[[0] * (nlevels+1)]*data_l, columns=colNm)
    # Loop on chunks
    for ist in st:
        # chunk the file
        chunk, ifin =   chunk_file(data, ist, chs)
        # Wavelet transform
        w   =   pywt.Wavelet(wtype)
        a   =   deepcopy(chunk)
        ca  =   []
        cd  =   []
        for i in range(nlevels):
            (a, d)  =   pywt.dwt(a, w, pywt.Modes.smooth)
            ca.append(a)
            cd.append(d)
        # iWavelet transform: scaling coefficients
        rec_a   =   []
        for i, coeff in enumerate(ca):
            coeff_list  =   [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w))
        # iWavelet transform: wavelet coefficients
        rec_d   =   []
        for i, coeff in enumerate(cd):
            coeff_list  =   [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w))
        # Append to container
        dwt.loc[ist:ifin-1, colNm[:-1]]   =   np.transpose([x[:ifin - ist] for x in rec_d])
        dwt.loc[ist:ifin-1, colNm[-1]]    =   rec_a[-1][:(ifin - ist)]
    return dwt


# Launcher
if  __name__=="__main__":
    # Load dataset
    file    =   '/home/younesz/Downloads/bc_new.csv'
    ds      =   pd.read_csv(file)
    # Get wavelet transform
    dwt     =   do_transform( ds['Close'], nlevels=5 )
    # Time-differentiated version
    dwt_td  =   do_transform( np.hstack( (0, (ds['Close'][1:].values-ds['Close'][:-1].values)) ), nlevels=5 )