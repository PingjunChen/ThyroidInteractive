# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import hdf5storage


if __name__ == "__main__":
    all_ret = {}
    RetMatPath = "./roiFeas/RetInd/TIdx02.mat"
    RetIndData = hdf5storage.loadmat(RetMatPath)
    RetResults = RetIndData['TIdx02']

    ret_k = np.arange(1, 12, 2)
    bit_inds = np.arange(0, 3)

    
