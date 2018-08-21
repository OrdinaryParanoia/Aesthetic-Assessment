#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:05:41 2018

@author: paranoia
"""

import os
import numpy as np
import pandas as pd

good = []
bad = []
f = open('/home/paranoia/desktop/AVA_dataset/images/AVA.txt')
addr = '/home/paranoia/desktop/AVA_dataset/images/images/'
for i, each_line in enumerate(f):
    name = addr + each_line.split(' ')[1] + '.jpg'
    if os.path.exists(name) == False:
        continue
    dist = np.linspace(0,9,10)
    temp = np.array(each_line.split(' '))
    temp2 = np.array(temp, dtype=float)
    if sum(dist*temp2[2:12])/sum(temp2[2:12]) >= 4.5:
        good.append(temp[0:12])
    else:
        bad.append(temp[0:12])
    if i == 10000:
        break

good = pd.DataFrame(good)
bad = pd.DataFrame(bad)

del good[0]
del bad[0]

good.to_csv('/home/paranoia/desktop/AVA_dataset/good.csv')
bad.to_csv('/home/paranoia/desktop/AVA_dataset/bad.csv')