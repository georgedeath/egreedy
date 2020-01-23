import pandas as pd
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def compute_angle(a, b, c):
    origin = np.zeros(2)
    ab = (b-origin) - (a-origin)
    ac = (c-origin) - (a-origin)
    if (np.linalg.norm(ab) * np.linalg.norm(ac)) > 0:
        theta = np.arccos(np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac)))
    else:
        theta = 0
    return theta, 2*np.pi - theta

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)

plt.ion()
TJunction = pd.read_csv('boundary_pitz.csv',delimiter=',',header=None)
boundingBox1 = pd.read_csv('boundary.csv',delimiter=',',header=None)

fixedPoint1 = pd.read_csv('fixed.csv',delimiter=',',header=None)

fixed_array = np.array(fixedPoint1)
boundary_array = np.array(boundingBox1)


fixedPoints1 =[]
for (i,j), val in np.ndenumerate(fixed_array):
    fixedPoints1.append(boundary_array[val])
fixedPoints1array = np.array(fixedPoints1)

"""
for i in range(0,len(fixedPoint1)):
    fixedPoints1.append([boundingBox1.iloc[fixedPoint1.iloc[i,0],0],boundingBox1.iloc[fixedPoint1.iloc[i,0],1]])
    fixedPoints1array = np.asarray(fixedPoints1)
"""

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('x (m)', fontsize = 30,  fontname = 'Times New Roman',style='italic')
ax1.set_ylabel('y (m)', fontsize = 30, fontname = 'Times New Roman',style='italic')
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)

ax1.plot(TJunction.iloc[:,0],TJunction.iloc[:,1], label='PitzDaily boundary', linewidth=2.5,c='k')
ax1.plot(boundingBox1.iloc[:,0],boundingBox1.iloc[:,1], '--', label='Bounding-box',  marker='o', linewidth=1.5,c='b')

ax1.scatter(fixedPoints1array[:,0],fixedPoints1array[:,1],label='Fixed points', marker='s', s=80, c='k')
ax1.set(xlim=[-0.05, 0.30], ylim=[-0.06, 0.06])
#plt.axis('scaled')
leg = plt.legend()
plt.setp(ax1.get_legend().get_texts(), fontsize='20') # for legend text
plt.tight_layout()
