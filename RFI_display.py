# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:08:41 2015

@author: lkock
"""

import h5py
#import pylab as pl
import numpy as np
import urllib
import sys
import os
import sched
import time
#import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


"""
get h5 file from server
read in spectra and mode
plot with information
"""

def h5_proc(date_in,date_in1):
    
    tstr=date_in+".h5"    
    filename=tstr
    tstr=date_in1+".h5"

    url="http://rfimonitor.kat.ac.za/rfi_data/"+tstr
    urllib.urlretrieve(url,filename)
    
    x=h5py.File(filename,'r')
    #x.keys()
    l=x['spectra']
    m=x['mode']
    
      
    def freq(chans,mode):
        if mode==4:
            freq=1800000000.0 +chans*27465.8203125
        if mode ==3:
            freq = 855000000.0 +chans*26092.529296875
        if mode ==2:
            freq= 600000000.0 + chans*18310.546875
        if mode ==1:
            freq=0+chans*27465.8203125
        return freq/1e6
    
    fchans=np.array(range(l.shape[1]))
    times=l.shape[0]
    #pl.grid(True)
    
    tf=np.bool()  # boolean array
    marr=np.array(m) # array for modes
    ll=np.array(l) # array for spectra
    tf=marr > 0 # true if value >0
    mode=max(marr)
    sp=ll[tf]
    f=freq(fchans,mode)
    os.remove(filename)
    return f,sp
    

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
    


#Start the fun



time_o=str(int(time.strftime("%H"))-1)
now=time.strftime("%Y_%m_%m_")
now=now+time_o
now1='2015/11/30/08'
now='2015_11_30_08'

f, sp=h5_proc(now,now1)
sp_av=np.zeros(np.shape(sp))
#detect=np.zeros(np.shape(sp))
detect_f=np.zeros(np.shape(sp))
for x in range(len(sp[:,1])):
    sp_av[x,:]=movingaverage(sp[x,:],200)+det_thresh
    
#    detect[x,0:len(sp[x,sp[x,:]>sp_av[x,:]])]=sp[x,sp[x,:]>sp_av[x,:]]
    detect_f[x,0:len(sp[x,sp[x,:]>sp_av[x,:]])]=f[sp[x,:]>sp_av[x,:]]


ff=np.reshape(detect_f,(np.size(detect_f),1))
ff=ff[ff>0]
dat=np.histogram(ff,len(sp[1,:]))
count=dat[0]
freqs=dat[1]
freqs=freqs[0:-1] + (f[1]-f[0])
occ=100*count/len(sp[:,1])



fig=plt.figure()
ax = fig.add_subplot(111, aspect=5)
plt.bar(freqs,occ)
plt.tight_layout()
plt.savefig('/var/www/html/occ.png',dpi=1000,transparent=True)


fig=plt.figure()
ax = fig.add_subplot(111)

plt.imshow(sp[:,500:-500], aspect='auto')
plt.tight_layout()
#plt.title('Occupancy in 1 h')
plt.savefig('/var/www/html/waterfall.png',dpi=1000)

#plt.imshow(sp[:,500:-500], aspect='auto',vmin=-200, vmax=-100)
