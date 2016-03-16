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
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta



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
    print url
    urllib.urlretrieve(url,filename)    
    x=h5py.File(filename,'r')
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
    tf=np.bool()  # boolean array
    marr=np.array(m) # array for modes
    ll=np.array(l) # array for spectra
    tf=marr > 0 # true if value >0
    mode=max(marr)
    sp=ll[tf]
    f=freq(fchans,mode)

    mod=np.median(sp,axis=0)
    maxim=np.max(sp,axis=0)
    minim=np.min(sp,axis=0)
    pk=np.max(maxim[500:-500]) # peak
    chpk=np.argmax(maxim[500:-500]) # channel with peakf
    fpk=freq(chpk,mode) # freq of channel with peak
    #plot min max med
    aver=np.average(mod[500:-500])

#plot minmax
    fig5=plt.figure(figsize=(18.8,9.8))
    az = fig5.add_subplot(111)
    plt.plot(f[500:-500],mod[500:-500],label="Median",color='c')
    plt.plot(f[500:-500],maxim[500:-500],label="Max", color='g')
    plt.plot(f[500:-500],minim[500:-500],label="Min",color='r')
    plt.title(filename+" Mode:"+str(mode))
    plt.xlabel("Freq/MHz")
    plt.ylabel("Power/dB")
    plt.text(fpk,pk,"peak "+str(pk))
    plt.text(1200,aver,"average "+str(aver),fontsize=10,color='k', weight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    az.set_axis_bgcolor((0, 0, 0))
    plt.savefig('/home/reverb-chamber/Desktop/Ratty/minmax.png',dpi=100)
#####################
    os.remove(filename)  #comment line out if used in Windows
    return f,sp
    

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
    
#############################################################################3
#Main Program start##########################################################

det_thresh=1
time_o=str(int(time.strftime("%H"))-1)
now=time.strftime("%Y_%m_%d_")
now=now+"0"+time_o
now1=time.strftime("%Y/%m/%d/")
now1=now1+"0"+time_o

if int(time_o)>9:
    now=time.strftime("%Y_%m_%d_")+time_o
    now1=time.strftime("%Y/%m/%d/")+time_o

print now
print now1

#Load data from ratty server
f, sp=h5_proc(now,now1)
sp_av=np.zeros(np.shape(sp))

#Apply mask for self RFI
mask=np.load('/home/reverb-chamber/Desktop/Ratty/RFI_mask.npy')
sp[:,mask]=sp[:,mask-1]

detect_f=np.zeros(np.shape(sp))
for x in range(len(sp[:,1])):
    sp_av[x,:]=movingaverage(sp[x,:],200)+det_thresh
    detect_f[x,0:len(sp[x,sp[x,:]>sp_av[x,:]])]=f[sp[x,:]>sp_av[x,:]]

detect=sp*(sp>sp_av)
f_scatter=f*(sp>sp_av)
timing=np.where(sp>sp_av)

power=np.reshape(detect,np.size(detect))
power=power[power!=0]
f_scatter=np.reshape(f_scatter,np.size(f_scatter))
f_scatter=f_scatter[f_scatter!=0]
timing=timing[0]

ff=np.reshape(detect_f,(np.size(detect_f),1))
ff=ff[ff>0]
dat=np.histogram(ff,len(sp[1,:]))
count=dat[0]
freqs=dat[1]
freqs=freqs[0:-1] + (f[1]-f[0])
occ=100*count/len(sp[:,1])

#Import culprit DB
cul_data=np.load('/home/reverb-chamber/Desktop/Ratty/Cul_db.npy')
culf=cul_data[0]
cul=cul_data[1]
culf = culf.astype(np.float)
cul=cul[culf>min(freqs)]
culf=culf[culf>min(freqs)]
cul=cul[culf<max(freqs)]
culf=culf[culf<max(freqs)]

#plot occ#######################

occ_data=np.load('/home/reverb-chamber/Desktop/Ratty/occupancy.npy')
occ_data=np.roll(occ_data,1,axis=0)
occ_data[0,:]=occ
occ_scatter=np.reshape(occ_data,np.size(occ_data))
f_scatter1=np.ones([48,len(freqs)])
f_scatter1[0:48,:]=freqs
f_scatter1=np.reshape(f_scatter1,np.size(f_scatter1))
t_scatter=np.zeros([48,len(occ)])
for i in range(len(t_scatter[:,1])):
    t_scatter[i,:]=i
    
t_scatter=np.reshape(t_scatter,np.size(t_scatter))
f_scatter1=f_scatter1[occ_scatter>0]
t_scatter=t_scatter[occ_scatter>0]
occ_scatter=occ_scatter[occ_scatter>0]

ticks=datetime.now()
ticks=ticks.hour

for x in range(1,49):
    w=datetime.now() - timedelta(hours=x)
    print w.hour
    ticks=np.append(ticks,w.hour)


fig=plt.figure(figsize=(18.8,9.8))
gs = gridspec.GridSpec(3,1)
ax = plt.subplot(gs[0, :])
plt.plot(freqs,occ)
plt.grid()
#colorbar()
plt.title('RFI Occupancy')
plt.tight_layout()
ax = plt.subplot(gs[1:, :])
plt.scatter(f_scatter1,t_scatter,c=occ_scatter,lw=0,s=6)
plt.ylim(0,47)
plt.gca().invert_yaxis()
plt.yticks(range(48),ticks)
#plt.colorbar(orientation='horizontal')
plt.title('RFI Occupancy - waterfall past 48 hours')
plt.grid()
plt.tight_layout()
plt.savefig('/home/reverb-chamber/Desktop/Ratty/occ.png',dpi=100)

##################################
np.save('/home/reverb-chamber/Desktop/Ratty/occupancy.npy',occ_data)

#################
#waterfall plot
f_plot=f[500:-500]
t_plot=[time.strftime("%Y/%m/%d/")+str(int(time.strftime("%H"))-1)+':00', time.strftime("%Y/%m/%d/")+str(int(time.strftime("%H"))-1)+':15', time.strftime("%Y/%m/%d/")+str(int(time.strftime("%H"))-1)+':30',time.strftime("%Y/%m/%d/")+str(int(time.strftime("%H"))-1)+':45', time.strftime("%Y/%m/%d/")+str(int(time.strftime("%H")))+':00']

plt.figure(figsize=(18.8,9.8))
plt.imshow(sp[:,500:-500], aspect='auto')
plt.tight_layout()
#plt.gca().invert_yaxis()
plt.xticks(np.arange(0,len(f_plot),len(f_plot)/9),np.round(f_plot[0:-1:len(f_plot)/9]))
plt.yticks(np.arange(0,len(sp[:,1])+1,(len(sp[:,1]))/4),t_plot,size='small')
plt.savefig('/home/reverb-chamber/Desktop/Ratty/waterfall.png',dpi=100)

###############33
#plot scatter
fig1 = plt.figure(figsize=(18.8,9.8))
ay = fig1.add_subplot(111)
#ay = plt.axes([0.0, 0.0, 1.0, 1.0]) 
plt.scatter(f_scatter,timing,s=5,c=power/max(power), lw = 0)
plt.tight_layout()
ay.set_axis_bgcolor((0, 0, 0))
plt.title('RFI Detections')
plt.xlim(min(f),max(f))
plt.ylim(0,len(sp[:,1]))
plt.gca().invert_yaxis()
plt.yticks(np.arange(0,len(sp[:,1])+1,(len(sp[:,1]))/4),t_plot,size='small')
plt.savefig('/home/reverb-chamber/Desktop/Ratty/scatter.png',dpi=100)
###############33
#plot zoomed

for i, txt in enumerate(cul):
    ay.annotate(txt, (culf[i],len(sp[:,1])),rotation='vertical',fontsize=15,ha='left',va='bottom',color='White')
    
plt.xlim(935,945)
plt.title('RFI Detections: GSM')
plt.savefig('/home/reverb-chamber/Desktop/Ratty/site_gsm.png',dpi=100)
plt.xlim(1020,1160)
plt.title('RFI Detections: Aeronautical band')
plt.savefig('/home/reverb-chamber/Desktop/Ratty/airband.png',dpi=100)

plt.xlim(1500,1700)
plt.title('RFI Detections: Sat Comms')
plt.savefig('/home/reverb-chamber/Desktop/Ratty/sat.png',dpi=100)

