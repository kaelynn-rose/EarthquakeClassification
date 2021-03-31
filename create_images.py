'''

Code by Kaelynn Rose (c)
Created on 3/24/2021

'''

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from joblib import Parallel,delayed

# paths to csv and hdf5 (waveform/signal) files
noise_csv_path = 'data/chunk1/chunk1.csv'
noise_sig_path = 'data/chunk1/chunk1.hdf5'
eq1_csv_path = 'data/chunk2/chunk2.csv'
eq1_sig_path = 'data/chunk2/chunk2.hdf5'
eq2_csv_path = 'data/chunk3/chunk3.csv'
eq2_sig_path = 'data/chunk3/chunk3.hdf5'
eq3_csv_path = 'data/chunk4/chunk4.csv'
eq3_sig_path = 'data/chunk4/chunk4.hdf5'
eq4_csv_path = 'data/chunk5/chunk5.csv'
eq4_sig_path = 'data/chunk5/chunk5.hdf5'
eq5_csv_path = 'data/chunk6/chunk6.csv'
eq5_sig_path = 'data/chunk6/chunk6.hdf5'

# read the noise and earthquake csv files into separate dataframes:
noise = pd.read_csv(noise_csv_path)
earthquakes_1 = pd.read_csv(eq1_csv_path)
earthquakes_2 = pd.read_csv(eq2_csv_path)
earthquakes_3 = pd.read_csv(eq3_csv_path)
earthquakes_4 = pd.read_csv(eq4_csv_path)
earthquakes_5 = pd.read_csv(eq5_csv_path)

# create a dataframe with all of the earthquake data
#earthquakes = earthquakes_1.append(earthquakes_2)

# create a dataframe with all of the noise data
#full_data = earthquakes.append(noise)

# filtering the dataframe: uncomment if needed
#df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
#print(f'total events selected: {len(df)}')


# making a list of trace names for the first earthquake set
eq1_list = earthquakes_1['trace_name'].to_list()

# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(eq1_sig_path, 'r')
eq1_waveforms = []
count = 0
for c, evi in enumerate(eq1_list):
    dataset = dtfl.get('data/'+str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    eq1_waveforms.append(data)
    count +=1
    print('working on eq1 waveform ' +str(evi) + ' number ' +str(count))

# making a list of trace names for the second earthquake set
eq2_list = earthquakes_2['trace_name'].to_list()

# retrieving selected waveforms from the hdf5 file:
dtf2 = h5py.File(eq2_sig_path, 'r')
eq2_waveforms = []
count = 0
for c, evi in enumerate(eq2_list):
    dataset = dtf2.get('data/'+str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    eq2_waveforms.append(data)
    count +=1
    print('working on eq2 waveform ' +str(evi) + ' number ' +str(count))
    
# making a list of trace names for the noise set
noise_list = noise['trace_name'].to_list()

# retrieving selected waveforms from the hdf5 file:
dtf3 = h5py.File(noise_sig_path, 'r')
noise_waveforms = []
count = 0
for c, evi in enumerate(noise_list):
    dataset = dtf3.get('data/'+str(evi))
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    noise_waveforms.append(data)
    count +=1
    print('working on noise waveform ' +str(evi) + ' number ' +str(count))


np.save('')


## Create Images (in parallel)

# making a list of trace names for the toy dataset
traces = noise['trace_name'].to_list()[10000:20000]
set = 'noise'
path = noise_sig_path

def make_images(i):
    # retrieving selected waveforms from the hdf5 file:
    try:
        dtfl = h5py.File(path, 'r')
        dataset = dtfl.get('data/'+str(traces[i]))
        # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
        data = np.array(dataset)
        print('working on ' + set + ' waveform ' +str(traces[i]) + ' number ' +str(i))
        
        fig, ax = plt.subplots(figsize=(2,2))
        ax.specgram(data[:,2],Fs=100,NFFT=256,cmap='gray');
        #ax.set_ylim([0,20])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig('specs/fig_spec_'+traces[i]+'.png',bbox_inches='tight',transparent = True,pad_inches=0,dpi=50)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(2,2))
        ax.specgram(data[:,2],Fs=100,NFFT=256,cmap='gray',vmin=-10,vmax=25);
        #ax.set_ylim([0,20])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig('specs_alt/fig_specalt_'+traces[i]+'.png',bbox_inches='tight',transparent = True,pad_inches=0,dpi=50)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(2,2))
        n = 4
        ax.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1)
        #ax.set_ylim([0,20])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig('waves/fig_wave_'+traces[i]+'.png',bbox_inches='tight',dpi=50)
        plt.close()
    except:
        print('String index out of range')


start = time.time()
print(start)
Parallel(n_jobs=-2)(delayed(make_images)(i) for i in range(0,len(traces))) # run make_images loop in parallel on all but 2 cores for each value of i
end = time.time()
print(f'Took {end-start} s')



## Make images for the big dataset

eqlist = noise['trace_name'].to_list()
#random_signals = np.random.choice(eqlist,80000,replace=False)
starts = list(np.linspace(80000,240000,32))
ends = list(np.linspace(85000,245000,32))
eqpath = noise_sig_path
set = 'noise'

count = 0
for n in range(0,len(starts)):
    traces = eqlist[int(starts[n]):int(ends[n])]
    path = eqpath
    count += 1
    
    def make_images(i):
        # retrieving selected waveforms from the hdf5 file:
        try:
            dtfl = h5py.File(path, 'r')
            dataset = dtfl.get('data/'+str(traces[i]))
            # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
            data = np.array(dataset)
            print('working on ' + set + ' waveform ' +str(traces[i]) + ' chunk '+str(count) + ' number ' +str(i))
            
            fig, ax = plt.subplots(figsize=(3,2))
            ax.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1)
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig('images/big_data_random/waves/'+traces[i]+'.png',bbox_inches='tight',dpi=50)
            plt.close()
            
            fig, ax = plt.subplots(figsize=(3,2))
            ax.specgram(data[:,2],Fs=100,NFFT=256,cmap='gray',vmin=-10,vmax=25);
            ax.set_xlim([0,60])
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.savefig('images/big_data_random/specs/'+traces[i]+'.png',bbox_inches='tight',transparent = True,pad_inches=0,dpi=50)
            plt.close()
            
        except:
            print('String index out of range')


    start = time.time()
    print(start)
    Parallel(n_jobs=-2)(delayed(make_images)(i) for i in range(0,len(traces))) # run make_images loop in parallel on all but 2 cores for each value of i
    end = time.time()
    print(f'Took {end-start} s')




# YARR.AU_20180115060030_NO
# YARR.AU_20180115055906_NO
# TUC.IU_20180115000909_NO
# SOFL.GB_20180116180543_NO
