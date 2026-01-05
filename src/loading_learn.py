import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
def generate_simple_ppg(heart_rate=70):
    duration =10
    sampling_rate=125
    total_samples=duration*sampling_rate
    
    #creating an array from 0 to 10 with 1250 points
    time=np.linspace(0,duration,total_samples)
    
    #create empty signal
    ppg=np.zeros(total_samples)
    
    beats_per_second=heart_rate/60
    time_between_beats=1/beats_per_second
    
    num_beats=int(heart_rate*duration/60)
    
    print(f"Heart rate:{heart_rate} bpm")
    print(f"Time between beats: {time_between_beats:.2f} seconds")
    print(f"Number of beats in 10 sec:{num_beats}")
    
    for beat_number in range(num_beats):
        beat_time=beat_number+time_between_beats
    #for each time point calcl dist from heartbeat
    for i, t in enumerate(time):
        distance=t-beat_time
        
        #create Gaussian bump (bell curve)
        ppg[i]=np.exp(-(distance ** 2)/0.05)
        
    return ppg,time

if __name__=="__main__":
    ppg,time=generate_simple_ppg()
    print(f"Generated PPG signal with {len(ppg)} samples")
    print(f"First 10 values : {ppg[:10]}")

    # Plot it
    plt.figure(figsize=(12, 4))
    plt.plot(time, ppg)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Simple PPG Signal - One Heartbeat')
    plt.grid(True)
    plt.show()  