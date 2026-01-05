# src/simple_loader.py - COMPLETE SIMPLE VERSION

import pandas as pd
import numpy as np
import pickle
import os

def generate_ppg_from_bp(systolic, diastolic, heart_rate):
    """Generate realistic PPG signal from BP values"""
    
    # Setup: 10 seconds at 125 Hz = 1250 samples
    time = np.linspace(0, 10, 1250)
    ppg = np.zeros(1250)
    
    # BP affects pulse shape
    # High BP → narrower pulses
    pulse_width = 0.18 - (systolic - 120) * 0.0004
    pulse_width = max(0.10, min(pulse_width, 0.25))  # Keep in range
    
    # BP affects amplitude
    amplitude = 0.8 + (systolic - 120) * 0.004
    
    # Generate heartbeats
    beats_per_second = heart_rate / 60
    time_between_beats = 1 / beats_per_second
    num_beats = int(heart_rate * 10 / 60)
    
    for beat in range(num_beats):
        beat_time = beat * time_between_beats
        
        # Add Gaussian pulse at beat_time
        for i, t in enumerate(time):
            distance = t - beat_time
            ppg[i] += amplitude * np.exp(-(distance ** 2) / (2 * pulse_width ** 2))
    
    # Add noise
    ppg += np.random.normal(0, 0.03, 1250)
    
    # Normalize
    ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
    
    return ppg


def create_dataset(csv_path):
    """Read CSV and create dataset"""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} patients in CSV")
    
    dataset = []
    skipped = 0
    
    # Process each patient
    for idx, row in df.iterrows():
        try:
            # Extract values
            sys = float(row['Systolic Blood Pressure(mmHg)'])
            dia = float(row['Diastolic Blood Pressure(mmHg)'])
            hr = float(row['Heart Rate(b/m)'])
            
            # Validate
            if pd.isna(sys) or pd.isna(dia) or pd.isna(hr):
                skipped += 1
                continue
            
            if not (70 <= sys <= 200):
                skipped += 1
                continue
            
            if not (40 <= dia <= 130):
                skipped += 1
                continue
            
            if not (40 <= hr <= 150):
                skipped += 1
                continue
            
            # Generate PPG
            ppg = generate_ppg_from_bp(sys, dia, hr)
            
            # Store
            dataset.append({
                'ppg': ppg,
                'systolic': sys,
                'diastolic': dia
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)} patients")
        
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\nCreated {len(dataset)} samples")
    print(f"Skipped {skipped} invalid patients")
    
    return dataset


def save_dataset(dataset, path='data/ppg_dataset.pkl'):
    """Save dataset"""
    
    # Create folder
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n Saved to {path}")
    
    # Stats
    sys_vals = [d['systolic'] for d in dataset]
    dia_vals = [d['diastolic'] for d in dataset]
    
    print("\nStatistics:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Systolic: {np.mean(sys_vals):.1f} ± {np.std(sys_vals):.1f} mmHg")
    print(f"  Diastolic: {np.mean(dia_vals):.1f} ± {np.std(dia_vals):.1f} mmHg")


if __name__ == "__main__":
    # Run everything
    dataset = create_dataset('data/raw/ppg.csv')
    save_dataset(dataset)
    