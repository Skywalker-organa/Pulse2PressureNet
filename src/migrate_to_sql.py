# src/migrate_to_sql.py - Convert pickle data to SQL

import pickle
from database import PPGDatabase
from tqdm import tqdm

print("Migrating data to SQL database...")

# Load pickle data
with open('data/ppg_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Found {len(data)} records")

# Create database
db = PPGDatabase('data/ppg_database.db')

# Migrate each record
for record in tqdm(data, desc="Migrating"):
    # Insert patient
    db.insert_patient(
        patient_id=record['subject_id'],
        age=record['age'],
        sex=record['sex']
    )
    
    # Insert recording
    db.insert_recording(
        patient_id=record['subject_id'],
        systolic=record['systolic'],
        diastolic=record['diastolic'],
        heart_rate=record['heart_rate'],
        signal=record['ppg']
    )

# Show stats
stats = db.get_statistics()
print(f"\n✅ Migration complete!")
print(f"   Patients: {stats['total_patients']}")
print(f"   Recordings: {stats['total_recordings']}")
print(f"   Average BP: {stats['avg_bp']} mmHg")

db.close()