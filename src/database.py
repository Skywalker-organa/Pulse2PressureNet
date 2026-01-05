# src/database.py - SQL Database Management

import sqlite3
import numpy as np
import io
import pandas as pd

class PPGDatabase:
    """Manage PPG data with SQLite"""
    
    def __init__(self, db_path='data/ppg_database.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema"""
        
        # Patient metadata table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY,
                age INTEGER,
                sex TEXT,
                height REAL,
                weight REAL,
                bmi REAL,
                hypertension TEXT
            )
        ''')
        
        # PPG recordings table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                systolic REAL,
                diastolic REAL,
                heart_rate REAL,
                signal BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        ''')
        
        # Predictions table (for trained model outputs)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER,
                predicted_systolic REAL,
                predicted_diastolic REAL,
                true_systolic REAL,
                true_diastolic REAL,
                error_systolic REAL,
                error_diastolic REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
            )
        ''')
        
        self.conn.commit()
        print("Database tables created")
    
    def _adapt_array(self, arr):
        """Convert numpy array to bytes for storage"""
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    def _convert_array(self, blob):
        """Convert bytes back to numpy array"""
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)
    
    def insert_patient(self, patient_id, age, sex, height=None, weight=None, bmi=None, hypertension=None):
        """Add patient to database"""
        self.cursor.execute('''
            INSERT OR IGNORE INTO patients 
            (patient_id, age, sex, height, weight, bmi, hypertension)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (patient_id, age, sex, height, weight, bmi, hypertension))
        self.conn.commit()
    
    def insert_recording(self, patient_id, systolic, diastolic, heart_rate, signal):
        """Add PPG recording"""
        signal_blob = self._adapt_array(signal)
        
        self.cursor.execute('''
            INSERT INTO recordings 
            (patient_id, systolic, diastolic, heart_rate, signal)
            VALUES (?, ?, ?, ?, ?)
        ''', (patient_id, systolic, diastolic, heart_rate, signal_blob))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_prediction(self, recording_id, pred_sys, pred_dia, true_sys, true_dia):
        """Store model prediction"""
        error_sys = abs(pred_sys - true_sys)
        error_dia = abs(pred_dia - true_dia)
        
        self.cursor.execute('''
            INSERT INTO predictions 
            (recording_id, predicted_systolic, predicted_diastolic, 
             true_systolic, true_diastolic, error_systolic, error_diastolic)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (recording_id, pred_sys, pred_dia, true_sys, true_dia, error_sys, error_dia))
        
        self.conn.commit()
    
    def get_all_recordings(self):
        """Get all recordings"""
        self.cursor.execute('''
            SELECT r.recording_id, r.patient_id, r.systolic, r.diastolic, 
                   r.heart_rate, r.signal, p.age, p.sex
            FROM recordings r
            LEFT JOIN patients p ON r.patient_id = p.patient_id
        ''')
        return self.cursor.fetchall()
    
    def get_high_bp_patients(self, threshold=140):
        """Query patients with high BP"""
        self.cursor.execute('''
            SELECT p.patient_id, p.age, p.sex, AVG(r.systolic) as avg_systolic
            FROM patients p
            JOIN recordings r ON p.patient_id = r.patient_id
            GROUP BY p.patient_id
            HAVING avg_systolic > ?
            ORDER BY avg_systolic DESC
        ''', (threshold,))
        return self.cursor.fetchall()
    
    def get_statistics(self):
        """Get database statistics"""
        stats = {}
        
        # Total records
        self.cursor.execute('SELECT COUNT(*) FROM patients')
        stats['total_patients'] = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM recordings')
        stats['total_recordings'] = self.cursor.fetchone()[0]
        
        # Average BP
        self.cursor.execute('SELECT AVG(systolic), AVG(diastolic) FROM recordings')
        avg_sys, avg_dia = self.cursor.fetchone()
        stats['avg_bp'] = f"{avg_sys:.1f}/{avg_dia:.1f}"
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Test it
if __name__ == "__main__":
    print("Testing database...")
    
    # Create database
    db = PPGDatabase('data/test.db')
    
    # Add test patient
    db.insert_patient(patient_id=1, age=45, sex='Male')
    
    # Add test recording
    test_signal = np.random.randn(1250)
    recording_id = db.insert_recording(
        patient_id=1,
        systolic=140,
        diastolic=90,
        heart_rate=75,
        signal=test_signal
    )
    print(f"Inserted recording ID: {recording_id}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase stats: {stats}")
    
    db.close()
    print("Database test complete")