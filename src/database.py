"""Database module for storing EEG analysis results using SQLite"""

import sqlite3
import json
from datetime import datetime

class EEGDatabase:
    def __init__(self, db_path='eeg_data.db'):
        """Initializing database connection"""
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.dp_path)
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """Create database tables if not already there"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recordings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                run_id INTEGER NOT NULL,
                duration REAL,
                n_channels INTEGER,
                sampling_rate REAL,
                processed_at TIMESTAMP,
                UNIQUE(subject_id, run_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frequency_bands(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER,
                band_name TEXT NOT NULL,
                power REAL NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings(id)
            )
        ''')

        conn.commit()
        self.close()