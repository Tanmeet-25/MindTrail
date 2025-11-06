import sqlite3
import os
import pandas as pd

# Build the database path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "mindtrail.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

print(f"Connecting to: {DB_PATH}")

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create table structure
cursor.execute("""
CREATE TABLE IF NOT EXISTS career_data (
    O_score REAL,
    C_score REAL,
    E_score REAL,
    A_score REAL,
    N_score REAL,
    Numerical_Aptitude REAL,
    Spatial_Aptitude REAL,
    Perceptual_Aptitude REAL,
    Abstract_Aptitude REAL,
    Mechanical_Aptitude REAL,
    Verbal_Aptitude REAL,
    Career TEXT
);
""")

# ✅ FIXED: the line that caused your error
CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned", "cleaned_data.csv")

# Load the CSV using pandas
print("Loading CSV data...")
df = pd.read_csv(CSV_PATH)

# Insert the CSV data into SQLite
print("Inserting data into database...")
df.to_sql("career_data", conn, if_exists="replace", index=False)

print("✅ Data inserted successfully!")

# Example: retrieve and show a few rows
print("\nPreview of data from SQLite:")
result = pd.read_sql_query("SELECT * FROM career_data LIMIT 10;", conn)
print(result)

# Example query: total rows
count = pd.read_sql_query("SELECT COUNT(*) AS total FROM career_data;", conn)
print("\nTotal rows in career_data:", count.iloc[0]['total'])

conn.close()
