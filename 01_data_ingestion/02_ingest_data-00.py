# 01_data_ingestion/02_ingest_data.py

import os
import hashlib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Initialize context and credentials
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Configuration
RAW_DATA_PATH = "../data/duolingo_slam_release.csv" # Adjust to your file path
CHUNK_SIZE = 100000 # Process 100k rows at a time to protect server memory

def generate_exercise_hash(row):
    """
    Creates a unique fingerprint for an exercise session to link tokens correctly.
    """
    unique_string = f"{row['user']}_{row['client']}_{row['session']}_{row['time']}_{row['format']}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def process_chunk(chunk, conn, cursor):
    """
    Normalizes a chunk of flat data and bulk-inserts it into PostgreSQL.
    """
    # 1. Generate the unique hash for the exercise
    chunk['exercise_hash'] = chunk.apply(generate_exercise_hash, axis=1)

    # 2. Isolate unique exercises from this chunk
    exercises = chunk[['exercise_hash', 'user', 'countries', 'days', 'client', 'session', 'format', 'time']].drop_duplicates()
    
    # We must add a temporary column to raw_exercises in the DB to store the hash, 
    # or handle it purely in memory. For maximum stability, we will pass the hash, 
    # insert the exercise, and use PostgreSQL's RETURNING clause to get the new DB ID.
    
    insert_exercise_query = """
        INSERT INTO raw_exercises (user_id, countries, days_in_course, client, session_type, exercise_format, time_taken)
        VALUES %s
        RETURNING exercise_id;
    """
    
    exercise_values = [
        (row['user'], row['countries'], row['days'], row['client'], row['session'], row['format'], row['time'])
        for _, row in exercises.iterrows()
    ]
    
    # Execute bulk insert for exercises and fetch the generated IDs
    execute_values(cursor, insert_exercise_query, exercise_values, fetch=True)
    generated_ids = cursor.fetchall()
    
    # Map the new database IDs back to our unique exercises
    exercises['db_exercise_id'] = [item[0] for item in generated_ids]
    
    # 3. Merge the database IDs back to the granular token chunk
    chunk = chunk.merge(exercises[['exercise_hash', 'db_exercise_id']], on='exercise_hash')

    # 4. Prepare token data for bulk insert
    insert_token_query = """
        INSERT INTO raw_tokens (exercise_id, token_order, surface_form, part_of_speech, morphology, dependency_label, is_error)
        VALUES %s;
    """
    
    # Assuming the raw CSV has a 'token' column for the word, and 'label' for the error (1 or 0)
    # We use enumerate to generate the token_order on the fly per exercise
    chunk['token_order'] = chunk.groupby('db_exercise_id').cumcount() + 1
    
    token_values = [
        (row['db_exercise_id'], row['token_order'], row['token'], row['part_of_speech'], row['morphology'], row['dependency_label'], bool(row['label']))
        for _, row in chunk.iterrows()
    ]
    
    # Execute bulk insert for tokens
    execute_values(cursor, insert_token_query, token_values)
    conn.commit()

def run_ingestion_pipeline():
    print("Initiating SLA Knowledge Tracing Data Ingestion...")
    
    try:
        # Establish database connection
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Stream the CSV in chunks
        chunk_counter = 1
        for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
            print(f"Processing chunk {chunk_counter} ({CHUNK_SIZE} rows)...")
            process_chunk(chunk, conn, cursor)
            chunk_counter += 1
            
        print("\nData ingestion completed successfully. Operations seamless.")
        
    except Exception as e:
        print(f"\nPipeline Failure: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    run_ingestion_pipeline()