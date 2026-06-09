# 01_data_ingestion/02_ingest_data.py

import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "en_es_slam")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Point this to your actual .train file
RAW_DATA_PATH = "./en_es.slam.20190204.train" 
BATCH_SIZE = 5000  # Number of exercises to hold in memory before writing to the database

def flush_batch_to_db(cursor, conn, exercises, tokens):
    """Bulk inserts a batch of exercises and their associated tokens."""
    if not exercises: return

    # 1. Insert exercises and retrieve the new database IDs
    insert_exercise_query = """
        INSERT INTO raw_exercises (user_id, countries, days_in_course, client, session_type, exercise_format, time_taken)
        VALUES %s
        RETURNING exercise_id;
    """
    
    def clean_val(val, target_type):
        if val is None or val == 'null' or val == '':
            return None
        try:
            return target_type(val)
        except (ValueError, TypeError):
            return None

    exercise_values = [
        (
            ex['user'],
            clean_val(ex.get('countries'), str),
            clean_val(ex.get('days'), float),
            clean_val(ex.get('client'), str),
            clean_val(ex.get('session'), str),
            clean_val(ex.get('format'), str),
            clean_val(ex.get('time'), int)
        )
        for ex in exercises
    ]
    
    generated_ids = execute_values(cursor, insert_exercise_query, exercise_values, fetch=True)
    
    # 2. Map the new database IDs back to the tokens based on their batch index
    token_values = []
    for batch_idx, db_id in enumerate(generated_ids):
        real_exercise_id = db_id[0]
        # Get all tokens that belong to this exercise index
        exercise_tokens = [t for t in tokens if t['batch_idx'] == batch_idx]
        
        for t in exercise_tokens:
            token_values.append((
                real_exercise_id, 
                t['token_order'], 
                t['surface_form'], 
                t['part_of_speech'], 
                t['morphology'], 
                t['dependency_label'], 
                t['is_error']
            ))

    # 3. Insert the tokens
    insert_token_query = """
        INSERT INTO raw_tokens (exercise_id, token_order, surface_form, part_of_speech, morphology, dependency_label, is_error)
        VALUES %s;
    """
    execute_values(cursor, insert_token_query, token_values)
    conn.commit()

def run_ingestion_pipeline():
    print("Initiating SLA Knowledge Tracing Data Ingestion (Text Stream)...")
    
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cursor = conn.cursor()
        
        # Determine how many exercises are already in the database to resume safely
        cursor.execute("SELECT COUNT(*) FROM raw_exercises;")
        skip_count = cursor.fetchone()[0]
        if skip_count > 0:
            print(f"Database already contains {skip_count} exercises. Resuming ingestion from exercise #{skip_count + 1}...")
        
        exercises_batch = []
        tokens_batch = []
        
        current_exercise = {}
        token_counter = 1
        parsed_exercises_count = 0
        
        with open(RAW_DATA_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip blank lines
                
                if line.startswith('# prompt:'):
                    continue  # Ignore prompt lines completely
                    
                elif line.startswith('# user:'):
                    # We hit a new exercise block. If we have a previous one, save it first.
                    if 'user' in current_exercise:
                        parsed_exercises_count += 1
                        if parsed_exercises_count > skip_count:
                            exercises_batch.append(current_exercise)
                            
                            # Flush to database if batch is full
                            if len(exercises_batch) >= BATCH_SIZE:
                                flush_batch_to_db(cursor, conn, exercises_batch, tokens_batch)
                                print(f"Flushed batch of {BATCH_SIZE} exercises (progress: {parsed_exercises_count}/824012)...")
                                exercises_batch = []
                                tokens_batch = []
                                
                    # Reset context for the new exercise
                    current_exercise = {}
                    token_counter = 1
                    
                    # Parse the user metadata line
                    parts = line.replace('# user:', '').strip().split()
                    current_exercise['user'] = parts[0]
                    for part in parts[1:]:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            current_exercise[key] = value
                            
                elif not line.startswith('#'):
                    # It is a token line. Parse the 7 columns.
                    if parsed_exercises_count >= skip_count:
                        parts = line.split()
                        if len(parts) >= 7:
                            tokens_batch.append({
                                'batch_idx': len(exercises_batch), # Links token to the exercise currently being built
                                'token_order': token_counter,
                                'surface_form': parts[1],
                                'part_of_speech': parts[2],
                                'morphology': parts[3],
                                'dependency_label': parts[4],
                                'is_error': bool(int(parts[6])) # Convert 0/1 to boolean
                            })
                            token_counter += 1

            # Append the final exercise if it wasn't appended
            if 'user' in current_exercise:
                parsed_exercises_count += 1
                if parsed_exercises_count > skip_count:
                    exercises_batch.append(current_exercise)

            # Flush the final remaining batch
            if exercises_batch:
                flush_batch_to_db(cursor, conn, exercises_batch, tokens_batch)
                print(f"Flushed final batch of {len(exercises_batch)} exercises (progress: {parsed_exercises_count}/824012)...")
                
        print("\nData ingestion completed successfully. Operations seamless.")
        
    except Exception as e:
        print(f"\nPipeline Failure: {e}")
        if 'conn' in locals(): conn.rollback()
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    run_ingestion_pipeline()
