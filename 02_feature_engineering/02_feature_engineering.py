# 02_feature_engineering/02_feature_engineering.py

import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from dotenv import load_dotenv

# Initialize configurations
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "en_es_slam")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

BATCH_SIZE_EXERCISES = 15000  # Number of exercises (and their tokens) to process in a single chunk

def setup_features_table(cursor, conn):
    """Creates the engineered_features table and its indexes if they do not exist."""
    print("Setting up engineered_features schema...")
    
    # Read schema SQL file
    schema_path = os.path.join(os.path.dirname(__file__), "01_features_schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
        
    # We check if table exists first
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'engineered_features'
        );
    """)
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        print("Executing features schema creation...")
        cursor.execute(schema_sql)
        conn.commit()
        print("engineered_features table and indexes created successfully.")
    else:
        print("engineered_features table already exists. Schema setup skipped.")

def run_feature_engineering_pipeline():
    print("Initiating SLA Knowledge Tracing Feature Engineering Pipeline...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # 1. Initialize schema
        setup_features_table(cursor, conn)
        
        # 2. Get exercise ID range
        cursor.execute("SELECT MIN(exercise_id), MAX(exercise_id) FROM raw_exercises;")
        min_id, max_id = cursor.fetchone()
        if min_id is None or max_id is None:
            print("No raw exercises found in the database. Ingestion must be run first.")
            return
            
        # 3. Calculate scaling parameters on first 80% (training split) to prevent data leakage
        split_boundary = int(min_id + (max_id - min_id) * 0.8)
        print(f"Fitting RobustScaler on training set (exercise_id <= {split_boundary})...")
        
        cursor.execute(f"""
            SELECT 
                percentile_cont(0.5) WITHIN GROUP (ORDER BY days_in_course) as median,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY days_in_course) as q25,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY days_in_course) as q75
            FROM raw_exercises
            WHERE exercise_id <= {split_boundary};
        """)
        median, q25, q75 = cursor.fetchone()
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1.0  # Prevent division by zero
            
        print(f"Scaler parameters: Median={median}, IQR={iqr}")
        
        # 4. Check for resume boundary
        cursor.execute("""
            SELECT MAX(t.exercise_id) 
            FROM engineered_features ef 
            JOIN raw_tokens t ON ef.token_id = t.token_id;
        """)
        max_processed_ex_id = cursor.fetchone()[0]
        if max_processed_ex_id is not None:
            start_id = max_processed_ex_id + 1
            print(f"Resuming feature engineering pipeline from exercise ID {start_id}...")
        else:
            start_id = min_id
            print("Starting feature engineering pipeline from scratch...")
            
        # 5. Process data in chunks of exercise_id ranges
        current_id = start_id
        while current_id <= max_id:
            end_chunk_id = min(current_id + BATCH_SIZE_EXERCISES - 1, max_id)
            print(f"Processing chunk: exercises {current_id} to {end_chunk_id}...")
            
            # Fetch raw data for the current exercise range
            query = """
                SELECT t.token_id, t.is_error, t.part_of_speech, t.morphology, t.dependency_label, t.token_order,
                       e.days_in_course, e.client, e.session_type, e.exercise_format
                FROM raw_exercises e
                JOIN raw_tokens t ON e.exercise_id = t.exercise_id
                WHERE e.exercise_id BETWEEN %s AND %s;
            """
            cursor.execute(query, (current_id, end_chunk_id))
            rows = cursor.fetchall()
            
            if not rows:
                current_id = end_chunk_id + 1
                continue
                
            # Read into Pandas DataFrame
            df = pd.DataFrame(rows, columns=[
                'token_id', 'is_error', 'part_of_speech', 'morphology', 'dependency_label', 'token_order',
                'days_in_course', 'client', 'session_type', 'exercise_format'
            ])
            
            # Feature Creation
            # A. 3rd person singular present verbs
            is_verb = df['part_of_speech'].isin(['VERB', 'AUX'])
            is_3sg = df['morphology'].str.contains('Person=3', na=False) & df['morphology'].str.contains('Number=Sing', na=False)
            df['is_verb_3sg'] = is_verb & is_3sg
            
            # B. Subject pronouns (pro-drop structural clash)
            df['is_pron_subject'] = df['part_of_speech'].eq('PRON') & df['dependency_label'].eq('nsubj')
            
            # C. Prepositions (adposition mapping lexical clash)
            df['is_preposition'] = df['part_of_speech'].eq('ADP')
            
            # D. Format: Listen (auditory spelling recall)
            df['format_listen'] = df['exercise_format'].eq('listen')
            
            # E. Robust scaled days_in_course progress
            df['days_in_course_scaled'] = (df['days_in_course'].astype(float) - median) / iqr
            
            # F. One-hot contexts
            df['client_web'] = df['client'].eq('web')
            df['client_android'] = df['client'].eq('android')
            df['client_ios'] = df['client'].eq('ios')
            
            df['session_lesson'] = df['session_type'].eq('lesson')
            df['session_practice'] = df['session_type'].eq('practice')
            df['session_test'] = df['session_type'].eq('test')
            
            df['format_reverse_translate'] = df['exercise_format'].eq('reverse_translate')
            df['format_reverse_tap'] = df['exercise_format'].eq('reverse_tap')
            
            # G. Structure values to insert
            insert_query = """
                INSERT INTO engineered_features (
                    token_id, is_error, is_verb_3sg, is_pron_subject, is_preposition, format_listen, days_in_course,
                    client_web, client_android, client_ios, session_lesson, session_practice, session_test,
                    format_reverse_translate, format_reverse_tap, token_order
                )
                VALUES %s
                ON CONFLICT (token_id) DO NOTHING;
            """
            
            feature_values = [
                (
                    row['token_id'],
                    bool(row['is_error']),
                    bool(row['is_verb_3sg']),
                    bool(row['is_pron_subject']),
                    bool(row['is_preposition']),
                    bool(row['format_listen']),
                    float(row['days_in_course_scaled']),
                    bool(row['client_web']),
                    bool(row['client_android']),
                    bool(row['client_ios']),
                    bool(row['session_lesson']),
                    bool(row['session_practice']),
                    bool(row['session_test']),
                    bool(row['format_reverse_translate']),
                    bool(row['format_reverse_tap']),
                    int(row['token_order'])
                )
                for _, row in df.iterrows()
            ]
            
            execute_values(cursor, insert_query, feature_values)
            conn.commit()
            
            # Increment range
            current_id = end_chunk_id + 1
            
        print("\nFeature engineering pipeline completed successfully. Operations seamless.")
        
    except Exception as e:
        print(f"\nPipeline Failure: {e}")
        if 'conn' in locals(): conn.rollback()
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    run_feature_engineering_pipeline()
