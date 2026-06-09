-- 01_data_ingestion/01_schema.sql

-- TABLE 1: Exercise Metadata
-- Stores the context of the learning event. 
CREATE TABLE raw_exercises (
    exercise_id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    countries VARCHAR(100),
    days_in_course NUMERIC,
    client VARCHAR(50),             -- iOS, Android, web
    session_type VARCHAR(50),       -- lesson, practice, test
    exercise_format VARCHAR(50),    -- reverse_translate, reverse_tap, listen
    time_taken INT
);

-- Indexing user_id to speed up chronological learner-history queries
CREATE INDEX idx_user_id ON raw_exercises(user_id);


-- TABLE 2: Token-Level Performance
-- Stores the granular linguistic data and the target variable (the error).
CREATE TABLE raw_tokens (
    token_id BIGSERIAL PRIMARY KEY,
    exercise_id BIGINT REFERENCES raw_exercises(exercise_id) ON DELETE CASCADE,
    token_order INT NOT NULL,       -- The position of the word in the sentence
    surface_form VARCHAR(255),      -- The actual word (e.g., "runs")
    part_of_speech VARCHAR(50),     -- e.g., VERB, NOUN
    morphology VARCHAR(255),        -- e.g., Person=3|Tense=Pres
    dependency_label VARCHAR(50),   -- Syntax mapping
    is_error BOOLEAN NOT NULL       -- 0 = Correct, 1 = Error (Our Target Variable)
);

-- Indexing the foreign key and the target variable for fast aggregations
CREATE INDEX idx_exercise_id ON raw_tokens(exercise_id);
CREATE INDEX idx_surface_form ON raw_tokens(surface_form);
CREATE INDEX idx_is_error ON raw_tokens(is_error);