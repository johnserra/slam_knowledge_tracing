-- 02_feature_engineering/01_features_schema.sql

-- TABLE: Engineered Features
-- Stores the parsed syntactic features, context markers, and preprocessed progress signals.
CREATE TABLE engineered_features (
    token_id BIGINT PRIMARY KEY REFERENCES raw_tokens(token_id) ON DELETE CASCADE,
    is_error BOOLEAN NOT NULL,
    is_verb_3sg BOOLEAN NOT NULL,
    is_pron_subject BOOLEAN NOT NULL,
    is_preposition BOOLEAN NOT NULL,
    format_listen BOOLEAN NOT NULL,
    days_in_course NUMERIC NOT NULL,
    client_web BOOLEAN NOT NULL,
    client_android BOOLEAN NOT NULL,
    client_ios BOOLEAN NOT NULL,
    session_lesson BOOLEAN NOT NULL,
    session_practice BOOLEAN NOT NULL,
    session_test BOOLEAN NOT NULL,
    format_reverse_translate BOOLEAN NOT NULL,
    format_reverse_tap BOOLEAN NOT NULL,
    token_order INT NOT NULL
);

-- Indexing token_id and is_error for fast model retrieval and querying
CREATE INDEX idx_features_token_id ON engineered_features(token_id);
CREATE INDEX idx_features_is_error ON engineered_features(is_error);
CREATE INDEX idx_features_is_verb_3sg ON engineered_features(is_verb_3sg);
CREATE INDEX idx_features_is_pron_subject ON engineered_features(is_pron_subject);
CREATE INDEX idx_features_is_preposition ON engineered_features(is_preposition);
