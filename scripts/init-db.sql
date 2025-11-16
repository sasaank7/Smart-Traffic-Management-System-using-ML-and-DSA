-- Initialize database with PostGIS extension

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create schema version table
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial version
INSERT INTO schema_version (version) VALUES ('1.0.0') ON CONFLICT DO NOTHING;

-- Create spatial indexes function
CREATE OR REPLACE FUNCTION create_spatial_indexes()
RETURNS void AS $$
BEGIN
    -- This function will create spatial indexes after tables are created
    -- Executed by Alembic migrations
    RAISE NOTICE 'Spatial indexes will be created by migrations';
END;
$$ LANGUAGE plpgsql;

COMMENT ON DATABASE smart_traffic_db IS 'Smart Traffic Management System Database';
