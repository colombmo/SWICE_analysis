-- Tables used to store the OSM data locally

-- Define the table schema for storing vehicle names
CREATE TABLE OSMVehicles (
    id INTEGER PRIMARY KEY,
    name TEXT
);
-- Prepopulate this table with the names of the vehicles
INSERT INTO OSMVehicles (name) VALUES ('TRAIN'); -- LineString
INSERT INTO OSMVehicles (name) VALUES ('TRAM'); -- LineString
INSERT INTO OSMVehicles (name) VALUES ('BUS'); -- LineString
INSERT INTO OSMVehicles (name) VALUES ('LAKE'); -- Polygon

-- Define the table schema for storing spatial data
CREATE TABLE SpatialData (
    id INTEGER PRIMARY KEY,
    vehicle INTEGER REFERENCES OSMVehicles(id),
    geometry TEXT, -- Serialized representation of the geometry - using Strings
    minX REAL, -- Minimum X coordinate of the bounding box
    minY REAL, -- Minimum Y coordinate of the bounding box
    maxX REAL, -- Maximum X coordinate of the bounding box
    maxY REAL -- Maximum Y coordinate of the bounding box
);

-- Create an index on the bounding box coordinates
CREATE INDEX spatial_data_bbox_index ON SpatialData (minX, minY, maxX, maxY);