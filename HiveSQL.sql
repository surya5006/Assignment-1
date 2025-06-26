--  Start Hive Metastore (use in Terminal or CLI)
-- Run this command in the background before Hive CLI starts
! hive --service metastore &

-- Then enter Hive CLI
! hive

-- Create External Table for Reviews
-- This table parses user-generated reviews from the JSONL file
CREATE EXTERNAL TABLE IF NOT EXISTS amazon_reviews (
  sort_timestamp BIGINT,              -- Unix timestamp of the review
  rating FLOAT,                       -- Rating between 1.0 to 5.0
  helpful_votes INT,                  -- Number of helpful votes received
  title STRING,                       -- Title of the review
  text STRING,                        -- Full review text
  asin STRING,                        -- Amazon Standard Identification Number
  parent_asin STRING,                 -- Parent ASIN (grouping variants)
  user_id STRING,                     -- Unique user identifier
  verified_purchase BOOLEAN           -- Whether the purchase was verified
)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION '/amazon_reviews/';

-- Create External Table for Product Metadata
-- This table handles item-level features like descriptions and average rating
CREATE EXTERNAL TABLE IF NOT EXISTS amazon_meta (
  main_category STRING,               -- Main product category (e.g., Subscription Boxes)
  title STRING,                       -- Product title
  average_rating FLOAT,               -- Average customer rating
  rating_number INT,                  -- Total number of ratings
  features ARRAY<STRING>,            -- Feature list in bullet-point format
  description ARRAY<STRING>,         -- Detailed multi-line description
  price FLOAT,                        -- Product price in USD
  parent_asin STRING,                -- Parent ASIN to join with reviews
  store STRING                        -- Store or brand name
)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION '/amazon_reviews/';
