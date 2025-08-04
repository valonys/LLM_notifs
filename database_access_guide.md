# Database Access Guide

## Accessing the DigiTwin Database

The DigiTwin RAG system uses PostgreSQL database for storing documents, cached DataFrames, and analysis results.

### 1. Direct SQL Access
You can access the database using the built-in SQL execution tool in the app:

#### Main Tables:
- `uploaded_files` - File metadata and upload history
- `processed_documents` - Document chunks and content
- `cached_dataframes` - Permanently stored Excel DataFrames
- `pivot_cache` - Cached pivot table results
- `query_history` - Chat and analysis history

#### Example Queries:
```sql
-- View all uploaded files
SELECT file_name, file_type, upload_date FROM uploaded_files ORDER BY upload_date DESC;

-- Check cached DataFrames
SELECT file_name, row_count, created_at FROM cached_dataframes;

-- View document chunks
SELECT file_name, COUNT(*) as chunk_count FROM processed_documents GROUP BY file_name;

-- Check pivot cache
SELECT cache_key, created_at FROM pivot_cache ORDER BY created_at DESC LIMIT 10;
```

### 2. Database Connection Details
The database URL is available through the environment variable `DATABASE_URL`.

### 3. Through the Application
- The app automatically connects to the database on startup
- All Excel uploads are permanently cached in `cached_dataframes` table
- Documents are stored in `processed_documents` for vector search
- Pivot analysis results are cached for faster retrieval

### 4. Troubleshooting Database Issues
If you see connection errors:
1. Check if DATABASE_URL environment variable is set
2. Verify PostgreSQL service is running
3. Check database permissions

### 5. Data Persistence
- Excel DataFrames are automatically stored and restored on app restart
- Document chunks are preserved across sessions
- Chat history and analysis results are maintained