"""
Database migration script to add file_content column to brand_documents table.

Run this script ONCE to update your production database schema.

Usage:
    python add_file_content_column.py
"""

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

load_dotenv()

def migrate_database():
    """Add file_content column to brand_documents table"""
    
    db_url = os.getenv("POSTGRES_URI")
    if not db_url:
        print("ERROR: POSTGRES_URI not found in environment variables")
        return False
    
    try:
        # Connect to database
        print("Connecting to database...")
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='brand_documents' 
            AND column_name='file_content'
        """)
        
        if cursor.fetchone():
            print("SUCCESS: Column 'file_content' already exists. No migration needed.")
            return True
        
        # Add the column
        print("Adding file_content column...")
        cursor.execute("""
            ALTER TABLE brand_documents 
            ADD COLUMN file_content TEXT
        """)
        
        print("SUCCESS: Migration successful!")
        print("Note: Existing documents will have NULL file_content.")
        print("Re-upload documents to populate this column.")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("DATABASE MIGRATION: Add file_content column")
    print("="*60)
    
    success = migrate_database()
    
    if success:
        print("\nSUCCESS: Migration completed successfully!")
        print("\nNext steps:")
        print("1. Restart your application")
        print("2. Upload brand documents via /api/upload")
        print("3. Documents will now persist across deployments")
    else:
        print("\nERROR: Migration failed. Please check the error above.")
    
    print("="*60)
