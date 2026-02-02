# test_db.py
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("POSTGRES_URI")
print(f"Testing connection to: {uri[:50]}...")

try:
    conn = psycopg2.connect(uri)
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")