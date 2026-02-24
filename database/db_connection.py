"""
Database Connection Module
Smart City Traffic & Accident Risk Analytics System

Handles:
- Secure DB connection using .env
- SQLAlchemy engine creation
- Data insertion
- Data fetching
"""

import os
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv



# Load Environment Variables

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")



# MySQL Connection (Basic Connector)

def get_connection():
    """
    Returns MySQL connection object
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print("Database connection successful.")
        return connection

    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None



# SQLAlchemy Engine (For Pandas to_sql)

def get_sqlalchemy_engine():
    """
    Returns SQLAlchemy engine for bulk insert operations
    """
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    )
    return engine



# Insert DataFrame into Database

def insert_dataframe(df, table_name="accidents"):
    """
    Inserts DataFrame into MySQL table
    """
    try:
        engine = get_sqlalchemy_engine()

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",
            index=False,
            chunksize=5000
        )

        print("Data inserted successfully into database.")

    except Exception as e:
        print(f"Error inserting data: {e}")



# Fetch Sample Records

def fetch_sample_records(limit=5):
    """
    Fetch sample records from accidents table
    """
    conn = get_connection()

    if conn:
        cursor = conn.cursor()
        query = f"SELECT * FROM accidents LIMIT {limit}"
        cursor.execute(query)

        rows = cursor.fetchall()

        for row in rows:
            print(row)

        cursor.close()
        conn.close()


# Count Total Records

def count_records():
    """
    Returns total row count in accidents table
    """
    conn = get_connection()

    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM accidents;")
        count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        print(f"Total Records in accidents table: {count}")
        return count



# Test Script

if __name__ == "__main__":
    count_records()