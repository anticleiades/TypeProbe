import pandas as pd
import sqlite3
import os
import argparse
import sys

def process_files(db_name, parquet_files):
    """Loads Parquet files into an SQLite database with table names matching filenames."""
    try:
        conn = sqlite3.connect(db_name)
        print(f"Connected to database: {db_name}")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    for file_path in parquet_files:
        table_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            print(f"Processing '{file_path}'...")
            df = pd.read_parquet(file_path)

            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"  -> Success: Table '{table_name}' created.")

        except Exception as e:
            print(f"  -> [ERROR] Failed to process {file_path}: {e}")

    conn.close()
    print("\nAll tasks completed.")


def main():
    parser = argparse.ArgumentParser(description="Import Parquet files into SQLite tables.")
    parser.add_argument("-o", "--output", required=True, help="Output SQLite database filename")
    parser.add_argument("files", nargs="+", help="One or more Parquet files to import")

    args = parser.parse_args()

    process_files(args.output, args.files)


if __name__ == "__main__":
    main()
