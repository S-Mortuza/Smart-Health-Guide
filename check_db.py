import sqlite3
import os

# ডাটাবেস ফাইল আছে কিনা চেক করা
db_file = 'users.db'

if not os.path.exists(db_file):
    print(f"❌ Error: '{db_file}' file not found in this folder!")
    print("Please run 'python app.py' first to create the database.")
else:
    print(f"✅ Found database file: {db_file}")
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # ১. টেবিলের নাম খোঁজা
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if tables:
            print(f"📋 Tables found: {tables}")
            
            # ২. প্রথম টেবিল থেকে ডাটা দেখানো
            for table in tables:
                table_name = table[0]
                print(f"\n--- Data in '{table_name}' table ---")
                
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                
                if rows:
                    for row in rows:
                        print(row)
                else:
                    print("Table is empty (No users registered).")
        else:
            print("⚠️ Database exists but has NO tables inside.")

        conn.close()

    except Exception as e:
        print(f"Error reading database: {e}")