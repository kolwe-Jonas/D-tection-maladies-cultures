import sqlite3
import os

db_path = "diseases.db"
print(f"[DB] Vérification de {db_path}...")
print(f"[DB] Existe: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    print(f"[DB] Tables: {tables}")
    
    if tables:
        cur.execute("SELECT COUNT(*) FROM diseases")
        count = cur.fetchone()[0]
        print(f"[DB] Nombre de maladies: {count}")
        
        if count > 0:
            cur.execute("SELECT id, disease_name FROM diseases LIMIT 3")
            for row in cur.fetchall():
                print(f"  - {row}")
    conn.close()
else:
    print("[DB] Fichier introuvable - création nécessaire")
