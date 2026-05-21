import sqlite3

conn = sqlite3.connect("diseases.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS diseases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_name TEXT,
    scientific_name TEXT,
    symptoms TEXT,
    causes TEXT,
    treatment TEXT,
    prevention TEXT,
    leaf_color TEXT,
    severity TEXT,
    leaf_texture TEXT,
    plant_type TEXT
)
""")

diseases = [
    (
        "Blight (Brûlure des feuilles)",
        "Phytophthora infestans",
        "taches brunes, feuilles sèches",
        "champignon",
        "retirer les feuilles infectées + fongicide",
        "rotation des cultures",
        "brun",
        "élevé",
        "taches nécrotiques",
        "tomate"
    ),
    (
        "Chlorose",
        "Carence en fer",
        "feuilles jaunes",
        "manque de nutriments",
        "engrais riche en fer",
        "fertilisation régulière",
        "jaune",
        "faible",
        "jaune pâle",
        "tomate"
    ),
    (
        "Mosaïque virale",
        "Virus de la mosaïque",
        "taches jaunes et vertes",
        "virus transmis par insectes",
        "éliminer les plantes infectées",
        "lutte contre les insectes",
        "jaune vert",
        "élevé",
        "motifs marbrés",
        "tomate"
    ),
    (
        "Pourriture noire",
        "Xanthomonas",
        "taches noires",
        "bactérie",
        "antibiotique agricole",
        "hygiène des champs",
        "noir brun",
        "sévère",
        "taches humides",
        "tomate"
    )
]

cur.executemany("""
INSERT INTO diseases (
    disease_name,
    scientific_name,
    symptoms,
    causes,
    treatment,
    prevention,
    leaf_color,
    severity,
    leaf_texture,
    plant_type
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", diseases)

conn.commit()
conn.close()

print("Base de données créée avec succès ✅")