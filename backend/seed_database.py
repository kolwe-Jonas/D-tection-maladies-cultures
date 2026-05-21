"""
Remplit diseases.db avec le catalogue descriptif complet.

Usage : python seed_database.py
"""

from database import connect_db, create_tables, insert_disease, get_db_path
from diseases_catalog import get_descriptive_catalog


def seed_database(db_path: str = None) -> None:
	conn = connect_db(db_path)
	try:
		create_tables(conn)
		print("Mise à jour des données sans suppression des anciennes entrées...")

		diseases = get_descriptive_catalog()
		print(f"Ajout de maladies descriptives manquantes ({len(diseases)} en catalogue)...")

		inserted = 0
		updated = 0
		by_plant: dict = {}
		for disease in diseases:
			cur = conn.execute(
				"SELECT id FROM diseases WHERE disease_name = ? AND plant_type = ?",
				(disease.get("disease_name"), disease.get("plant_type"))
			)
			if cur.fetchone():
				updated += 1
				continue
			insert_disease(conn, disease)
			inserted += 1
			plant = disease.get("plant_type", "?")
			by_plant[plant] = by_plant.get(plant, 0) + 1

		print(f"Nouvelle insertion : {inserted} maladies")
		print(f"Maladies déjà présentes : {updated}")

		print("Base renseignée avec succès.")
		for plant, count in sorted(by_plant.items()):
			print(f"  - {plant}: {count} maladies")
	finally:
		conn.close()


if __name__ == "__main__":
	db_file = get_db_path()
	print(f"Base : {db_file}")
	seed_database(db_file)
