"""
Simple SQLite database module for the plant disease detection app.

This module creates a `diseases` table and provides reusable functions
for common database operations (connect, create, insert, query, update,
delete). It also includes a `seed_data` function that inserts at least
10 agricultural diseases with French descriptions.

Beginner-friendly comments and straightforward functions are used so
you can read and adapt the code easily.
"""

import os
import sqlite3
from typing import Dict, List, Optional


def get_db_path(filename: str = "diseases.db") -> str:
	"""Return an absolute path for the database file (placed next to this file)."""
	base = os.path.dirname(__file__)
	return os.path.join(base, filename)


def connect_db(db_path: Optional[str] = None) -> sqlite3.Connection:
	"""Create and return a SQLite connection using `Row` factory.

	Args:
		db_path: optional path to the SQLite file. If None, uses default
				 file `diseases.db` in the `backend` folder.
	"""
	if db_path is None:
		db_path = get_db_path()
	conn = sqlite3.connect(db_path)
	# Use Row so we can access columns by name (like a dict)
	conn.row_factory = sqlite3.Row
	return conn


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
	"""Return True if the given table already contains the specified column."""
	cur = conn.execute(f"PRAGMA table_info({table})")
	columns = [row[1] for row in cur.fetchall()]
	return column in columns


# Colonnes descriptives pour la comparaison visuelle (disease_detector)
DESCRIPTIVE_COLUMNS = [
	"type_taches",
	"couleur_taches",
	"taille_taches",
	"disposition_taches",
	"texture_feuille",
	"couleur_generale",
	"zones_atteintes",
	"progression_maladie",
]


def create_tables(conn: sqlite3.Connection) -> None:
	"""Create the `diseases` table and migrate missing descriptive columns."""
	sql = """
	CREATE TABLE IF NOT EXISTS diseases (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		disease_name TEXT NOT NULL,
		scientific_name TEXT,
		symptoms TEXT,
		causes TEXT,
		treatment TEXT,
		prevention TEXT,
		severity TEXT,
		leaf_color TEXT,
		leaf_texture TEXT,
		plant_type TEXT,
		type_taches TEXT,
		couleur_taches TEXT,
		couleur_generale TEXT,
		taille_taches TEXT,
		disposition_taches TEXT,
		texture_feuille TEXT,
		zones_atteintes TEXT,
		progression_maladie TEXT
	);
	"""
	conn.execute(sql)
	conn.commit()

	for column in ["plant_type"] + DESCRIPTIVE_COLUMNS:
		if not _table_has_column(conn, "diseases", column):
			conn.execute(f"ALTER TABLE diseases ADD COLUMN {column} TEXT")
	conn.commit()


def insert_disease(conn: sqlite3.Connection, disease: Dict) -> int:
	"""Insert a disease record and return its new id.

	`disease` should be a dict containing keys matching the table columns
	(except `id`).
	"""
	keys = [
		"disease_name",
		"scientific_name",
		"symptoms",
		"causes",
		"treatment",
		"prevention",
		"severity",
		"leaf_color",
		"leaf_texture",
		"plant_type",
	] + DESCRIPTIVE_COLUMNS
	columns = ", ".join(keys)
	placeholders = ", ".join(["?" for _ in keys])
	values = [disease.get(k) for k in keys]
	cur = conn.execute(f"INSERT INTO diseases ({columns}) VALUES ({placeholders})", values)
	conn.commit()
	return cur.lastrowid


def get_all_diseases(conn: sqlite3.Connection) -> List[Dict]:
	"""Return a list of all diseases as dictionaries."""
	cur = conn.execute("SELECT * FROM diseases ORDER BY id")
	rows = cur.fetchall()
	return [dict(row) for row in rows]


def get_disease_by_id(conn: sqlite3.Connection, disease_id: int) -> Optional[Dict]:
	"""Return a single disease by `id`, or None if not found."""
	cur = conn.execute("SELECT * FROM diseases WHERE id = ?", (disease_id,))
	row = cur.fetchone()
	return dict(row) if row else None


def update_disease(conn: sqlite3.Connection, disease_id: int, updates: Dict) -> bool:
	"""Update fields for a disease. `updates` is a dict of column->value.

	Returns True if a row was changed, False otherwise.
	"""
	if not updates:
		return False
	allowed = {
		"disease_name", "scientific_name", "symptoms", "causes",
		"treatment", "prevention", "severity", "leaf_color",
		"leaf_texture", "plant_type",
	} | set(DESCRIPTIVE_COLUMNS)
	keys = [k for k in updates.keys() if k in allowed]
	if not keys:
		return False
	assignments = ", ".join([f"{k} = ?" for k in keys])
	values = [updates[k] for k in keys] + [disease_id]
	cur = conn.execute(f"UPDATE diseases SET {assignments} WHERE id = ?", values)
	conn.commit()
	return cur.rowcount > 0


def delete_disease(conn: sqlite3.Connection, disease_id: int) -> bool:
	"""Delete a disease by id. Returns True if a row was deleted."""
	cur = conn.execute("DELETE FROM diseases WHERE id = ?", (disease_id,))
	conn.commit()
	return cur.rowcount > 0


def seed_data(conn: sqlite3.Connection) -> None:
	"""Insert at least 10 agricultural diseases with French descriptions.

	This function is idempotent in the sense that it first checks whether
	there are already rows and skips inserting if the table is not empty.
	"""
	cur = conn.execute("SELECT COUNT(1) as c FROM diseases")
	count = cur.fetchone()[0]
	if count > 0:
		# Already seeded (or user data exists). Skip seeding.
		return

	diseases = [
		{
			"disease_name": "Mildiou",
			"scientific_name": "Phytophthora infestans",
			"symptoms": "Taches brunes huileuses sur les feuilles, duvet blanc sous les feuilles, pourriture des tubercules.",
			"causes": "Oomycète (champignon-like) favorisé par l'humidité et la pluie.",
			"treatment": "Fongicides systémiques, élimination des parties affectées.",
			"prevention": "Rotation des cultures, semences saines, bonne aération.",
			"severity": "élevé",
			"leaf_color": "vert jaunissant puis brun",
			"leaf_texture": "taches huileuses, duvet sporadique",
		},
		{
			"disease_name": "Oïdium",
			"scientific_name": "Erysiphe spp.",
			"symptoms": "Poudre blanche sur la surface des feuilles, déformation et dessèchement.",
			"causes": "Champignons ascomycètes favorisés par temps sec et ombragé.",
			"treatment": "Fongicides à base de soufre ou de myclobutanil, suppression des feuilles atteintes.",
			"prevention": "Espace entre plants, éviter l'excès d'azote.",
			"severity": "modéré",
			"leaf_color": "poudre blanche sur vert",
			"leaf_texture": "poudreux",
		},
		{
			"disease_name": "Rouille",
			"scientific_name": "Puccinia spp.",
			"symptoms": "Petites pustules orangées/brunes sur les faces inférieures des feuilles.",
			"causes": "Champignons parasites nécessitant humidité et chaleur modérée.",
			"treatment": "Fongicides, enlever feuilles fortement infectées.",
			"prevention": "Variétés résistantes, rotation des cultures.",
			"severity": "modéré",
			"leaf_color": "taches orange/brun",
			"leaf_texture": "pustules poudreuses",
		},
		{
			"disease_name": "Alternariose (Tache foliaire)",
			"scientific_name": "Alternaria solani",
			"symptoms": "Taches brunes concentriques sur feuilles et fruits, dessèchement.",
			"causes": "Champignon saprophyte devenu pathogène sur plantes affaiblies.",
			"treatment": "Fongicides, nettoyage des résidus de culture.",
			"prevention": "Rotation, élimination des débris végétaux.",
			"severity": "modéré",
			"leaf_color": "brun foncé",
			"leaf_texture": "taches nécrotiques",
		},
		{
			"disease_name": "Fusariose (flétrissement)",
			"scientific_name": "Fusarium oxysporum",
			"symptoms": "Flétrissement unilatéral, jaunissement des feuilles, nécrose vasculaire.",
			"causes": "Champignon du sol qui envahit les vaisseaux de la plante.",
			"treatment": "Peu de traitements efficaces; élimination des plantes affectées.",
			"prevention": "Utiliser des porte-greffes résistants, rotation longue.",
			"severity": "élevé",
			"leaf_color": "jaune puis brun",
			"leaf_texture": "flétri",
		},
		{
			"disease_name": "Botrytis (pourriture grise)",
			"scientific_name": "Botrytis cinerea",
			"symptoms": "Pourriture molle recouverte d'un duvet gris, surtout sur fleurs et fruits.",
			"causes": "Champignon opportuniste en atmosphère humide.",
			"treatment": "Fongicides et suppression des tissus atteints.",
			"prevention": "Sécher les plants, bonne circulation d'air.",
			"severity": "modéré",
			"leaf_color": "brun/gris",
			"leaf_texture": "mou et duveteux",
		},
		{
			"disease_name": "Tache bactérienne",
			"scientific_name": "Xanthomonas campestris",
			"symptoms": "Taches humides sur feuilles et fruits, lézardes sur tiges.",
			"causes": "Bactérie dispersée par eau et outils contaminés.",
			"treatment": "Élimination des parties infectées; quelques traitements phytosanitaires en prévention.",
			"prevention": "Désinfection des outils, éviter arrosage foliaire.",
			"severity": "modéré",
			"leaf_color": "taches brunes/humides",
			"leaf_texture": "luisant/visqueux",
		},
		{
			"disease_name": "Mosaïque (virus)",
			"scientific_name": "Potyvirus / Tobamovirus (ex.)",
			"symptoms": "Motifs en mosaïque jaune/vert, feuilles déformées, croissance réduite.",
			"causes": "Virus transmis par insectes, semences ou contact.",
			"treatment": "Aucun traitement curatif; retirer les plants infectés.",
			"prevention": "Utiliser semences saines, contrôle des vecteurs.",
			"severity": "élevé",
			"leaf_color": "mosaïque jaune/vert",
			"leaf_texture": "rugueux/déformé",
		},
		{
			"disease_name": "Nématodes à galles",
			"scientific_name": "Meloidogyne spp.",
			"symptoms": "Retard de croissance, galles visibles sur racines, jaunissement général.",
			"causes": "Nématodes du sol qui parasitent les racines.",
			"treatment": "Amendements organiques, nématicides (usage réglementé).",
			"prevention": "Rotation longue, solarisation du sol.",
			"severity": "modéré",
			"leaf_color": "jaunissement général",
			"leaf_texture": "feuilles fines et rabougries",
		},
		{
			"disease_name": "Cercosporiose",
			"scientific_name": "Cercospora spp.",
			"symptoms": "Petites taches brunes avec bord clair, parfois chutes de feuilles précoces.",
			"causes": "Champignon favorisé par l'humidité et des feuilles humides prolongées.",
			"treatment": "Fongicides, enlever les feuilles affectées.",
			"prevention": "Espacer les plants, éviter l'irrigation par aspersion.",
			"severity": "faible",
			"leaf_color": "taches brunes avec halo clair",
			"leaf_texture": "légèrement rugueux",
		},
	]

	for d in diseases:
		name = (d.get("disease_name") or "").lower()
		if "maïs" in name or "maize" in name:
			d["plant_type"] = "maïs"
		elif "pomme de terre" in name or "tubercule" in name or "rhizoctone" in name:
			d["plant_type"] = "pomme de terre"
		elif "manioc" in name or "cassava" in name:
			d["plant_type"] = "manioc"
		elif "riz" in name or "rice" in name or "pyriculariose" in name or "bakanae" in name:
			d["plant_type"] = "riz"
		elif "haricot" in name or "bean" in name:
			d["plant_type"] = "haricot"
		else:
			d["plant_type"] = "tomate"
		insert_disease(conn, d)

if __name__ == "__main__":
	# Quick demonstration: create DB, tables and seed data, then print a summary.
	db_file = get_db_path("diseases.db")
	print(f"Using database file: {db_file}")
	conn = connect_db(db_file)
	create_tables(conn)
	seed_data(conn)
	all_d = get_all_diseases(conn)
	print(f"Inserted {len(all_d)} diseases. Sample:")
	for d in all_d[:5]:
		print(f"- {d['id']}: {d['disease_name']} ({d['scientific_name']})")
	conn.close()

