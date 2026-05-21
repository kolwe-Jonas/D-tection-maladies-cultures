import argparse
from typing import List, Optional, Tuple

from database import connect_db, create_tables

PLANT_TYPE_KEYWORDS = [
    ("manioc", ["manioc", "cassava", "cassava mosaic", "manioc"]),
    ("maïs", ["maïs", "mais", "corn"]),
    ("tomate", ["tomate", "tomato", "lycopersici", "alternariose", "mildiou", "oidium", "bactéri", "tâche bactérienne", "chlorose"]),
    ("pomme de terre", ["pomme de terre", "patate", "tubercule", "rhizoctone", "phoma", "mildiou", "pourriture sèche"]),
    ("riz", ["riz", "rice", "pyriculariose", "bakanae", "bipolaris", "magno", "brown spot"]),
    ("haricot", ["haricot", "bean", "phaseolus", "anthracnose du haricot", "mosaïque du haricot", "oidium du haricot"]),
]


def normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def detect_plant_type(disease_name: str) -> Optional[str]:
    lower_name = normalize_name(disease_name)
    for plant_type, keywords in PLANT_TYPE_KEYWORDS:
        for keyword in keywords:
            if keyword in lower_name:
                return plant_type
    return None


def update_plant_types(db_path: str) -> Tuple[int, int, List[Tuple[int, str, Optional[str], Optional[str]]]]:
    conn = connect_db(db_path)
    try:
        create_tables(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT id, disease_name, plant_type FROM diseases ORDER BY id")
        rows = cursor.fetchall()

        updated = 0
        skipped = 0
        failed: List[Tuple[int, str, Optional[str], Optional[str]]] = []

        for row in rows:
            disease_id = row[0]
            disease_name = row[1] or ""
            current_type = row[2]
            new_type = detect_plant_type(disease_name)

            if new_type is None:
                skipped += 1
                failed.append((disease_id, disease_name, current_type, new_type))
                continue

            if current_type == new_type:
                continue

            cursor.execute(
                "UPDATE diseases SET plant_type = ? WHERE id = ?",
                (new_type, disease_id),
            )
            updated += 1

        conn.commit()
        return updated, skipped, failed
    finally:
        conn.close()


def print_summary(updated: int, skipped: int, failed: List[Tuple[int, str, Optional[str], Optional[str]]]) -> None:
    print(f"Colonne plant_type vérifiée et mise à jour.")
    print(f"Enregistrements modifiés : {updated}")
    print(f"Enregistrements sans correspondance automatique : {skipped}")
    if failed:
        print("\nListe des maladies ignorées (aucune correspondance fiable trouvée) :")
        for disease_id, disease_name, current_type, _ in failed:
            print(f"  - id={disease_id}, disease_name={disease_name!r}, plant_type actuel={current_type!r}")
        print("\nRelancer en ajustant les mots-clés si nécessaire pour ces maladies.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Met à jour la colonne plant_type de diseases.db en l'ajustant aux types de plantes réels."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Chemin vers le fichier SQLite (par défaut backend/diseases.db).",
    )
    args = parser.parse_args()

    db_path = args.db
    if db_path is None:
        from database import get_db_path

        db_path = get_db_path()

    updated, skipped, failed = update_plant_types(db_path)
    print_summary(updated, skipped, failed)


if __name__ == "__main__":
    main()
