#!/usr/bin/env python3
import shutil
from pathlib import Path

def reset_directory(dir_path: str):
    """
    Supprime récursivement un dossier s'il existe, puis le recrée vide.
    """
    path = Path(dir_path)
    # 1. Suppression récursive si le dossier existe
    if path.exists() and path.is_dir():
        print(f"[-] Nettoyage de l'arborescence : {path.resolve()}")
        shutil.rmtree(path)
    elif path.exists():
        print(f"[!] Attention : {path} existe mais n'est pas un dossier.")
        return
    # 2. Recréation du dossier racine (les sous-dossiers seront gérés par le code C)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[+] Dossier initialisé   : {dir_path}/")

if __name__ == "__main__":
    print("=== Initialisation de l'espace de travail ===")
    # Liste des répertoires d'I/O à purger
    target_dirs = ["vtk_pieces", "vtu"]
    for d in target_dirs:
        reset_directory(d)
    print("=== Espace de travail prêt pour la simulation ===")