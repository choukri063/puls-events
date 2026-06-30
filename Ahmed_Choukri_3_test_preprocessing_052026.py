"""
test_preprocessing.py
Puls-Events — Tests unitaires : validation des données preprocessées
"""

import json
import pytest
from datetime import datetime, timezone, timedelta

JSON_PATH     = "data_processed.json"
REGIONS_OK    = {"Île-de-France", "Hauts-de-France"}
DATE_LIMITE   = datetime.now(timezone.utc) - timedelta(days=365)


@pytest.fixture(scope="module")
def evenements():
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) > 0, "Le fichier JSON est vide !"
    return data


# Test 1 : fichier non vide
def test_fichier_non_vide(evenements):
    assert len(evenements) > 0


# Test 2 : toutes les clés attendues sont présentes
def test_cles_presentes(evenements):
    cles_attendues = {
        "id", "titre", "description", "categorie",
        "ville", "region", "date_debut", "date_fin",
        "texte_document",
    }
    for evt in evenements:
        manquantes = cles_attendues - set(evt.keys())
        assert not manquantes, (
            f"Clés manquantes pour id={evt.get('id')} : {manquantes}"
        )


# Test 3 : régions uniquement Île-de-France ou Hauts-de-France
def test_regions_valides(evenements):
    for evt in evenements:
        assert evt["region"] in REGIONS_OK, (
            f"Région invalide : '{evt['region']}' pour id={evt['id']}"
        )


# Test 4 : pas d'événements antérieurs à 1 an (avec marge de 7 jours)
def test_dates_moins_un_an(evenements):
    DATE_LIMITE_STRICTE = datetime.now(timezone.utc) - timedelta(days=372)  # 365 + 7 jours de marge
    hors_periode = []
    for evt in evenements:
        date_fin_str = evt.get("date_fin", "")
        if not date_fin_str or str(date_fin_str).strip() in ("", "nan"):
            continue
        try:
            dt = datetime.fromisoformat(str(date_fin_str).strip())
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < DATE_LIMITE_STRICTE:
                hors_periode.append(evt["id"])
        except (ValueError, AttributeError):
            continue
    assert len(hors_periode) == 0, (
        f"{len(hors_periode)} événements hors période détectés : {hors_periode[:5]}"
    )


# Test 5 : titre ou description non vide
def test_titre_ou_description_non_vide(evenements):
    for evt in evenements:
        assert evt["titre"] or evt["description"], (
            f"Événement sans titre ni description : id={evt['id']}"
        )


# Test 6 : texte_document non vide
def test_texte_document_non_vide(evenements):
    for evt in evenements:
        assert evt["texte_document"].strip(), (
            f"texte_document vide pour id={evt['id']}"
        )


# Test 7 : pas de doublons sur l'id
def test_pas_de_doublons(evenements):
    ids = [evt["id"] for evt in evenements]
    assert len(ids) == len(set(ids)), (
        "Des doublons d'identifiants ont été détectés !"
    )


# Test 8 : volume minimum cohérent
def test_volume_minimum(evenements):
    assert len(evenements) >= 100, (
        f"Trop peu d'événements : {len(evenements)} (minimum attendu : 100)"
    )