import json
from datetime import datetime

import pytest

import garantie


@pytest.fixture
def registre(tmp_path, monkeypatch):
    fichier = tmp_path / "garanties.json"
    fichier.write_text(
        json.dumps(
            {
                "PX200-0042": {
                    "modele": "PX-200",
                    "debut_garantie": "2026-01-01",
                    "fin_garantie": "2026-11-21",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(garantie, "REGISTRE", fichier)
    return fichier


@pytest.mark.parametrize(
    ("jour", "statut"),
    [
        ("2025-12-31", "non_commencee"),
        ("2026-01-01", "active"),
        ("2026-11-21", "active"),
        ("2026-11-22", "expiree"),
    ],
)
def test_dates_et_lecture_seule(registre, monkeypatch, jour, statut):
    class HorlogeFixe:
        @staticmethod
        def now(tz):
            return datetime.fromisoformat(jour).replace(tzinfo=tz)

    monkeypatch.setattr(garantie, "datetime", HorlogeFixe)
    avant = registre.read_bytes()
    resultat = garantie.consulter_garantie.invoke({"numero_serie": " px200-0042 "})
    assert resultat["statut"] == statut
    assert resultat["date_verification"] == jour
    assert registre.read_bytes() == avant


def test_numero_inconnu(registre):
    assert garantie.consulter_garantie.invoke({"numero_serie": "absent"}) == {
        "numero_serie": "ABSENT",
        "statut": "introuvable",
    }
