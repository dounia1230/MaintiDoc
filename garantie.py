"""Registre de garanties fictif, utilisable sans modèle de langage."""

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

from langchain.tools import tool

from config import DATA

REGISTRE = DATA / "garanties.json"


@tool
def consulter_garantie(numero_serie: str) -> dict:
    """Consulter la garantie d'un appareil par numéro de série, en lecture seule."""
    numero = numero_serie.strip().upper()
    registre = json.loads(REGISTRE.read_text(encoding="utf-8"))
    appareil = registre.get(numero)
    if appareil is None:
        return {"numero_serie": numero, "statut": "introuvable"}

    aujourd_hui = datetime.now(ZoneInfo("Africa/Casablanca")).date()
    debut = date.fromisoformat(appareil["debut_garantie"])
    fin = date.fromisoformat(appareil["fin_garantie"])
    if aujourd_hui < debut:
        statut = "non_commencee"
    elif aujourd_hui <= fin:
        statut = "active"
    else:
        statut = "expiree"
    return {
        "numero_serie": numero,
        **appareil,
        "date_verification": aujourd_hui.isoformat(),
        "statut": statut,
    }


if __name__ == "__main__":
    import sys

    numero = sys.argv[1] if len(sys.argv) > 1 else "PX200-0042"
    print(
        json.dumps(
            consulter_garantie.invoke({"numero_serie": numero}),
            indent=2,
            ensure_ascii=False,
        )
    )
