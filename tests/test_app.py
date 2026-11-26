"""Vérifier l'interface sans Ollama ni serveur MCP."""

from pathlib import Path

import httpx
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
import fichiers
import orchestrateur

APP = Path(__file__).resolve().parents[1] / "app.py"
REPONSE = (
    "Voir manuel.pdf, page 2.\n\nSources consultées :\n"
    "- manuel.pdf, page 2, figure : figures/voyant.png"
)


@pytest.fixture
def interface(monkeypatch, tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (8, 8)).save(photos / "appareil.png")
    figures = tmp_path / "figures"
    figures.mkdir()
    Image.new("RGB", (160, 120), "blue").save(figures / "voyant.png")
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    appels = []

    class GrapheSimule:
        async def astream(self, etat, stream_mode):
            appels.append(etat)
            assert stream_mode == "updates"
            yield {"orchestrateur": {"destination": "documentation"}}
            if etat["question"] == "Échec":
                raise ConnectionError("Serveur indisponible")
            yield {"agent_documentaire": {"reponse": REPONSE}}
            yield {"orchestrateur": {"destination": "fin"}}

    monkeypatch.setattr(orchestrateur, "construire_graphe", GrapheSimule)
    return AppTest.from_file(str(APP), default_timeout=20).run(), appels


def test_question_photo_sources_et_historique_independant(interface):
    app, appels = interface
    assert not app.exception
    assert not appels  # Ouvrir l'interface n'appelle aucun modèle.
    app.selectbox[0].select("appareil.png").run()
    app.chat_input[0].set_value("Que signifie E02 ?").run()
    assert not app.exception
    assert appels == [
        {"question": "Que signifie E02 ?", "photo_reference": "appareil.png"}
    ]
    assert app.chat_message[-1].markdown[0].value == REPONSE
    assert app.session_state.messages[-1]["figures"][0]["legende"] == (
        "manuel.pdf, page 2 — figure : figures/voyant.png"
    )
    assert len(app.chat_message[-1].image) == 1
    assert app.chat_message[-1].caption[0].value == (
        "Orchestrateur → Agent documentaire → Orchestrateur → END"
    )
    app.selectbox[0].select("").run()
    assert len(appels) == 1  # Un changement de photo ne relance pas la question.
    app.chat_input[0].set_value("Autre question").run()
    assert not app.exception
    assert appels[-1] == {"question": "Autre question", "photo_reference": ""}
    assert len(app.chat_message) == 4
    assert app.chat_message[0].caption[0].value == "Photo jointe : appareil.png"
    app.button(key="effacer").click().run()
    assert not app.chat_message
    assert len(appels) == 2


def test_erreur_visible_et_nouvelle_question_possible(interface):
    app, appels = interface
    app.chat_input[0].set_value("Échec").run()
    assert not app.exception
    assert "Ollama" in app.error[0].value
    assert app.code[0].value == "Serveur indisponible"
    app.chat_input[0].set_value("Réessayer").run()
    assert not app.exception
    assert len(appels) == 2
    assert app.chat_message[-1].markdown[0].value == REPONSE


def test_photo_supprimee_avant_envoi_sans_appel_modele(interface, monkeypatch):
    app, appels = interface
    app.selectbox[0].select("appareil.png").run()
    trouver_photo = config.trouver_photo
    lectures = []

    def photo_absente(reference):
        lectures.append(reference)
        if len(lectures) == 1:
            return trouver_photo(reference)  # L'aperçu fonctionne encore.
        raise FileNotFoundError("Photo supprimée")

    monkeypatch.setattr(config, "trouver_photo", photo_absente)
    app.chat_input[0].set_value("Une question avec photo").run()
    assert not app.exception
    assert app.code[0].value == "Photo supprimée"
    assert not appels


def test_import_pdf_indexation_et_erreur_recuperable(interface, monkeypatch):
    import pymupdf

    app, appels = interface
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 40), "Manuel de test")
        contenu = pdf.tobytes()
    app.file_uploader[0].upload("manuel.pdf", contenu, "application/pdf").run()
    assert not list((fichiers.DATA / "manuels").glob("*.pdf"))
    app.button(key="importer").click().run()
    assert not app.exception
    assert len(list((fichiers.DATA / "manuels").glob("*.pdf"))) == 1
    assert not appels

    async def echec():
        raise RuntimeError("Ollama indisponible")

    monkeypatch.setattr(fichiers, "indexer_manuels", echec)
    app.button(key="indexer").click().run()
    assert not app.exception
    assert "Ollama indisponible" in app.error[0].value
    assert app.session_state["import_pdf"]

    async def indexer():
        return "Indexation terminée : 1 passage et 0 figure."

    monkeypatch.setattr(fichiers, "indexer_manuels", indexer)
    app.button(key="indexer").click().run()
    assert not app.exception
    assert "Indexation terminée" in app.success[0].value
    assert "import_pdf" not in app.session_state


def test_indexation_affiche_la_cause_du_groupe_erreurs(interface, monkeypatch):
    app, _ = interface

    async def indexer():
        raise ExceptionGroup("TaskGroup", [httpx.ConnectError("")])

    monkeypatch.setattr(fichiers, "indexer_manuels", indexer)
    app.button(key="indexer").click().run()
    assert not app.exception
    assert "Serveur MCP injoignable" in app.error[0].value
    assert "TaskGroup" not in app.error[0].value


def test_import_photo_selectionne_sans_relancer_la_question(interface):
    from io import BytesIO

    app, appels = interface
    image = BytesIO()
    Image.new("RGB", (12, 12), "green").save(image, format="PNG")
    app.file_uploader[0].upload("photo.png", image.getvalue(), "image/png").run()
    app.button(key="importer").click().run()
    assert not app.exception
    reference = app.selectbox[0].value
    assert reference.startswith("photo_")
    assert (fichiers.DATA / "photos" / reference).is_file()
    assert not appels
    app.chat_input[0].set_value("Que montre le manuel ?").run()
    assert not app.exception
    assert appels[0]["photo_reference"] == reference
