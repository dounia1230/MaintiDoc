"""Parcours réel PDF/photo → MCP → LangGraph, activé avec RUN_MODEL_TESTS=1."""

import asyncio
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pymupdf
import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from PIL import Image, ImageDraw
from qdrant_client import QdrantClient

import agents
import config
from orchestrateur import construire_graphe


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="Modèles réels et Qdrant requis : RUN_MODEL_TESTS=1",
)
def test_pdf_photo_bge_qdrant_mcp_langgraph(tmp_path, monkeypatch):
    dossier = tmp_path / "corpus"
    (dossier / "manuels").mkdir(parents=True)
    (dossier / "photos").mkdir()
    photo = dossier / "photos" / "voyant.png"
    image = Image.new("RGB", (200, 160), "white")
    dessin = ImageDraw.Draw(image)
    dessin.polygon([(100, 20), (20, 140), (180, 140)], fill="red")
    dessin.text((90, 95), "E02", fill="black")
    image.save(photo)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            pymupdf.Rect(40, 40, 520, 160),
            "Manuel de test fictif MAINTiDOC. Equipement DEMO100.\n"
            "Code E02 : le bac a papier est vide. Ajouter du papier dans le bac.\n"
            "La figure represente un voyant triangulaire rouge.",
        )
        page.insert_image(pymupdf.Rect(40, 180, 240, 340), filename=str(photo))
        pdf.save(dossier / "manuels" / "manuel_smoke.pdf")
    with socket.socket() as prise:
        prise.bind(("127.0.0.1", 0))
        port = prise.getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    prefixe = "test_models_" + uuid4().hex
    environnement = {
        **os.environ,
        "QDRANT_PREFIX": prefixe,
        "MCP_URL": url,
        "PYTHONUNBUFFERED": "1",
    }
    demarrage = (
        "import sys; from pathlib import Path; import config; "
        "config.DATA=Path(sys.argv[1]); import mcp_server; "
        "mcp_server.mcp.settings.port=int(sys.argv[2]); "
        "mcp_server.mcp.run(transport='streamable-http')"
    )
    monkeypatch.setattr(agents, "MCP_URL", url)
    journal = tmp_path / "mcp.log"

    async def parcours():
        client = MultiServerMCPClient(
            {
                "documentation": {
                    "transport": "streamable_http",
                    "url": url,
                    "sse_read_timeout": 1800,
                }
            }
        )
        async with client.session("documentation") as session:
            resultat = await session.call_tool(
                "indexer_manuels", {}, read_timeout_seconds=timedelta(minutes=30)
            )
        assert not resultat.isError, resultat.content
        texte_index = "\n".join(
            bloc.text for bloc in resultat.content if bloc.type == "text"
        )
        print("\nMCP :", texte_index)
        assert "1 figures" in texte_index
        etat = await construire_graphe().ainvoke(
            {
                "question": "Que signifie le code E02 sur le DEMO100 ? Cite le manuel et la page.",
                "photo_reference": "voyant.png",
            },
            config={"recursion_limit": 12},
        )
        print("Réponse réelle :", etat["reponse"])
        assert "manuel_smoke.pdf" in etat["reponse"]
        assert "page 1" in etat["reponse"]
        assert "papier" in etat["reponse"].lower()

    with journal.open("w", encoding="utf-8") as sortie:
        serveur = subprocess.Popen(
            [sys.executable, "-c", demarrage, str(dossier), str(port)],
            cwd=Path(__file__).resolve().parents[1],
            env=environnement,
            stdout=sortie,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            limite = time.monotonic() + 60
            with httpx.Client(timeout=1) as http:
                while True:
                    assert serveur.poll() is None, journal.read_text(encoding="utf-8")
                    try:
                        http.get(url)
                        break
                    except httpx.TransportError:
                        if time.monotonic() > limite:
                            pytest.fail(journal.read_text(encoding="utf-8"))
                        time.sleep(0.25)
            asyncio.run(parcours())
        except BaseException:
            print(journal.read_text(encoding="utf-8"))
            raise
        finally:
            if os.name == "nt":
                # Le python.exe du venv peut lancer un processus Python enfant.
                subprocess.run(
                    ["taskkill", "/PID", str(serveur.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                serveur.terminate()
            serveur.wait(timeout=20)
            qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
            try:
                for collection in qdrant.get_collections().collections:
                    if collection.name.startswith(prefixe + "_"):
                        qdrant.delete_collection(collection.name)
            finally:
                qdrant.close()
