"""Vérifier les imports et les références d'images sans modèle."""

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pymupdf
import pytest
from PIL import Image

import fichiers


def image_png(taille=(160, 120)):
    image = BytesIO()
    Image.new("RGB", taille).save(image, format="PNG")
    return image.getvalue()


def test_pdf_valide_sans_ecrasement_ni_sortie_du_dossier(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 40), "Premier manuel")
        contenu = pdf.tobytes()
        chemin = fichiers.enregistrer_fichier("../../manuel.pdf", contenu)
        assert fichiers.enregistrer_fichier("manuel.pdf", contenu) == chemin
        pdf[0].insert_text((40, 80), "Nouvelle version")
        autre = fichiers.enregistrer_fichier("manuel.pdf", pdf.tobytes())
    assert chemin.parent == tmp_path / "manuels"
    assert chemin != autre
    assert chemin.read_bytes() == contenu
    assert len(list(chemin.parent.glob("*.pdf"))) == 2


@pytest.mark.parametrize(
    "nom,contenu",
    [
        ("manuel.pdf", b"invalide"),
        ("photo.png", b"invalide"),
        ("photo.jpg", image_png()),
        ("script.py", b"print(1)"),
        ("vide.pdf", b""),
    ],
)
def test_import_invalide_necrit_rien(tmp_path, monkeypatch, nom, contenu):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    with pytest.raises(ValueError):
        fichiers.enregistrer_fichier(nom, contenu)
    assert not list(tmp_path.iterdir())


def test_figures_citees_et_pixels_conserves_sans_chemin_externe(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    dossier = tmp_path / "figures"
    dossier.mkdir()
    (dossier / "voyant.png").write_bytes(image_png())
    (tmp_path / "prive.png").write_bytes(image_png())
    reponse = (
        "Réponse.\n\nSources consultées :\n"
        "- manuel.pdf, page 4, figure : figures/voyant.png\n"
        "- manuel.pdf, page 4, figure : figures/voyant.png\n"
        "- autre.pdf, page 1, figure : figures/../prive.png\n"
        "- autre.pdf, page 1, figure : figures/absente.png\n"
    )
    figures = fichiers.figures_reponse(reponse)
    assert len(figures) == 1
    assert figures[0]["legende"] == "manuel.pdf, page 4 — figure : figures/voyant.png"
    (dossier / "voyant.png").unlink()
    assert figures[0]["contenu"] == image_png()
    assert fichiers.figures_reponse("Aucune source") == []


def test_apercu_des_pages_citees_conserve_les_dessins_vectoriels(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    dossier = tmp_path / "manuels"
    dossier.mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=200, height=200)
        page.draw_rect(pymupdf.Rect(40, 40, 160, 160), color=(1, 0, 0), fill=(1, 0, 0))
        pdf.new_page(width=200, height=200)
        pdf.save(dossier / "manuel.pdf")
    reponse = (
        "Procédure.\n\nSources consultées :\n"
        "- manuel.pdf, page 1\n- manuel.pdf, page 1\n- manuel.pdf, page 2"
    )
    apercus = fichiers.figures_reponse(reponse)
    assert len(apercus) == 2
    assert apercus[0]["legende"] == "manuel.pdf, page 1 — aperçu du manuel"
    with Image.open(BytesIO(apercus[0]["contenu"])) as image:
        assert image.getpixel((image.width // 2, image.height // 2)) == (255, 0, 0)
    (tmp_path / "prive.pdf").write_bytes((dossier / "manuel.pdf").read_bytes())
    assert (
        fichiers.figures_reponse(
            "\n\nSources consultées :\n- ../prive.pdf, page 1\n- manuel.pdf, page 0\n- manuel.pdf, page 999"
        )
        == []
    )


@pytest.mark.parametrize("page_figure", [1, 2])
@pytest.mark.parametrize("citation", ["Source : manuel.pdf, page 1", "page 1"])
def test_icone_ou_figure_hors_reponse_remplacee_par_page_citee(
    tmp_path, monkeypatch, page_figure, citation
):
    monkeypatch.setattr(fichiers, "DATA", tmp_path)
    (tmp_path / "manuels").mkdir()
    (tmp_path / "figures").mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=200, height=200)
        page.draw_rect((40, 40, 160, 160), color=(1, 0, 0), fill=(1, 0, 0))
        pdf.new_page(width=200, height=200)
        pdf.save(tmp_path / "manuels" / "manuel.pdf")
    # Même page : icône minuscule. Autre page : vraie figure, mais hors réponse.
    taille = (14, 14) if page_figure == 1 else (360, 260)
    (tmp_path / "figures" / "capture.png").write_bytes(image_png(taille))
    resultat = fichiers.figures_reponse(
        f"Enregistrer le fichier ({citation}).\n\nSources consultées :\n"
        "- manuel.pdf, page 1\n- manuel.pdf, page 2\n"
        f"- manuel.pdf, page {page_figure}, figure : figures/capture.png"
    )
    assert len(resultat) == 1
    assert resultat[0]["legende"] == "manuel.pdf, page 1 — aperçu du manuel"
    with Image.open(BytesIO(resultat[0]["contenu"])) as image:
        assert image.getpixel((image.width // 2, image.height // 2)) == (255, 0, 0)


@pytest.mark.parametrize("erreur", [False, True])
def test_indexation_utilise_mcp_et_remonte_ses_erreurs(monkeypatch, erreur):
    class Session:
        async def call_tool(self, nom, arguments, read_timeout_seconds):
            assert nom == "indexer_manuels"
            assert arguments == {}
            assert read_timeout_seconds.total_seconds() == 1800
            return SimpleNamespace(
                isError=erreur,
                content=[SimpleNamespace(type="text", text="Résultat MCP")],
            )

    class Client:
        def __init__(self, configuration):
            assert configuration["documentation"]["transport"] == "streamable_http"
            assert configuration["documentation"]["sse_read_timeout"] == 1800

        @asynccontextmanager
        async def session(self, nom):
            assert nom == "documentation"
            yield Session()

    monkeypatch.setattr(fichiers, "MultiServerMCPClient", Client)
    if erreur:
        with pytest.raises(RuntimeError, match="Résultat MCP"):
            asyncio.run(fichiers.indexer_manuels())
    else:
        assert asyncio.run(fichiers.indexer_manuels()) == "Résultat MCP"


def test_groupe_erreurs_mcp_deplie_la_cause_sans_perdre_les_details():
    erreur = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [
            ExceptionGroup("groupe interne", [httpx.ReadTimeout("")]),
            ValueError("Détail utile"),
            ValueError("Détail utile"),
        ],
    )
    message = fichiers.message_erreur(erreur)
    assert "délai d'attente" in message
    assert "continuer côté serveur" in message
    assert message.count("Détail utile") == 1
    assert "TaskGroup" not in message


def test_connexion_mcp_refusee_donne_la_commande_de_demarrage():
    assert "uv run mcp_server.py" in fichiers.message_erreur(httpx.ConnectError(""))


@pytest.mark.parametrize("expiration", [False, True])
def test_vrai_transport_mcp_et_expiration_de_la_connexion(
    monkeypatch, tmp_path, expiration
):
    # Vrai échange HTTP local ; seul le travail d'indexation est simulé.
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from streamlit.testing.v1 import AppTest

    import config

    monkeypatch.setattr(config, "DATA", tmp_path)

    mcp = FastMCP("test-indexation", stateless_http=True, json_response=True)

    @mcp.tool()
    async def indexer_manuels() -> str:
        await asyncio.sleep(0.1)
        return "Indexation de test terminée"

    async def executer():
        serveur = uvicorn.Server(
            uvicorn.Config(
                mcp.streamable_http_app(),
                host="127.0.0.1",
                port=0,
                log_level="critical",
            )
        )
        tache = asyncio.create_task(serveur.serve())
        try:
            async with asyncio.timeout(5):
                while not serveur.started:
                    if tache.done():
                        await tache
                    await asyncio.sleep(0.01)
            port = serveur.servers[0].sockets[0].getsockname()[1]
            monkeypatch.setattr(fichiers, "MCP_URL", f"http://127.0.0.1:{port}/mcp")
            if expiration:
                client_mcp = fichiers.MultiServerMCPClient

                def client_court(configuration):
                    configuration["documentation"]["sse_read_timeout"] = 0.02
                    return client_mcp(configuration)

                monkeypatch.setattr(fichiers, "MultiServerMCPClient", client_court)
                with pytest.raises(RuntimeError, match="délai d'attente") as erreur:
                    await fichiers.indexer_manuels()
                assert "TaskGroup" not in str(erreur.value)
            else:
                assert await fichiers.indexer_manuels() == "Indexation de test terminée"

            def verifier_interface():
                app = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "app.py"),
                    default_timeout=20,
                ).run()
                app.button(key="indexer").click().run()
                assert not app.exception
                if expiration:
                    assert "délai d'attente" in app.error[0].value
                    assert "TaskGroup" not in app.error[0].value
                else:
                    assert app.success[0].value == "Indexation de test terminée"

            # Laisser la boucle HTTP active pendant que Streamlit utilise son propre thread.
            await asyncio.to_thread(verifier_interface)
        finally:
            serveur.should_exit = True
            await asyncio.wait_for(tache, timeout=5)

    asyncio.run(executer())


def test_deux_clients_mcp_attendent_une_seule_indexation(monkeypatch):
    import threading

    import uvicorn
    from mcp.server.fastmcp import FastMCP

    import mcp_server

    terminer = threading.Event()
    calculs = []

    def reconstruire():
        calculs.append("indexation")
        if not terminer.wait(timeout=10):
            raise RuntimeError("Indexation de test non libérée")
        return {"passages": 3, "tableaux": 1, "figures": 1}

    async def executer():
        clients = []
        deux_clients = asyncio.Event()
        monkeypatch.setattr(mcp_server, "verrou_index", asyncio.Lock())
        monkeypatch.setattr(mcp_server, "tache_indexation", None)
        monkeypatch.setattr(mcp_server, "reconstruire_index", reconstruire)
        mcp = FastMCP("test-concurrence", stateless_http=True, json_response=True)

        @mcp.tool()
        async def indexer_manuels() -> str:
            clients.append("appel")
            if len(clients) == 2:
                deux_clients.set()
            return await mcp_server.indexer_manuels()

        serveur = uvicorn.Server(
            uvicorn.Config(
                mcp.streamable_http_app(),
                host="127.0.0.1",
                port=0,
                log_level="critical",
            )
        )
        tache = asyncio.create_task(serveur.serve())
        try:
            async with asyncio.timeout(5):
                while not serveur.started:
                    if tache.done():
                        await tache
                    await asyncio.sleep(0.01)
            port = serveur.servers[0].sockets[0].getsockname()[1]
            monkeypatch.setattr(fichiers, "MCP_URL", f"http://127.0.0.1:{port}/mcp")
            premier = asyncio.create_task(fichiers.indexer_manuels())
            second = asyncio.create_task(fichiers.indexer_manuels())
            await asyncio.wait_for(deux_clients.wait(), timeout=5)
            assert not premier.done() and not second.done()
            terminer.set()
            resultats = await asyncio.wait_for(
                asyncio.gather(premier, second), timeout=5
            )
            assert resultats[0] == resultats[1]
            assert "Indexation terminée" in resultats[0]
            assert calculs == ["indexation"]
        finally:
            terminer.set()
            serveur.should_exit = True
            await asyncio.wait_for(tache, timeout=5)

    asyncio.run(executer())
