"""Extraction réelle, modèles simulés et index en mémoire pour les tests courants."""

import asyncio
import threading
from copy import deepcopy

import pymupdf
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from PIL import Image

import ingestion
import mcp_server
import rag
import recherche
from recherche import figures_des_pages, rechercher_passages


@pytest.fixture
def documents_test(tmp_path):
    (tmp_path / "manuels").mkdir()
    (tmp_path / "photos").mkdir()
    photo = tmp_path / "photos" / "photo_test.png"
    Image.new("RGB", (32, 32), "red").save(photo)
    manuel = tmp_path / "manuels" / "manuel_test.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), "Le bac a papier est vide. Ajouter du papier.")
        page.insert_image(pymupdf.Rect(40, 80, 140, 180), filename=str(photo))
        for x in [40, 160, 400]:
            page.draw_line((x, 240), (x, 320))
        for y in [240, 280, 320]:
            page.draw_line((40, y), (400, y))
        for x, y, texte in [
            (50, 260, "Code"),
            (170, 260, "Action"),
            (50, 300, "E02"),
            (170, 300, "Ajouter du papier"),
        ]:
            page.insert_text((x, y), texte)
        pdf.save(manuel)
    return manuel, photo


@pytest.fixture
def bases(monkeypatch, tmp_path):
    etat = {}
    pixels = []

    class IndexMemoire:
        def __init__(self):
            self.noeuds = []

        def ajouter(self, noeuds):
            self.noeuds.extend(deepcopy(noeuds))

        def supprimer(self, version):
            self.noeuds = [n for n in self.noeuds if n.metadata["version"] != version]

        def documents(self, version=None, criteres=None):
            version = version or etat.get("version")
            return [
                Document(page_content=n.get_content(), metadata=n.metadata)
                for n in self.noeuds
                if n.metadata["version"] == version
                and all(n.metadata.get(k) == v for k, v in (criteres or {}).items())
            ]

        def proches(self, vecteur, nombre=6, sources=None):
            return [
                d
                for d in self.documents()
                if not sources or d.metadata["source"] in sources
            ][:nombre]

    textes, images = IndexMemoire(), IndexMemoire()

    def publier(version, compteurs):
        etat.clear()
        etat.update(version=version, **compteurs)

    def encoder(chemins):
        for chemin in chemins:
            with Image.open(chemin) as image:
                pixels.append(image.getpixel((0, 0)))
        return [[1.0, 0.0, 0.0] for _ in chemins]

    monkeypatch.setattr(ingestion, "DATA", tmp_path)
    monkeypatch.setattr(ingestion, "preparer_stockage", lambda: None)
    monkeypatch.setattr(ingestion, "publier_index", publier)
    monkeypatch.setattr(ingestion, "embeddings", lambda: MockEmbedding(embed_dim=16))
    monkeypatch.setattr(recherche, "embeddings", lambda: MockEmbedding(embed_dim=16))
    monkeypatch.setattr(
        ingestion, "decrire_image", lambda chemin: "Carré rouge visible dans la figure."
    )
    for module in [ingestion, rag]:
        monkeypatch.setattr(module, "base_documentaire", lambda: textes)
        monkeypatch.setattr(module, "base_figures", lambda: images)
        monkeypatch.setattr(module, "etat_index", lambda: etat or None)
        monkeypatch.setattr(module, "encoder_images", encoder)
    return textes, images, etat, pixels


def noeud(texte, page=1, source="manuel.pdf", type="texte", **metadata):
    return TextNode(
        text=texte,
        embedding=[0.5] * 16,
        metadata={
            "source": source,
            "page": page,
            "type": type,
            "version": "ancienne",
            **metadata,
        },
    )


def test_extraction_llamaindex_texte_image_et_tableau(bases, documents_test):
    manuel, _ = documents_test
    documents = ingestion.extraire_pdf(manuel)
    assert {n.metadata["type"] for n in documents} == {"texte", "figure", "tableau"}
    assert all(
        n.metadata["source"] == manuel.name and n.metadata["page"] == 1
        for n in documents
    )
    tableau = next(n for n in documents if n.metadata["type"] == "tableau")
    assert tableau.metadata["lignes"] == [
        ["Code", "Action"],
        ["E02", "Ajouter du papier"],
    ]
    assert "| E02 | Ajouter du papier |" in tableau.text
    figure = next(n for n in documents if n.metadata["type"] == "figure")
    assert (ingestion.DATA / figure.metadata["image"]).is_file()


def test_ingestion_indexe_les_descriptions_et_les_pixels(bases, documents_test):
    textes, images, etat, pixels = bases
    resultat = ingestion.main()
    assert resultat["tableaux"] == 1 and resultat["figures"] == 1
    assert pixels == [(255, 0, 0)]
    descriptions = [d for d in textes.documents() if d.metadata["type"] == "figure"]
    assert "Carré rouge" in descriptions[0].page_content
    assert (
        images.documents()[0].metadata["description"]
        == "Carré rouge visible dans la figure."
    )
    assert images.documents()[0].metadata["version"] == etat["version"]


def test_dessin_vectoriel_converti_en_figure_indexable(bases, tmp_path):
    manuel = tmp_path / "schema.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), "Schema de la machine")
        page.draw_circle((200, 200), 80, color=(1, 0, 0), fill=(1, 0, 0))
        pdf.save(manuel)
    figures = [
        n for n in ingestion.extraire_pdf(manuel) if n.metadata["type"] == "figure"
    ]
    assert len(figures) == 1
    assert figures[0].metadata["origine"] == "dessin_vectoriel"
    assert (tmp_path / figures[0].metadata["image"]).is_file()


def test_rag_recherche_texte_descriptions_et_sources(bases, monkeypatch):
    textes, images, etat, _ = bases
    etat["version"] = "ancienne"
    textes.ajouter([noeud("E02 : ajouter du papier.", page=4)])
    images.ajouter(
        [
            noeud(
                "Voyant papier rouge.",
                page=4,
                type="figure",
                image="figures/voyant.png",
            )
        ]
    )
    monkeypatch.setattr(rag, "modele", lambda: None)

    def agent_simule(model, tools, system_prompt):
        class Agent:
            async def ainvoke(self, etat, **kwargs):
                contenu = await tools[0].ainvoke({"requete": "E02 papier"})
                assert (
                    "ajouter du papier" in contenu and "figures/voyant.png" in contenu
                )
                return {
                    "messages": [
                        ToolMessage(content=contenu, tool_call_id="test"),
                        AIMessage(content="Ajouter du papier [manuel.pdf, page 4]."),
                    ]
                }

        return Agent()

    monkeypatch.setattr(rag, "create_agent", agent_simule)
    resultat = asyncio.run(rag.repondre_documentation("Que signifie E02 ?"))
    assert {s["type"] for s in resultat["sources"]} == {"texte", "figure"}
    assert "page 4" in resultat["reponse"]


def test_base_vide_ne_genere_pas_de_reponse(bases):
    resultat = asyncio.run(rag.repondre_documentation("Question sans source"))
    assert resultat["sources"] == [] and "vide" in resultat["reponse"]


def test_question_generale_recherche_meme_si_le_modele_oublie_outil(bases, monkeypatch):
    textes, _, etat, _ = bases
    etat["version"] = "ancienne"
    textes.ajouter([noeud("Guide utilisateur Epson ET-2850.", page=11)])
    monkeypatch.setattr(rag, "modele", lambda: None)
    appels = []

    class Agent:
        async def ainvoke(self, etat, **kwargs):
            appels.append(etat)
            if len(appels) == 1:
                return {"messages": [AIMessage(content="Précisez votre appareil.")]}
            messages = etat["messages"]
            assert messages[1].tool_calls[0]["name"] == "rechercher_manuels"
            assert isinstance(messages[2], ToolMessage)
            assert "Guide utilisateur Epson ET-2850" in messages[2].text
            return {
                "messages": [
                    *messages,
                    AIMessage(content="Le PDF est le guide Epson ET-2850, page 11."),
                ]
            }

    monkeypatch.setattr(rag, "create_agent", lambda **kwargs: Agent())
    resultat = asyncio.run(rag.repondre_documentation("c quoi le pdf"))
    assert len(appels) == 2
    assert "guide Epson" in resultat["reponse"]
    assert resultat["sources"][0]["page"] == 11


def test_recherche_procedure_pages_de_suite_et_bon_modele(bases, monkeypatch):
    textes, _, etat, _ = bases
    etat["version"] = "ancienne"
    source = "epson_et2850_guide_fr.pdf"
    donnees = [
        (
            8,
            "Bourrages papier................224\nBourrages arrière................227",
        ),
        (59, "Types de papier Epson compatibles."),
        (224, "Bourrage papier : retirez le papier coincé."),
        (225, "Ouvrez le scanner à deux mains."),
        (226, "Ne touchez pas au câble plat blanc."),
        (227, "Bourrage papier dans le capot arrière."),
    ]
    textes.ajouter(
        [noeud(t, page=p, source=source) for p, t in donnees]
        + [noeud("Bourrage papier", source="epson_et4750.pdf")]
    )
    monkeypatch.setattr(
        textes, "proches", lambda *args, **kwargs: textes.documents()[:1]
    )
    resultat = rechercher_passages(
        textes,
        "Comment retirer un bourrage papier sur mon Epson ET-2850 ?",
        "Papier Epson ET-4750",
    )
    assert {224, 225, 226}.issubset({d.metadata["page"] for d in resultat})
    assert all(d.metadata["source"] == source for d in resultat)
    assert 8 not in {d.metadata["page"] for d in resultat}
    assert len(resultat) <= 6


def test_recherche_descriptions_et_tableaux_interrogeables(bases):
    textes, _, etat, _ = bases
    etat["version"] = "ancienne"
    textes.ajouter(
        [
            noeud(
                "Capture montrant un voyant triangulaire rouge.", page=12, type="figure"
            ),
            noeud(
                "Tableau 1 : E02 | Ajouter du papier",
                page=4,
                type="tableau",
                lignes=[["E02", "Ajouter du papier"]],
            ),
        ]
    )
    assert (
        rechercher_passages(textes, "voyant triangulaire rouge")[0].metadata["page"]
        == 12
    )
    assert (
        "Ajouter du papier" in rechercher_passages(textes, "Code E02")[0].page_content
    )


def test_page_commencant_par_etape_quatre_conserve_la_precedente(bases):
    textes, _, etat, _ = bases
    etat["version"] = "ancienne"
    textes.ajouter(
        [
            noeud("14\n1. Ouvrez le capot.", page=14),
            noeud("15\n4. Retirez le papier coincé.", page=15),
        ]
    )
    assert [
        d.metadata["page"]
        for d in rechercher_passages(textes, "Retirer le papier coincé")[:2]
    ] == [14, 15]


def test_illustrations_appartiennent_aux_pages_retrouvees(bases):
    _, images, etat, _ = bases
    etat["version"] = "ancienne"
    images.ajouter(
        [
            noeud("Figure", page=p, type="figure", image=f"figures/{p}.png")
            for p in [93, 224, 225]
        ]
    )
    documents = [
        Document(page_content="Procédure", metadata={"source": "manuel.pdf", "page": p})
        for p in [224, 225, 226]
    ]
    assert [d.metadata["page"] for d in figures_des_pages(images, documents)] == [
        224,
        225,
    ]


def test_figure_principale_preferee_aux_icones(bases):
    _, images, etat, _ = bases
    etat["version"] = "ancienne"
    images.ajouter(
        [
            noeud("Icône", type="figure", image="icone.png", bbox=[0, 0, 14, 14]),
            noeud(
                "Petite capture",
                type="figure",
                image="petite.png",
                bbox=[0, 0, 100, 80],
            ),
            noeud(
                "Capture principale",
                type="figure",
                image="principale.png",
                bbox=[0, 0, 300, 200],
            ),
        ]
    )
    document = Document(
        page_content="Procédure", metadata={"source": "manuel.pdf", "page": 1}
    )
    assert (
        figures_des_pages(images, [document])[0].metadata["image"] == "principale.png"
    )
    images.noeuds = images.noeuds[:1]
    assert figures_des_pages(images, [document]) == []


@pytest.mark.parametrize("echec", ["", "texte", "figure", "description"])
def test_lots_et_version_active_preservee_en_cas_echec(
    bases, documents_test, monkeypatch, echec
):
    textes, images, etat, _ = bases
    _, photo = documents_test
    etat["version"] = "ancienne"
    textes.ajouter([noeud("Ancien manuel")])
    images.ajouter([noeud("Ancienne figure", type="figure")])
    donnees = [noeud(f"Passage {i}") for i in range(33)] + [
        noeud("Figure", type="figure", image=f"photos/{photo.name}") for _ in range(9)
    ]
    monkeypatch.setattr(ingestion, "extraire_pdf", lambda pdf: donnees)
    lots_textes, lots_figures, descriptions = [], [], []

    class EmbeddingsMesures(MockEmbedding):
        def get_text_embedding_batch(self, text_list, **kwargs):
            lots_textes.append(len(text_list))
            if echec == "texte" and len(lots_textes) == 2:
                raise ConnectionError("Ollama indisponible")
            return super().get_text_embedding_batch(text_list, **kwargs)

    def encoder(chemins):
        lots_figures.append(len(chemins))
        if echec == "figure" and len(lots_figures) == 2:
            raise RuntimeError("CLIP indisponible")
        return [[1.0, 0.0, 0.0] for _ in chemins]

    def decrire(chemin):
        descriptions.append(chemin)
        if echec == "description" and len(descriptions) == 2:
            raise RuntimeError("Vision indisponible")
        return "Carré rouge"

    monkeypatch.setattr(
        ingestion, "embeddings", lambda: EmbeddingsMesures(embed_dim=16)
    )
    monkeypatch.setattr(ingestion, "encoder_images", encoder)
    monkeypatch.setattr(ingestion, "decrire_image", decrire)
    if echec:
        with pytest.raises(RuntimeError, match="précédent reste actif"):
            ingestion.main()
        assert etat["version"] == "ancienne"
        assert [d.page_content for d in textes.documents()] == ["Ancien manuel"]
        assert [d.page_content for d in images.documents()] == ["Ancienne figure"]
    else:
        assert ingestion.main() == {"passages": 33, "tableaux": 0, "figures": 9}
        assert len(textes.documents()) == 42  # Passages et descriptions visuelles.
        assert len(images.documents()) == 9
        assert not textes.documents(version="ancienne")
        assert lots_textes == [16, 16, 10] and lots_figures == [4, 4, 1]


def test_mcp_conserve_les_references_sans_llm(monkeypatch):
    async def resultat_rag(question, photo_reference):
        return {
            "reponse": "Ajouter du papier.",
            "sources": [
                {
                    "source": "manuel.pdf",
                    "page": 4,
                    "type": "figure",
                    "image": "figures/voyant.png",
                }
            ],
        }

    monkeypatch.setattr(mcp_server, "repondre_documentation", resultat_rag)
    texte = asyncio.run(mcp_server.rag_multimodal("E02 ?"))
    assert texte.startswith("Ajouter du papier.")
    assert "manuel.pdf, page 4, figure : figures/voyant.png" in texte


@pytest.mark.parametrize("deconnexion", [False, True])
def test_mcp_partage_indexation_et_attend_avant_recherche(monkeypatch, deconnexion):
    terminer = threading.Event()
    appels = []

    async def executer():
        debut = asyncio.Event()
        boucle = asyncio.get_running_loop()

        def reconstruire():
            appels.append("indexation")
            boucle.call_soon_threadsafe(debut.set)
            if not terminer.wait(timeout=5):
                raise RuntimeError("Le test n'a pas libéré l'indexation")
            appels.append("fin indexation")
            return {"passages": 3, "figures": 1, "tableaux": 1}

        async def rechercher(question, photo_reference):
            appels.append("recherche")
            return {"reponse": "Prêt", "sources": []}

        monkeypatch.setattr(mcp_server, "verrou_index", asyncio.Lock())
        monkeypatch.setattr(mcp_server, "tache_indexation", None)
        monkeypatch.setattr(mcp_server, "reconstruire_index", reconstruire)
        monkeypatch.setattr(mcp_server, "repondre_documentation", rechercher)
        indexation = asyncio.create_task(mcp_server.indexer_manuels())
        try:
            await asyncio.wait_for(debut.wait(), timeout=5)
            doublon = asyncio.create_task(mcp_server.indexer_manuels())
            await asyncio.sleep(0)
            assert not doublon.done()
            if deconnexion:
                indexation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await indexation
                assert mcp_server.verrou_index.locked()
            recherche = asyncio.create_task(mcp_server.rag_multimodal("Question"))
            await asyncio.sleep(0)
            assert appels == ["indexation"]
            terminer.set()
            resultat = await doublon
            assert "1 tableaux" in resultat
            if not deconnexion:
                assert await indexation == resultat
            assert await recherche == "Prêt"
        finally:
            terminer.set()

    asyncio.run(executer())
    assert appels == ["indexation", "fin indexation", "recherche"]


def test_indexation_attend_la_recherche_en_cours(monkeypatch):
    async def executer():
        terminer = asyncio.Event()
        debut = asyncio.Event()
        appels = []

        async def rechercher(question, photo_reference):
            debut.set()
            await terminer.wait()
            appels.append("recherche terminée")
            return {"reponse": "Prêt", "sources": []}

        def reconstruire():
            appels.append("indexation")
            return {"passages": 3, "tableaux": 1, "figures": 1}

        monkeypatch.setattr(mcp_server, "verrou_index", asyncio.Lock())
        monkeypatch.setattr(mcp_server, "tache_indexation", None)
        monkeypatch.setattr(mcp_server, "repondre_documentation", rechercher)
        monkeypatch.setattr(mcp_server, "reconstruire_index", reconstruire)
        recherche = asyncio.create_task(mcp_server.rag_multimodal("Question"))
        await asyncio.wait_for(debut.wait(), timeout=5)
        indexation = asyncio.create_task(mcp_server.indexer_manuels())
        try:
            await asyncio.sleep(0)
            assert not indexation.done() and not appels
        finally:
            terminer.set()
        await asyncio.wait_for(recherche, timeout=5)
        assert "terminée" in await asyncio.wait_for(indexation, timeout=5)
        assert appels == ["recherche terminée", "indexation"]

    asyncio.run(executer())


def test_indexation_peut_reessayer_apres_echec(monkeypatch):
    appels = []

    def reconstruire():
        appels.append("indexation")
        if len(appels) == 1:
            raise RuntimeError("Vision indisponible")
        return {"passages": 3, "tableaux": 1, "figures": 1}

    async def executer():
        monkeypatch.setattr(mcp_server, "verrou_index", asyncio.Lock())
        monkeypatch.setattr(mcp_server, "tache_indexation", None)
        monkeypatch.setattr(mcp_server, "reconstruire_index", reconstruire)
        with pytest.raises(RuntimeError, match="Vision indisponible"):
            await mcp_server.indexer_manuels()
        assert not mcp_server.verrou_index.locked()
        assert "terminée" in await mcp_server.indexer_manuels()

    asyncio.run(executer())
    assert len(appels) == 2
