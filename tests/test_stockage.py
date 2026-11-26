"""Contrat Qdrant réel : mémoire, disque et serveur HTTP optionnel."""

import os
from uuid import uuid4

import pytest
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient, models

import stockage

MODES = ["memoire", "disque"]
if os.getenv("RUN_QDRANT_TESTS") == "1":
    MODES.append("serveur")


@pytest.fixture(params=MODES)
def client(monkeypatch, tmp_path, request):
    prefixe = "test_maintidoc_" + uuid4().hex
    if request.param == "serveur":
        qdrant = QdrantClient(url=stockage.QDRANT_URL, api_key=stockage.QDRANT_API_KEY)
    elif request.param == "disque":
        qdrant = QdrantClient(path=str(tmp_path / "qdrant"))
    else:
        qdrant = QdrantClient(":memory:")
    monkeypatch.setattr(stockage, "client_qdrant", lambda: qdrant)
    monkeypatch.setattr(stockage, "QDRANT_PREFIX", prefixe)
    yield qdrant
    if request.param == "serveur":
        for collection in qdrant.get_collections().collections:
            if collection.name.startswith(prefixe + "_"):
                qdrant.delete_collection(collection.name)
    qdrant.close()


def noeud(version="v1", source="manuel.pdf", **metadata):
    return TextNode(
        text="Code E02 : ajouter du papier",
        embedding=[1.0, 0.0, 0.0],
        metadata={"version": version, "source": source, "page": 4, **metadata},
    )


def test_versions_communes_metadonnees_et_filtres(client):
    textes = stockage.IndexQdrant("textes", 3)
    images = stockage.IndexQdrant("images", 3)
    tableau = noeud(type="tableau", lignes=[["Code", "Action"], ["E02", "Papier"]])
    textes.ajouter([tableau])
    images.ajouter([noeud(type="figure", image="figures/schema.png")])
    assert stockage.etat_index() is None
    assert textes.documents() == images.documents() == []
    stockage.publier_index("v1", {"passages": 0, "tableaux": 1, "figures": 1})
    assert textes.documents()[0].metadata["lignes"][1] == ["E02", "Papier"]
    assert textes.proches([1.0, 0.0, 0.0])[0].metadata["page"] == 4
    assert textes.documents(criteres={"page": 4})
    assert not textes.documents(criteres={"page": 9})
    assert not textes.proches([1.0, 0.0, 0.0], sources={"autre.pdf"})
    textes.ajouter([noeud(version="v2", source="nouveau.pdf")])
    assert textes.documents()[0].metadata["version"] == "v1"
    assert images.documents()[0].metadata["version"] == "v1"
    images.ajouter([noeud(version="v2", type="figure")])
    stockage.publier_index("v2", {"passages": 1, "tableaux": 0, "figures": 1})
    assert textes.documents()[0].metadata["source"] == "nouveau.pdf"
    assert images.documents()[0].metadata["version"] == "v2"
    textes.supprimer("v1")
    images.supprimer("v1")
    assert not textes.documents(version="v1")
    assert textes.documents() and images.documents()
    assert stockage.etat_index()["embedding_model"] == stockage.EMBEDDING_MODEL


def test_pagination_et_ajout_idempotent(client):
    index = stockage.IndexQdrant("pagination", 3)
    noeuds = [noeud(page=i) for i in range(300)]
    index.ajouter(noeuds[:150])
    index.ajouter(noeuds[150:])
    index.ajouter(noeuds[:1])
    stockage.publier_index("v1", {"passages": 300})
    assert len(index.documents()) == 300
    assert {d.metadata["page"] for d in index.documents()} == set(range(300))


def test_dimension_incorrecte_necrit_rien(client):
    index = stockage.IndexQdrant("dimension", 3)
    incorrect = noeud()
    incorrect.embedding = [1.0, 0.0]
    with pytest.raises(ValueError, match="dimension 2"):
        index.ajouter([noeud(), incorrect])
    assert not index.documents(version="v1")
    with pytest.raises(ValueError, match="dimension 2"):
        index.proches([1.0, 0.0])
    with pytest.raises(ValueError, match="incompatible"):
        stockage.IndexQdrant("dimension", 4)


def test_changement_modele_isole_index_et_publication(client, monkeypatch):
    ancien = stockage.IndexQdrant("textes", 3)
    ancien.ajouter([noeud()])
    stockage.publier_index("v1", {"passages": 1})
    monkeypatch.setattr(stockage, "EMBEDDING_MODEL", "autre-modele")
    nouveau = stockage.IndexQdrant("textes", 3)
    assert nouveau.collection != ancien.collection
    assert stockage.etat_index() is None
    assert not nouveau.documents()


def test_collection_preexistante_de_mauvaise_dimension(client):
    nom = f"{stockage.QDRANT_PREFIX}_incompatible_{stockage.cle_index()}"
    client.create_collection(
        nom, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    with pytest.raises(ValueError, match="incompatible"):
        stockage.IndexQdrant("incompatible", 3)


def test_persistance_apres_reouverture(tmp_path, monkeypatch):
    chemin = str(tmp_path / "persistant")
    premier = QdrantClient(path=chemin)
    monkeypatch.setattr(stockage, "client_qdrant", lambda: premier)
    index = stockage.IndexQdrant("persistance", 3)
    index.ajouter([noeud()])
    stockage.publier_index("v1", {"passages": 1})
    premier.close()
    second = QdrantClient(path=chemin)
    monkeypatch.setattr(stockage, "client_qdrant", lambda: second)
    try:
        assert stockage.etat_index()["version"] == "v1"
        assert (
            index.proches([1.0, 0.0, 0.0])[0].page_content
            == "Code E02 : ajouter du papier"
        )
    finally:
        second.close()
