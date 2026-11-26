"""Collections Qdrant texte/CLIP et pointeur commun de version publiée."""

import hashlib
from functools import cache
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from llama_index.core.schema import MetadataMode
from qdrant_client import QdrantClient, models

from config import (
    CLIP_CHECKPOINT,
    CLIP_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_PREFIX,
    QDRANT_URL,
    VISION_MODEL,
)


def cle_index():
    configuration = f"v2:{EMBEDDING_MODEL}:{EMBEDDING_DIM}:{VISION_MODEL}:{CLIP_MODEL}:{CLIP_CHECKPOINT}"
    return hashlib.sha256(configuration.encode()).hexdigest()[:12]


@cache
def client_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def preparer_collection(nom, dimension):
    client = client_qdrant()
    if not client.collection_exists(nom):
        client.create_collection(
            collection_name=nom,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )
    vecteurs = client.get_collection(nom).config.params.vectors
    if not isinstance(vecteurs, models.VectorParams) or vecteurs.size != dimension:
        raise ValueError(
            f"Collection {nom} incompatible avec la dimension {dimension}. "
            "Vérifiez EMBEDDING_MODEL/EMBEDDING_DIM et réindexez dans un nouveau QDRANT_PREFIX."
        )


def preparer_stockage():
    preparer_collection(f"{QDRANT_PREFIX}_index_actif", 1)


def id_etat():
    return str(uuid5(NAMESPACE_URL, f"{QDRANT_PREFIX}:{cle_index()}"))


def etat_index():
    preparer_stockage()
    points = client_qdrant().retrieve(
        collection_name=f"{QDRANT_PREFIX}_index_actif",
        ids=[id_etat()],
        with_payload=True,
        with_vectors=False,
    )
    return points[0].payload if points else None


def publier_index(version, compteurs):
    preparer_stockage()
    # Un seul point publie la version commune après le succès des deux collections.
    client_qdrant().upsert(
        collection_name=f"{QDRANT_PREFIX}_index_actif",
        points=[
            models.PointStruct(
                id=id_etat(),
                vector=[1.0],
                payload={
                    "version": version,
                    **compteurs,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_dim": EMBEDDING_DIM,
                    "vision_model": VISION_MODEL,
                    "clip_model": CLIP_MODEL,
                    "clip_checkpoint": CLIP_CHECKPOINT,
                },
            )
        ],
        wait=True,
    )


class IndexQdrant:
    def __init__(self, nom, dimension):
        self.collection = f"{QDRANT_PREFIX}_{nom}_{cle_index()}"
        self.dimension = dimension
        preparer_collection(self.collection, dimension)

    def filtres(self, version=None, criteres=None):
        if version is None:
            etat = etat_index()
            version = etat["version"] if etat else "aucune_version_active"
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.version", match=models.MatchValue(value=version)
                ),
                *[
                    models.FieldCondition(
                        key=f"metadata.{cle}", match=models.MatchValue(value=valeur)
                    )
                    for cle, valeur in (criteres or {}).items()
                ],
            ]
        )

    def verifier_vecteur(self, vecteur):
        if len(vecteur) != self.dimension:
            raise ValueError(
                f"Embedding de dimension {len(vecteur)} ; {self.dimension} attendue "
                f"pour {self.collection}. Vérifiez le modèle et réindexez."
            )

    def ajouter(self, noeuds):
        points = []
        for noeud in noeuds:
            vecteur = noeud.get_embedding()
            self.verifier_vecteur(vecteur)
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, noeud.node_id)),
                    vector=vecteur,
                    payload={
                        "text": noeud.get_content(metadata_mode=MetadataMode.NONE),
                        "metadata": noeud.metadata,
                    },
                )
            )
        if points:
            client_qdrant().upsert(
                collection_name=self.collection, points=points, wait=True
            )

    def supprimer(self, version):
        client_qdrant().delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self.filtres(version)),
            wait=True,
        )

    @staticmethod
    def document(point):
        return Document(
            page_content=point.payload["text"], metadata=point.payload["metadata"]
        )

    def documents(self, version=None, criteres=None):
        resultat, offset = [], None
        filtres = self.filtres(version, criteres)
        while True:
            points, offset = client_qdrant().scroll(
                collection_name=self.collection,
                scroll_filter=filtres,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            resultat.extend(self.document(point) for point in points)
            if offset is None:
                return resultat

    def proches(self, vecteur, nombre=6, sources=None):
        self.verifier_vecteur(vecteur)
        filtres = self.filtres()
        if sources:
            filtres.must.append(
                models.FieldCondition(
                    key="metadata.source", match=models.MatchAny(any=sorted(sources))
                )
            )
        points = (
            client_qdrant()
            .query_points(
                collection_name=self.collection,
                query=vecteur,
                query_filter=filtres,
                limit=nombre,
                with_payload=True,
                with_vectors=False,
            )
            .points
        )
        return [self.document(point) for point in points]


@cache
def base_documentaire():
    return IndexQdrant("passages", EMBEDDING_DIM)


@cache
def base_figures():
    return IndexQdrant("figures", 512)
