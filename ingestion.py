"""Extraire avec LlamaIndex, décrire les figures et indexer dans Qdrant."""

import hashlib
import re
from uuid import uuid4

import pymupdf
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageDocument, TextNode
from llama_index.readers.file import PyMuPDFReader

from config import DATA, embeddings
from stockage import (
    base_documentaire,
    base_figures,
    etat_index,
    preparer_stockage,
    publier_index,
)
from vision import decrire_image, encoder_images


def markdown_tableau(lignes):
    def ligne(cellules):
        return (
            "| "
            + " | ".join(
                (c or "").replace("\n", " ").replace("|", "\\|") for c in cellules
            )
            + " |"
        )

    return "\n".join(
        [
            ligne(lignes[0]),
            ligne(["---"] * len(lignes[0])),
            *[ligne(r) for r in lignes[1:]],
        ]
    )


def extraire_pdf(pdf):
    # Le lecteur et les documents LlamaIndex conservent une entrée par page.
    pages = PyMuPDFReader().load_data(file_path=pdf)
    empreinte = hashlib.sha256(pdf.read_bytes()).hexdigest()[:12]
    dossier = DATA / "figures" / f"{pdf.stem}_{empreinte}"
    dossier.mkdir(parents=True, exist_ok=True)
    documents, figures, tableaux = [], [], []
    with pymupdf.open(pdf) as manuel:
        for numero, page in enumerate(manuel, start=1):
            metadata = {"source": pdf.name, "page": numero, "type": "texte"}
            texte = pages[numero - 1].text.strip()
            if texte:
                documents.append(Document(text=texte, metadata=metadata))
            zones_tableaux = []
            for index, tableau in enumerate(
                page.find_tables(strategy="lines_strict", use_layout=False).tables,
                start=1,
            ):
                lignes = tableau.extract()
                if not lignes or not lignes[0]:
                    continue
                # Le détecteur peut ajouter une colonne de séparation entièrement vide.
                colonnes = [
                    i
                    for i in range(len(lignes[0]))
                    if any((r[i] or "").strip() for r in lignes)
                ]
                lignes = [[r[i] for i in colonnes] for r in lignes]
                if not colonnes:
                    continue
                zones_tableaux.append(pymupdf.Rect(tableau.bbox))
                # Les lignes/colonnes restent dans le payload de Qdrant.
                corps = lignes[1:] or [[]]
                for debut in range(0, len(corps), 15):
                    bloc = [lignes[0], *[r for r in corps[debut : debut + 15] if r]]
                    tableaux.append(
                        TextNode(
                            text=f"Tableau {index}, {pdf.name}, page {numero}.\n"
                            + markdown_tableau(bloc),
                            metadata={
                                **metadata,
                                "type": "tableau",
                                "tableau": index,
                                "lignes": bloc,
                                "ligne_debut": debut + 1,
                                "bbox": list(tableau.bbox),
                            },
                        )
                    )
            zones_images = []
            for index, bloc in enumerate(page.get_text("dict")["blocks"], start=1):
                if bloc["type"] != 1:
                    continue
                zones_images.append(pymupdf.Rect(bloc["bbox"]))
                chemin = dossier / f"page_{numero}_figure_{index}.png"
                pixels = pymupdf.Pixmap(bloc["image"])
                if pixels.colorspace is None or pixels.colorspace.n > 3:
                    pixels = pymupdf.Pixmap(pymupdf.csRGB, pixels)
                pixels.save(chemin)
                figures.append(
                    ImageDocument(
                        image_path=str(chemin),
                        metadata={
                            **metadata,
                            "type": "figure",
                            "image": str(chemin.relative_to(DATA)),
                            "origine": "image_integree",
                            "bbox": list(bloc["bbox"]),
                        },
                    )
                )
            # Les schémas vectoriels n'apparaissent pas dans les blocs image du PDF.
            dessins = [
                d
                for d in page.get_drawings()
                if max(d["rect"].width, d["rect"].height) >= 12
            ]
            # Les pages très complexes deviennent un aperçu unique, sans regroupement coûteux.
            if len(dessins) > 200:
                ensemble = pymupdf.Rect()
                for dessin in dessins:
                    ensemble |= dessin["rect"]
                zones = [ensemble]
            else:
                zones = page.cluster_drawings(drawings=dessins)
            for index, zone in enumerate(zones, start=1):
                zone = pymupdf.Rect(zone)
                if zone.width < 60 or zone.height < 60:
                    continue
                if any(
                    (zone & autre).get_area() / min(zone.get_area(), autre.get_area())
                    > 0.8
                    for autre in zones_tableaux + zones_images
                    if autre.get_area() > 0
                ):
                    continue
                clip = (
                    pymupdf.Rect(zone.x0 - 6, zone.y0 - 6, zone.x1 + 6, zone.y1 + 6)
                    & page.rect
                )
                if clip.is_empty:
                    continue
                chemin = dossier / f"page_{numero}_dessin_{index}.png"
                page.get_pixmap(clip=clip, dpi=120, alpha=False).save(chemin)
                figures.append(
                    ImageDocument(
                        image_path=str(chemin),
                        metadata={
                            **metadata,
                            "type": "figure",
                            "image": str(chemin.relative_to(DATA)),
                            "origine": "dessin_vectoriel",
                            "bbox": list(zone),
                        },
                    )
                )
    # Tokenisation locale explicite : aucun modèle/API distant n'est utilisé.
    splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=32,
        tokenizer=str.split,
        chunking_tokenizer_fn=lambda texte: re.split(r"(?<=[.!?])\s+|\n\n", texte),
    )
    return splitter.get_nodes_from_documents(documents) + tableaux + figures


def main():
    pdfs = sorted((DATA / "manuels").glob("*.pdf"))
    if not pdfs:
        raise ValueError(
            "Ajoute un manuel PDF dans data/manuels, ou importe-le depuis Streamlit."
        )
    preparer_stockage()
    ancien = etat_index()
    ancienne_version = ancien["version"] if ancien else None
    noeuds = []
    for pdf in pdfs:
        print(f"Extraction LlamaIndex : {pdf.name}", flush=True)
        noeuds.extend(extraire_pdf(pdf))
    if not noeuds:
        raise ValueError("Aucun texte, tableau ni figure extrait : vérifier les PDF.")
    version = uuid4().hex
    textes = [n for n in noeuds if n.metadata["type"] != "figure"]
    figures = [n for n in noeuds if n.metadata["type"] == "figure"]
    base, images = base_documentaire(), base_figures()
    modele = embeddings()
    compteurs = {
        "passages": sum(n.metadata["type"] == "texte" for n in textes),
        "tableaux": len(
            {
                (n.metadata["source"], n.metadata["page"], n.metadata["tableau"])
                for n in textes
                if n.metadata["type"] == "tableau"
            }
        ),
        "figures": len(figures),
    }
    try:
        for index, figure in enumerate(figures, start=1):
            description = decrire_image(DATA / figure.metadata["image"])
            textes.append(
                TextNode(
                    text="Description visuelle générée, à confirmer avec le manuel :\n"
                    + description,
                    metadata={**figure.metadata, "description": description},
                )
            )
            print(f"Descriptions visuelles : {index}/{len(figures)}", flush=True)
        for debut in range(0, len(textes), 16):
            lot = textes[debut : debut + 16]
            vecteurs = modele.get_text_embedding_batch([n.get_content() for n in lot])
            for i, (noeud, vecteur) in enumerate(zip(lot, vecteurs)):
                noeud.id_ = f"{version}_texte_{debut + i}"
                noeud.metadata["version"] = version
                noeud.embedding = vecteur
            base.ajouter(lot)
            print(
                f"Texte, tableaux et descriptions : {debut + len(lot)}/{len(textes)}",
                flush=True,
            )
        for debut in range(0, len(figures), 4):
            lot = figures[debut : debut + 4]
            vecteurs = encoder_images([DATA / n.metadata["image"] for n in lot])
            noeuds_images = []
            for i, (figure, vecteur) in enumerate(zip(lot, vecteurs)):
                description = decrire_image(DATA / figure.metadata["image"])
                noeuds_images.append(
                    TextNode(
                        id_=f"{version}_figure_{debut + i}",
                        text=description,
                        embedding=vecteur,
                        metadata={
                            **figure.metadata,
                            "description": description,
                            "version": version,
                        },
                    )
                )
            images.ajouter(noeuds_images)
            print(f"Figures CLIP : {debut + len(lot)}/{len(figures)}", flush=True)
    except Exception as erreur:
        base.supprimer(version)
        images.supprimer(version)
        raise RuntimeError(
            "Indexation interrompue ; l'index précédent reste actif. Les descriptions déjà calculées sont conservées pour réessayer."
            f" Détail : {erreur}"
        ) from erreur
    # Un seul point publie la version commune aux deux espaces vectoriels.
    publier_index(version, compteurs)
    if ancienne_version:
        base.supprimer(ancienne_version)
        images.supprimer(ancienne_version)
    print(
        f"Qdrant : {compteurs['passages']} passages, {compteurs['tableaux']} tableaux et {compteurs['figures']} figures décrites indexés.",
        flush=True,
    )
    return compteurs


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as erreur:
        raise SystemExit(str(erreur)) from erreur
