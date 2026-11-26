"""Recherche dans les passages indexés, avec mots-clés et pages de suite."""

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import snowballstemmer
from langchain_core.documents import Document

from config import embeddings

MOTS_VIDES = set(
    "a au aux avec ce ces comment dans de des du elle en est et fait il ils je la le "
    "les leur lui ma mais me mes mon ne nous on ou par pas pour que quel quelle "
    "quelles quels qui sa se ses son sur ta te tes ton tu un une vos votre vous "
    "explique expliquer etape etapes cite citer page pages manuel manuels utilise utilisees "
    "imprimante imprimantes appareil appareils produit produits".split()
)


@lru_cache(maxsize=4096)
def mots(texte):
    texte = "".join(
        lettre
        for lettre in unicodedata.normalize("NFD", texte.lower())
        if not unicodedata.combining(lettre)
    )
    texte = re.sub(r"(?<=\w)-(?=\w)", "", texte)
    texte = re.sub(r"\b([a-z]+) (\d+)\b", r"\1\2", texte)
    termes = [
        mot
        for mot in re.findall(r"[a-z0-9]+", texte)
        if mot not in MOTS_VIDES and (len(mot) > 2 or any(c.isdigit() for c in mot))
    ]
    return snowballstemmer.stemmer("french").stemWords(termes)


def rechercher_passages(base, question, reformulation=""):
    contenu = base.documents()
    pages_indexees = {}
    for document in contenu:
        texte, metadata = document.page_content, document.metadata
        cle = (metadata["source"], metadata["page"])
        if cle not in pages_indexees:
            pages_indexees[cle] = Document(page_content=texte, metadata=metadata)
        else:
            pages_indexees[cle].page_content += "\n" + texte
    # Un fragment de table des matières ne doit pas passer devant la procédure.
    documents = [
        doc
        for doc in pages_indexees.values()
        if not (
            re.search(r"\.{5,}", doc.page_content)
            and doc.page_content.count(".") / max(len(doc.page_content), 1) >= 0.08
        )
    ]
    if not documents:
        return []
    requete = set(mots(question + " " + reformulation))
    # Quand le modèle est nommé dans le fichier, rester dans son manuel.
    modeles = {
        mot for mot in mots(question) if re.fullmatch(r"[a-z]+\d+[a-z0-9]*", mot)
    }
    sources = {
        doc.metadata["source"]
        for doc in documents
        if modeles.intersection(mots(doc.metadata["source"]))
    }
    if sources:
        documents = [doc for doc in documents if doc.metadata["source"] in sources]
        # Le modèle sert à choisir le manuel, pas à privilégier sa page de couverture.
        requete.difference_update(mots(" ".join(sources)))
    semantiques = base.proches(
        embeddings().get_query_embedding(question), sources=sources
    )

    # BM25 : les mots rares, comme un code d'erreur, pèsent plus que « papier ».
    termes = [Counter(mots(doc.page_content)) for doc in documents]
    frequences = Counter(mot for termes_doc in termes for mot in termes_doc)
    moyenne = sum(sum(termes_doc.values()) for termes_doc in termes) / len(termes) or 1
    scores = []
    for doc, termes_doc in zip(documents, termes):
        longueur = sum(termes_doc.values())
        titre = set(mots("\n".join(doc.page_content.splitlines()[:5])))
        score = sum(
            math.log(
                1 + (len(documents) - frequences[mot] + 0.5) / (frequences[mot] + 0.5)
            )
            * (
                (termes_doc[mot] * 2.5)
                / (termes_doc[mot] + 1.5 * (0.25 + 0.75 * longueur / moyenne))
                + (mot in titre)
            )
            for mot in requete
            if termes_doc[mot]
        )
        scores.append(score)
    lexicaux = [
        documents[i]
        for i in sorted(range(len(documents)), key=lambda i: -scores[i])[:3]
        if scores[i] > 0
    ]

    # Conserver les étapes qui continuent sur les deux pages suivantes.
    resultat, pages = [], set()
    for ancre in lexicaux + semantiques:
        source, page = ancre.metadata["source"], ancre.metadata["page"]
        # Une page commençant à l'étape 4 ne doit pas perdre les étapes précédentes.
        texte_page = pages_indexees[(source, page)].page_content
        lignes = [ligne.strip() for ligne in texte_page.splitlines() if ligne.strip()]
        if lignes and lignes[0] == str(page):
            lignes = lignes[1:]
        debut = page - 1 if lignes and re.match(r"^[2-9]\d*[.)]", lignes[0]) else page
        for numero in range(max(1, debut), debut + 3):
            if (source, numero) in pages:
                continue
            passages = [
                doc.page_content
                for doc in documents
                if doc.metadata["source"] == source and doc.metadata["page"] == numero
            ]
            if passages:
                pages.add((source, numero))
                resultat.append(
                    Document(
                        page_content="\n".join(passages)[:3000],
                        metadata={"source": source, "page": numero, "type": "texte"},
                    )
                )
            if len(resultat) == 6:
                return resultat
    return resultat


def figures_des_pages(figures, documents):
    resultat = []
    for doc in documents:
        contenu = figures.documents(
            criteres={"source": doc.metadata["source"], "page": doc.metadata["page"]}
        )
        # Écarter les petites icônes et préférer la figure principale de la page.
        contenu = [
            figure
            for figure in contenu
            if "bbox" not in figure.metadata
            or min(
                figure.metadata["bbox"][2] - figure.metadata["bbox"][0],
                figure.metadata["bbox"][3] - figure.metadata["bbox"][1],
            )
            >= 40
        ]
        if contenu:
            resultat.append(
                max(
                    contenu,
                    key=lambda figure: (
                        (
                            (figure.metadata["bbox"][2] - figure.metadata["bbox"][0])
                            * (figure.metadata["bbox"][3] - figure.metadata["bbox"][1])
                        )
                        if "bbox" in figure.metadata
                        else 0
                    ),
                )
            )
        if len(resultat) == 2:
            break
    return resultat
