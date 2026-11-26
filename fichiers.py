"""Importer les documents et retrouver uniquement les figures locales citées."""

import hashlib
import re
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import httpx
import pymupdf
from langchain_mcp_adapters.client import MultiServerMCPClient
from PIL import Image

from config import DATA, MCP_URL


def message_erreur(erreur):
    """Afficher la cause utile, y compris dans les groupes d'erreurs asynchrones."""
    if isinstance(erreur, BaseExceptionGroup):
        messages = dict.fromkeys(message_erreur(cause) for cause in erreur.exceptions)
        return "\n".join(messages)
    if isinstance(erreur, (httpx.TimeoutException, TimeoutError)):
        return (
            "Le délai d'attente de la connexion MCP est dépassé. L'indexation peut "
            "continuer côté serveur : attendez sa fin avant de la relancer. "
            "Redémarrez Streamlit pour charger le délai d'attente de 30 minutes."
        )
    if isinstance(erreur, httpx.ConnectError):
        return (
            f"Serveur MCP injoignable à {MCP_URL}. "
            "Démarrez-le avec : uv run mcp_server.py"
        )
    return str(erreur) or type(erreur).__name__


def enregistrer_fichier(nom, contenu):
    # Le nom reçu du navigateur ne doit jamais choisir un chemin sur le disque.
    nom = nom.replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(nom).suffix.lower()
    if not contenu or len(contenu) > 50 * 1024 * 1024:
        raise ValueError(
            "Le fichier doit contenir des données et ne pas dépasser 50 Mo."
        )
    try:
        if extension == ".pdf":
            with pymupdf.open(stream=contenu, filetype="pdf") as pdf:
                if pdf.needs_pass or not len(pdf):
                    raise ValueError("PDF protégé ou sans page.")
            dossier = "manuels"
        elif extension in {".png", ".jpg", ".jpeg", ".webp"}:
            with Image.open(BytesIO(contenu)) as image:
                attendu = {
                    ".png": "PNG",
                    ".jpg": "JPEG",
                    ".jpeg": "JPEG",
                    ".webp": "WEBP",
                }
                if image.format != attendu[extension]:
                    raise ValueError(
                        "Le contenu ne correspond pas à l'extension de l'image."
                    )
                image.verify()
            dossier = "photos"
        else:
            raise ValueError("Formats acceptés : PDF, PNG, JPEG et WebP.")
    except Exception as erreur:
        raise ValueError(f"Fichier invalide : {erreur}") from erreur
    radical = re.sub(r"[^\w-]", "_", Path(nom).stem)[:80] or "document"
    empreinte = hashlib.sha256(contenu).hexdigest()[:12]
    chemin = DATA / dossier / f"{radical}_{empreinte}{extension}"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    # Le contenu détermine le nom : deux documents différents ne s'écrasent pas.
    try:
        with chemin.open("xb") as fichier:
            fichier.write(contenu)
    except FileExistsError:
        if chemin.read_bytes() != contenu:
            raise ValueError("Un fichier différent porte déjà ce nom.") from None
    return chemin


async def indexer_manuels():
    # L'index reste géré par le même processus que les recherches documentaires.
    client = MultiServerMCPClient(
        {
            "documentation": {
                "transport": "streamable_http",
                "url": MCP_URL,
                "sse_read_timeout": 1800,
            }
        }
    )
    try:
        async with client.session("documentation") as session:
            resultat = await session.call_tool(
                "indexer_manuels", {}, read_timeout_seconds=timedelta(minutes=30)
            )
    except Exception as erreur:
        raise RuntimeError(message_erreur(erreur)) from erreur
    texte = "\n".join(bloc.text for bloc in resultat.content if bloc.type == "text")
    if resultat.isError:
        raise RuntimeError(texte)
    return texte


def figures_reponse(reponse):
    # Ce pied de réponse est ajouté par MCP, après la génération du texte.
    texte, separateur, references = reponse.rpartition("\n\nSources consultées :\n")
    if not separateur:
        return []
    pages = list(
        dict.fromkeys(
            re.findall(
                r"^- (.+), page (\d+)(?:, figure : .+)?$",
                references,
                flags=re.MULTILINE,
            )
        )
    )
    # Une page consultée ne constitue pas forcément une source de la réponse finale.
    source_unique = len({source for source, _ in pages}) == 1
    citees = [
        (source, page)
        for source, page in pages
        if re.search(re.escape(source) + r"[`*\s]*,\s*page\s+" + page + r"\b", texte)
        or (
            source_unique
            and re.search(r"\bpages?\s+" + page + r"\b", texte, flags=re.IGNORECASE)
        )
    ]
    retenues = set(citees or pages)
    dossier = (DATA / "figures").resolve()
    figures, deja_vues = [], set()
    for source, page, reference in re.findall(
        r"^- (.+), page (\d+), figure : (.+)$", references, flags=re.MULTILINE
    ):
        if (source, page) not in retenues:
            continue
        chemin = (DATA / reference.strip()).resolve()
        if (
            not chemin.is_relative_to(dossier)
            or not chemin.is_file()
            or chemin.suffix.lower() != ".png"
            or chemin in deja_vues
        ):
            continue
        try:
            with Image.open(chemin) as image:
                if min(image.size) < 80:
                    continue
                image.verify()
        except (OSError, ValueError):
            continue
        deja_vues.add(chemin)
        figures.append(
            {
                # Garder les pixels dans l'historique, même après une réindexation.
                "contenu": chemin.read_bytes(),
                "legende": f"{source}, page {page} — figure : {reference}",
            }
        )
        if len(figures) == 2:
            break
    if figures:
        return figures

    # Les dessins vectoriels ne sont pas des images intégrées : montrer la page.
    dossier_manuels = (DATA / "manuels").resolve()
    for source, page in citees or pages:
        chemin = (dossier_manuels / source).resolve()
        cle = (chemin, int(page))
        if (
            not chemin.is_relative_to(dossier_manuels)
            or chemin.suffix.lower() != ".pdf"
            or cle in deja_vues
        ):
            continue
        try:
            with pymupdf.open(chemin) as pdf:
                if not 1 <= int(page) <= len(pdf):
                    continue
                pixels = pdf[int(page) - 1].get_pixmap(dpi=100, alpha=False)
                contenu = pixels.tobytes("png")
        except (OSError, RuntimeError, ValueError):
            continue  # Une source indisponible n'empêche pas de lire la réponse.
        deja_vues.add(cle)
        figures.append(
            {"contenu": contenu, "legende": f"{source}, page {page} — aperçu du manuel"}
        )
        if len(figures) == 2:
            break
    return figures
