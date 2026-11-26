"""Interface Streamlit locale du SAV : chaque question lance un nouveau parcours."""

import asyncio
from io import BytesIO

import streamlit as st
from PIL import Image

from config import DATA, trouver_photo
from fichiers import (
    enregistrer_fichier,
    figures_reponse,
    indexer_manuels,
    message_erreur,
)
from orchestrateur import construire_graphe

ETAPES = {
    "orchestrateur": "Orchestrateur",
    "agent_documentaire": "Agent documentaire",
    "agent_garantie": "Agent garantie",
}


async def interroger(question, photo_reference, statut):
    # Valider la photo avant tout appel au modèle, comme dans le terminal.
    if photo_reference:
        trouver_photo(photo_reference)
    graphe = construire_graphe()
    etat = {"question": question, "photo_reference": photo_reference}
    reponse, parcours = "", []
    async for mise_a_jour in graphe.astream(etat, stream_mode="updates"):
        for nom, valeurs in mise_a_jour.items():
            parcours.append(ETAPES[nom])
            statut.write(f"{ETAPES[nom]} : terminé")
            if valeurs.get("reponse"):
                reponse = valeurs["reponse"]
    if not reponse:
        raise RuntimeError(
            "Le parcours s'est terminé sans réponse. Merci de réessayer."
        )
    parcours.append("END")
    return reponse, " → ".join(parcours)


def afficher_message(message):
    with st.chat_message(message["role"]):
        if message.get("erreur"):
            st.error(message["content"])
            with st.expander("Détails de l'erreur"):
                st.code(message["erreur"], language=None)
        else:
            # Les références fichier/page/figure sont conservées telles quelles.
            st.markdown(message["content"])
        for figure in message.get("figures", []):
            with Image.open(BytesIO(figure["contenu"])) as image:
                largeur = min(image.width, 680)
            st.image(
                figure["contenu"],
                caption=figure["legende"],
                width=largeur,
                output_format="PNG",
            )
        if message.get("photo"):
            st.caption(f"Photo jointe : {message['photo']}")
        if message.get("parcours"):
            st.caption(message["parcours"])


def main():
    st.set_page_config(page_title="MAINTiDOC", page_icon="🔧", layout="centered")
    st.session_state.setdefault("messages", [])

    with st.sidebar:
        st.header("Votre appareil")
        fichier = st.file_uploader(
            "Ajouter un PDF ou une photo",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            max_upload_size=50,
            help="PDF : manuel à indexer. Image : photo pour votre prochaine question.",
        )
        if fichier and st.button("Importer le fichier", key="importer"):
            try:
                chemin = enregistrer_fichier(fichier.name, fichier.getvalue())
            except ValueError as erreur:
                st.error(str(erreur))
            else:
                if chemin.suffix == ".pdf":
                    st.session_state["import_pdf"] = chemin.name
                else:
                    st.session_state["photo_selection"] = chemin.name
                st.success(f"Fichier ajouté : {chemin.name}")
        if st.session_state.get("import_pdf"):
            st.info(
                f"PDF ajouté : {st.session_state['import_pdf']}. Lancez l'indexation ci-dessous."
            )
        if st.button("Indexer les PDF", key="indexer"):
            with st.spinner(
                "Indexation des PDF… Si un traitement est déjà en cours, attente de son résultat."
            ):
                try:
                    resultat = asyncio.run(indexer_manuels())
                except Exception as erreur:
                    st.error(f"Indexation échouée : {message_erreur(erreur)}")
                else:
                    st.session_state.pop("import_pdf", None)
                    st.success(resultat)
        st.caption(
            "Les PDF sont analysés avec LlamaIndex : texte, tableaux et descriptions visuelles. "
            "Une photo sert à retrouver des figures similaires avec CLIP. "
            "Un clic supplémentaire attend l'indexation existante."
        )
        st.divider()
        photos = sorted(
            photo.name
            for photo in (DATA / "photos").glob("*")
            if photo.is_file()
            and photo.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        photo = st.selectbox(
            "Photo pour la prochaine question",
            ["", *photos],
            format_func=lambda valeur: valeur or "Sans photo",
            help="Les photos doivent être placées dans data/photos/.",
            key="photo_selection",
        )
        if photo:
            st.image(str(trouver_photo(photo)), caption=photo)
        else:
            st.caption(
                "Facultatif : ajoutez une photo dans data/photos/ pour la choisir ici."
            )
        st.divider()
        st.caption(
            "Chaque question est indépendante. Précisez le modèle de l'appareil "
            "ou son numéro de série à chaque demande."
        )
        if st.button("Effacer l'historique", key="effacer", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("MAINTiDOC — Votre assistant de maintenance")
    st.write("Un problème avec votre appareil ? Trouvons la bonne réponse.")
    if not st.session_state.messages:
        documentation, garantie = st.columns(2)
        with documentation, st.container(border=True):
            st.subheader("Dépannage")
            st.write("Consultez vos manuels, avec les pages et figures utilisées.")
            st.caption(
                "Exemple : Comment retirer un bourrage papier sur une Epson ET-2850 ?"
            )
        with garantie, st.container(border=True):
            st.subheader("Garantie")
            st.write(
                "Vérifiez la couverture de votre appareil avec son numéro de série."
            )
            st.caption("Exemple : Mon imprimante PX200-0042 est-elle sous garantie ?")

    for message in st.session_state.messages:
        afficher_message(message)

    question = st.chat_input("Décrivez votre problème ou posez votre question…")
    if not question or not question.strip():
        return
    message = {"role": "user", "content": question.strip(), "photo": photo}
    st.session_state.messages.append(message)
    afficher_message(message)

    with st.status("Recherche d'une réponse…", expanded=True) as statut:
        try:
            reponse, parcours = asyncio.run(interroger(question.strip(), photo, statut))
        except Exception as erreur:
            statut.update(label="La demande a échoué", state="error", expanded=False)
            message = {
                "role": "assistant",
                "content": "Impossible de répondre. Vérifiez qu'Ollama est démarré et, "
                "pour les manuels, que le serveur MCP est lancé et l'ingestion effectuée. "
                "Vous pouvez ensuite poser à nouveau votre question.",
                "erreur": str(erreur),
            }
        else:
            statut.update(label="Réponse prête", state="complete", expanded=False)
            message = {
                "role": "assistant",
                "content": reponse,
                "parcours": parcours,
                "figures": figures_reponse(reponse),
            }
    st.session_state.messages.append(message)
    afficher_message(message)


if __name__ == "__main__":
    main()
