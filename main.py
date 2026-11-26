"""Point d'entrée : une question en argument, ou une boucle interactive."""

import argparse
import asyncio

from config import trouver_photo
from orchestrateur import construire_graphe


async def poser_question(graphe, question, photo_reference=""):
    if photo_reference:
        trouver_photo(
            photo_reference
        )  # Signaler une référence invalide avant d'appeler le LLM.
    etat = {"question": question, "photo_reference": photo_reference}
    reponse = ""
    async for mise_a_jour in graphe.astream(etat, stream_mode="updates"):
        for nom, valeurs in mise_a_jour.items():
            print(f"  -> {nom}", flush=True)
            if valeurs.get("reponse"):
                reponse = valeurs["reponse"]
    print(f"\n{reponse}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Service après-vente : documentation ou garantie"
    )
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--photo", default="", help="Référence d'une photo dans data/photos"
    )
    args = parser.parse_args()
    graphe = construire_graphe()
    if args.question:
        await poser_question(graphe, args.question, args.photo)
        return
    print("SAV — tape exit pour quitter. Chaque question est indépendante.")
    while True:
        question = input("Question : ").strip()
        if question.lower() == "exit":
            break
        if question:
            photo = input("Référence photo (Entrée si aucune) : ").strip()
            await poser_question(graphe, question, photo)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
