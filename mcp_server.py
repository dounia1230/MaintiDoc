"""Publication des outils documentaires MAINTiDOC en MCP Streamable HTTP."""

import asyncio

from mcp.server.fastmcp import FastMCP

from ingestion import main as reconstruire_index
from rag import repondre_documentation

mcp = FastMCP(
    name="maintidoc-documentation",
    host="127.0.0.1",
    port=24001,
    stateless_http=True,
    json_response=True,
)
verrou_index = asyncio.Lock()
tache_indexation = None


async def executer_indexation() -> str:
    async with verrou_index:
        resultat = await asyncio.to_thread(reconstruire_index)
    return (
        f"Indexation terminée : {resultat['passages']} passages et "
        f"{resultat['figures']} figures décrites et {resultat.get('tableaux', 0)} tableaux. "
        "Vous pouvez poser votre question."
    )


@mcp.tool()
async def indexer_manuels() -> str:
    """Indexer les PDF de data/manuels ou attendre l'indexation déjà en cours."""
    global tache_indexation
    # Tous les clients attendent le même calcul, même après une déconnexion.
    if tache_indexation is None or tache_indexation.done():
        tache_indexation = asyncio.create_task(executer_indexation())
    return await asyncio.shield(tache_indexation)


@mcp.tool()
async def rag_multimodal(question: str, photo_reference: str = "") -> str:
    """Dépanner un appareil à partir des manuels et d'une éventuelle photo dans data/photos.

    Transmettre la question complète et la référence exacte de la photo si elle est fournie.
    Renvoie une réponse en français accompagnée des sources (manuel, page, figure).
    """
    async with verrou_index:
        resultat = await repondre_documentation(question, photo_reference)
    references = []
    for source in resultat["sources"]:
        reference = f"- {source['source']}, page {source['page']}"
        if source["type"] == "figure":
            reference += f", figure : {source['image']}"
        references.append(reference)
    if not references:
        return resultat["reponse"]
    return resultat["reponse"] + "\n\nSources consultées :\n" + "\n".join(references)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
