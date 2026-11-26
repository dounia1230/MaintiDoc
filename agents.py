"""Agent garantie avec outil local et agent documentaire avec outil MCP."""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_URL, modele
from garantie import consulter_garantie


def creer_agent_garantie():
    return create_agent(
        model=modele(),
        tools=[consulter_garantie],
        system_prompt=(
            "Tu es l'agent garantie du SAV. Réponds en français. "
            "Si le numéro de série manque, demande-le sans inventer de numéro. "
            "Sinon, appelle obligatoirement consulter_garantie et utilise exclusivement son résultat. "
            "Donne le statut, les dates et la date de vérification. "
            "Un numéro introuvable ne signifie pas que la garantie a expiré."
        ),
    )


async def creer_agent_documentaire(photo_reference=""):
    client = MultiServerMCPClient(
        {
            "documentation": {"transport": "streamable_http", "url": MCP_URL},
        }
    )
    outils = await client.get_tools()
    outil_distant = next(outil for outil in outils if outil.name == "rag_multimodal")

    @tool(return_direct=True)
    async def rag_multimodal(question: str):
        """Rechercher dans les manuels et analyser la photo fournie, s'il y en a une."""
        # La photo vient de l'utilisateur : le LLM ne choisit jamais sa référence.
        return await outil_distant.ainvoke(
            {
                "question": question,
                "photo_reference": photo_reference,
            }
        )

    return create_agent(
        model=modele(),
        tools=[rag_multimodal],
        system_prompt=(
            "Tu es l'agent documentaire du SAV. Appelle obligatoirement rag_multimodal avec "
            "la question complète. L'outil reçoit déjà la bonne référence photo si elle existe. "
            "La réponse de l'outil sera transmise directement avec ses sources. "
            "N'ajoute aucune connaissance technique personnelle. Si l'outil échoue, indique l'échec."
        ),
    )
