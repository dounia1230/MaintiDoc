"""L'agent spécialisé rend toujours la main à l'orchestrateur LangGraph."""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agents import creer_agent_documentaire, creer_agent_garantie
from config import modele


class Etat(TypedDict, total=False):
    question: str
    photo_reference: str
    destination: str
    reponse: str


class Routage(BaseModel):
    categorie: Literal["documentation", "garantie"]


def construire_graphe():
    routeur = modele().with_structured_output(Routage, method="json_schema")

    async def orchestrateur(etat: Etat):
        # Au retour de l'agent, la réponse est déjà prête : on termine.
        if etat.get("reponse"):
            return {"destination": "fin"}
        decision = await routeur.ainvoke(
            [
                (
                    "system",
                    "Classe la demande du SAV dans une seule catégorie. "
                    "garantie : couverture, durée ou validité de garantie à partir d'un numéro de série. "
                    "documentation : panne, voyant, pictogramme, fonctionnement ou manuel. "
                    "La présence d'un numéro de série ne suffit pas à choisir garantie.",
                ),
                ("human", etat["question"]),
            ]
        )
        return {"destination": decision.categorie}

    async def agent_garantie(etat: Etat):
        agent = creer_agent_garantie()
        resultat = await agent.ainvoke(
            {"messages": [("human", etat["question"])]},
            config={"recursion_limit": 12},
        )
        return {"reponse": resultat["messages"][-1].text}

    async def agent_documentaire(etat: Etat):
        # Le serveur MCP n'est nécessaire que pour la branche documentaire.
        agent = await creer_agent_documentaire(etat.get("photo_reference", ""))
        question = etat["question"]
        if etat.get("photo_reference"):
            question += f"\nphoto_reference : {etat['photo_reference']}"
        resultat = await agent.ainvoke(
            {"messages": [("human", question)]},
            config={"recursion_limit": 12},
        )
        return {"reponse": resultat["messages"][-1].text}

    graphe = StateGraph(Etat)
    graphe.add_node("orchestrateur", orchestrateur)
    graphe.add_node("agent_documentaire", agent_documentaire)
    graphe.add_node("agent_garantie", agent_garantie)
    graphe.add_edge(START, "orchestrateur")
    graphe.add_conditional_edges(
        "orchestrateur",
        lambda etat: etat["destination"],
        {
            "documentation": "agent_documentaire",
            "garantie": "agent_garantie",
            "fin": END,
        },
    )
    graphe.add_edge("agent_documentaire", "orchestrateur")
    graphe.add_edge("agent_garantie", "orchestrateur")
    return graphe.compile()
