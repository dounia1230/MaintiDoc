import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import orchestrateur


@pytest.mark.parametrize("categorie", ["garantie", "documentation"])
def test_un_seul_specialiste_puis_retour(monkeypatch, categorie):
    appels = []

    class RouteurSimule:
        def with_structured_output(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            appels.append("routage")
            return SimpleNamespace(categorie=categorie)

    class AgentSimule:
        async def ainvoke(self, etat, **kwargs):
            appels.append(etat["messages"][0][1])
            return {"messages": [AIMessage(content="Réponse du spécialiste")]}

    async def documentaire(photo_reference):
        assert photo_reference == "photo_test"
        return AgentSimule()

    monkeypatch.setattr(orchestrateur, "modele", RouteurSimule)
    monkeypatch.setattr(orchestrateur, "creer_agent_garantie", AgentSimule)
    monkeypatch.setattr(orchestrateur, "creer_agent_documentaire", documentaire)

    async def executer():
        graphe = orchestrateur.construire_graphe()
        return [
            maj
            async for maj in graphe.astream(
                {
                    "question": "Question complète",
                    "photo_reference": "photo_test",
                },
                stream_mode="updates",
            )
        ]

    mises_a_jour = asyncio.run(executer())
    assert [next(iter(maj)) for maj in mises_a_jour] == [
        "orchestrateur",
        f"agent_{'documentaire' if categorie == 'documentation' else 'garantie'}",
        "orchestrateur",
    ]
    assert mises_a_jour[-1]["orchestrateur"]["destination"] == "fin"
    assert appels.count("routage") == 1
    assert len(appels) == 2
    assert ("photo_reference : photo_test" in appels[1]) == (
        categorie == "documentation"
    )
