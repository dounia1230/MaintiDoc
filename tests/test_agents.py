import asyncio
from types import SimpleNamespace

import pytest

import agents


@pytest.mark.parametrize("photo", ["", "img_px200_01"])
def test_photo_transmise_sans_etre_inventee(monkeypatch, photo):
    arguments = []

    class OutilDistant:
        name = "rag_multimodal"

        async def ainvoke(self, params):
            arguments.append(params)
            return "Réponse sourcée"

    class ClientSimule:
        def __init__(self, configuration):
            pass

        async def get_tools(self):
            return [OutilDistant()]

    monkeypatch.setattr(agents, "MultiServerMCPClient", ClientSimule)
    monkeypatch.setattr(agents, "modele", lambda: None)
    monkeypatch.setattr(
        agents, "create_agent", lambda **params: SimpleNamespace(**params)
    )

    async def executer():
        agent = await agents.creer_agent_documentaire(photo)
        assert set(agent.tools[0].args) == {"question"}
        return await agent.tools[0].ainvoke({"question": "Que signifie E02 ?"})

    assert asyncio.run(executer()) == "Réponse sourcée"
    assert arguments == [{"question": "Que signifie E02 ?", "photo_reference": photo}]
