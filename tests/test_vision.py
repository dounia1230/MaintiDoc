"""Vérifier les pixels envoyés au modèle de vision et le cache des descriptions."""

import base64
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

import vision


def test_description_utilise_les_pixels_et_le_cache(monkeypatch, tmp_path):
    chemin = tmp_path / "figure.png"
    Image.new("RGB", (12, 12), "red").save(chemin)
    monkeypatch.setattr(vision, "DATA", tmp_path)
    appels = []

    class ModeleSimule:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            contenu = messages[1].content[1]["image_url"]["url"].split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(contenu))) as image:
                appels.append(image.getpixel((0, 0)))
            return SimpleNamespace(text="Un carré rouge.")

    monkeypatch.setattr(vision, "ChatOllama", ModeleSimule)
    assert vision.decrire_image(chemin) == "Un carré rouge."
    assert vision.decrire_image(chemin) == "Un carré rouge."
    assert appels == [(255, 0, 0)]
    Image.new("RGB", (12, 12), "blue").save(chemin)
    vision.decrire_image(chemin)
    assert appels[-1] == (0, 0, 255) and len(appels) == 2
