"""Feladatkártya #5: a CI lefuttatja a kepek.ipynb-t, és ellenőrzi az eredményeket.

A d) rész a képosztályozási teszt: a CNN pontossága a 600 képes teszthalmazon legalább 0,93.
"""
import json
import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def e():
    path = ROOT / "kepek.ipynb"
    if not path.exists():
        pytest.fail("Hiányzik a kepek.ipynb (Colabból ugyanezen a néven mentsd vissza).", pytrace=False)
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    except CellExecutionError as hiba:
        uzenet = re.sub(r"\x1b\[[0-9;]*m", "", str(hiba)).strip().splitlines()[-1]
        pytest.fail("A notebook hibával állt le: " + uzenet, pytrace=False)
    return json.load(open(ROOT / "eredmenyek" / "kartya5.json"))


def test_a_konvolucio(e):
    assert e["konv"] == [[3.0, 3.0, 0.0]] * 3


def test_c_cnn_szerkezete(e):
    assert e["cnn_param"] == 55140, "A rétegek a táblázat szerint: 55 140 paraméter."


def test_d_keposztalyozasi_teszt(e):
    assert e["cnn_pont"] >= 0.93, f"A CNN pontossága {e['cnn_pont']:.3f}, legalább 0,93 kell."


def test_d_cnn_jobb_mint_az_mlp(e):
    assert e["cnn_pont"] > e["mlp_pont"] + 0.1


def test_e_yolo(e):
    assert "bus" in e["yolo_nevek"], "A YOLO-nak a buszt is meg kell találnia."
    assert e["emberek"] == 3
