import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from benchmark.benchmark import liste_nnz, operations, run_benchmark, rapport, tracer
from benchmark.__main__ import main


def test_liste_nnz():
    liste = liste_nnz(1000)
    assert liste[-1] == 1000
    assert liste == sorted(liste)


def test_operations_coherentes():
    """Les quatre implémentations donnent le même produit matrice-vecteur."""
    ops = operations(30, 100)
    assert set(ops) == {"coo_pure", "coo_optim", "numpy", "scipy"}
    ref = ops["numpy"]()
    for nom, op in ops.items():
        assert np.allclose(op(), ref), nom


def test_sans_pure():
    assert "coo_pure" not in operations(10, 5, avec_pure=False)


def test_run_benchmark():
    res = run_benchmark(50)
    for cle in ("nnz", "dim"):
        for temps in res[cle]["temps"].values():
            assert len(temps) == len(res[cle]["x"])
    assert res["nnz"]["x"] == [50]
    assert all(t > 0 for t in res["nnz"]["temps"]["scipy"])


def test_rapport():
    texte = rapport(run_benchmark(50, avec_pure=False))
    assert "coo_optim" in texte and "scipy" in texte
    assert "coo_pure" not in texte


def test_tracer(tmp_path):
    sortie = tmp_path / "bench.png"
    tracer(run_benchmark(50, avec_pure=False), str(sortie))
    assert sortie.stat().st_size > 0


def test_cli(tmp_path, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    rapport_txt, png = tmp_path / "r.txt", tmp_path / "g.png"
    main(["-s", str(rapport_txt), "-pN", "50", "-o", str(png), "--noCOOPure"])
    assert "scipy" in rapport_txt.read_text(encoding="utf-8")
    assert png.exists()


def test_cli_N_invalide():
    with pytest.raises(SystemExit):
        main(["-N", "0"])
