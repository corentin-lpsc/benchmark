"""
Benchmark du produit matrice-vecteur : MatCooPure, MatCooOptim, numpy (dense) et scipy.

Deux expériences sont menées :

* ``nnz`` : dimension fixe (la plus grande de :data:`DIMENSIONS`), nombre d'éléments non nuls variable ;
* ``dim`` : taux de remplissage fixe (:data:`DENSITE`), dimension variable.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

from .mat_coo_pure import MatCooPure
from .mat_coo_optim import MatCooOptim
from .utils import timer

DIMENSIONS = [20, 50, 100, 200, 500, 1000, 2000]
NNZ_DEFAUT = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
DENSITE = 0.005  # taux de remplissage de l'expérience "dim"


def liste_nnz(N):
    """
    Liste croissante de nombres d'éléments non nuls, terminée par ``N``.

    Examples
    --------
    >>> liste_nnz(1000)
    [50, 100, 200, 500, 1000]
    >>> liste_nnz(30)
    [30]
    """
    return [k for k in NNZ_DEFAUT if k < N] + [N]


def operations(n, nnz, avec_pure=True):
    """
    Construit une matrice aléatoire n x n avec nnz éléments et renvoie
    un dictionnaire {nom: fonction sans argument} calculant le produit matrice-vecteur.

    Toutes les implémentations utilisent la même matrice et le même vecteur.

    Examples
    --------
    >>> sorted(operations(10, 5, avec_pure=False))
    ['coo_optim', 'numpy', 'scipy']
    """
    optim = MatCooOptim.random(n, n, nnz, seed=0)
    x = np.random.default_rng(0).random(n)
    dense = optim.to_numpy()
    sparse = coo_matrix((optim.val, (optim.row, optim.col)), shape=(n, n))

    ops = {}
    if avec_pure:
        pure = MatCooPure(n, n, optim.data)
        ops["coo_pure"] = lambda: pure.mat_vec_product(x)
    ops["coo_optim"] = lambda: optim.mat_vec_product(x)
    ops["numpy"] = lambda: dense @ x
    ops["scipy"] = lambda: sparse @ x
    return ops


def _mesurer(n, nnz, avec_pure, temps):
    """Ajoute un temps (en s) par implémentation dans le dictionnaire ``temps``."""
    for nom, op in operations(n, nnz, avec_pure).items():
        timer(op, temps[nom])


def run_benchmark(N=1000, avec_pure=True):
    """
    Lance les deux expériences.

    Parameters
    ----------
    N : int
        Nombre maximal d'éléments non nuls.
    avec_pure : bool
        Inclure MatCooPure (lent) ou non.

    Returns
    -------
    dict
        ``{"nnz": {"n": ..., "x": [...], "temps": {nom: [...]}},
        "dim": {"x": [...], "nnz": [...], "temps": {nom: [...]}}}``
        Un temps vaut ``nan`` si la mesure précédente a dépassé 1 s.
    """
    noms = (["coo_pure"] if avec_pure else []) + ["coo_optim", "numpy", "scipy"]

    # Expérience 1 : dimension fixe, nnz variable
    n_fixe = DIMENSIONS[-1]
    nnz_list = liste_nnz(N)
    temps_nnz = {nom: [] for nom in noms}
    for nnz in nnz_list:
        _mesurer(n_fixe, nnz, avec_pure, temps_nnz)

    # Expérience 2 : taux de remplissage fixe, dimension variable (on respecte nnz <= N)
    nnz_de = {n: max(1, round(DENSITE * n * n)) for n in DIMENSIONS}
    dims = [n for n in DIMENSIONS if nnz_de[n] <= N]
    temps_dim = {nom: [] for nom in noms}
    for n in dims:
        _mesurer(n, nnz_de[n], avec_pure, temps_dim)

    return {
        "nnz": {"n": n_fixe, "x": nnz_list, "temps": temps_nnz},
        "dim": {"x": dims, "nnz": [nnz_de[n] for n in dims], "temps": temps_dim},
    }


def _tableau(titre, nom_x, xs, temps):
    noms = list(temps)
    lignes = [titre, " | ".join([f"{nom_x:>8}"] + [f"{nom:>10}" for nom in noms])]
    for k, x in enumerate(xs):
        lignes.append(" | ".join([f"{x:>8}"] + [f"{temps[nom][k]:>10.2e}" for nom in noms]))
    return "\n".join(lignes)


def rapport(res):
    """Rapport textuel des résultats de :func:`run_benchmark` (temps en secondes)."""
    n = res["nnz"]["n"]
    t1 = _tableau(f"Produit matrice-vecteur, matrice {n}x{n}, nnz variable (temps en s)",
                  "nnz", res["nnz"]["x"], res["nnz"]["temps"])
    t2 = _tableau(f"Produit matrice-vecteur, taux de remplissage {DENSITE:.1%}, dimension variable (temps en s)",
                  "n", res["dim"]["x"], res["dim"]["temps"])
    return t1 + "\n\n" + t2 + "\n"


def tracer(res, sortie="benchCOO.png"):
    """Trace les deux graphes côte à côte et les enregistre au format png dans ``sortie``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=100)
    n = res["nnz"]["n"]
    cas = [("nnz", "Nombre d'éléments non nuls", f"Matrice {n}x{n}"),
           ("dim", "Dimension n", f"Taux de remplissage {DENSITE:.1%}")]
    for ax, (cle, xlabel, titre) in zip(axes, cas):
        for nom, temps in res[cle]["temps"].items():
            ax.plot(res[cle]["x"], temps, marker="o", label=nom)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Temps (s)")
        ax.set_title(titre)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(sortie)
    return fig
