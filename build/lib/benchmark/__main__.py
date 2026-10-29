"""Ligne de commande ``benchCOO`` (équivalent : ``python -m benchmark``)."""

import argparse

import matplotlib.pyplot as plt

from .benchmark import run_benchmark, rapport, tracer


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="benchCOO",
        description="Benchmark du produit matrice-vecteur : MatCooPure, MatCooOptim, numpy, scipy.")
    parser.add_argument("-s", "--save", metavar="FICHIER",
                        help="enregistre le rapport texte dans FICHIER")
    parser.add_argument("-N", type=int, default=1000,
                        help="nombre maximal d'éléments non nuls (défaut : 1000)")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="trace le graphe, l'enregistre en png et l'affiche")
    parser.add_argument("-o", "--output", default="benchCOO.png", metavar="PNG",
                        help="fichier png du graphe (défaut : benchCOO.png)")
    parser.add_argument("--noCOOPure", action="store_true",
                        help="exclut MatCooPure du benchmark (lent)")
    args = parser.parse_args(argv)
    if args.N < 1:
        parser.error("N doit être supérieur ou égal à 1")

    res = run_benchmark(args.N, avec_pure=not args.noCOOPure)
    texte = rapport(res)
    print(texte)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(texte)
    if args.plot:
        tracer(res, args.output)
        plt.show()


if __name__ == "__main__":
    main()
