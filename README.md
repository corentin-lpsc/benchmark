Exemple d’utilisation de la bibliothèque ne CLI
===============================================

$ benchCOO -s results.txt -N 1000 # sauvegarde les benchmarks dans un fichier text pour 50, 100, 200 et 1000 éléments non nuls pour des dimensions de matrices définies par défaut

$ benchCOO -pN 20000 --noCOOPure # plot les résultats du benchamrks jusqu'a 20000 nnz et ne sauvegarde pas les résultats

Détail des arguments possibles :
s : enregistre le rapport du benchmark
N : nombre max de nnz, avec les paliers 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, tronqués à N, et N en dernier point
p : trace, affiche et sauvegarde un jpg des graphes
o : crée un fichier png des graphe
noCOOPure : exclue matCOOPure du benchmark

Installation : 

$ cd chemin/des/sources

$ pip install .
