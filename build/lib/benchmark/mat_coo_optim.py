import numpy as np
from typing import List, Tuple
from pprint import PrettyPrinter


def _triplets(M):
    """Renvoie (forme, row, col, val) d'une matrice COO : MatCooOptim, MatCooPure ou scipy.sparse.coo_matrix.
    (Duck typing plutôt qu'isinstance : plus robuste avec les imports de notebook.)"""
    if hasattr(M, 'val'):                                   # MatCooOptim
        return (M.n, M.m), M.row, M.col, M.val
    if hasattr(M, 'row'):                                   # scipy.sparse.coo_matrix
        return M.shape, M.row.astype(np.int64), M.col.astype(np.int64), M.data
    data = np.array(M.data).reshape(-1, 3)                  # MatCooPure : liste de triplets
    return (M.n, M.m), data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2]


def _fusionner(row, col, val, m):
    """Additionne les doublons (même case). Renvoie row, col, val triés par (row, col)."""
    cle = row * m + col                                     # une clé entière par case
    uniq, inv = np.unique(cle, return_inverse=True)
    return uniq // m, uniq % m, np.bincount(inv, weights=val)


class MatCooOptim():
    """
    COO matrix stored in numpy arrays (same interface as MatCooPure).

    Attributes:
    ----------
    n : int
        The number of rows.
    m : int
        The number of columns.
    row, col : numpy.ndarray (int64)
        Row and column index of each stored element.
    val : numpy.ndarray
        Value of each stored element.
    data : list of tuple (read only)
        The elements as a list of (i, j, value), as in MatCooPure.
    nnz : int
        The number of stored elements (read only).
    """

    def __init__(self, n: int, m: int, data: List[Tuple[int, int, float]] = None):
        """Initialize from a list of (i, j, value) triplets (or nothing for an empty matrix)."""
        self.n = n
        self.m = m
        if data is None or len(data) == 0:
            self.row = np.zeros(0, dtype=np.int64)
            self.col = np.zeros(0, dtype=np.int64)
            self.val = np.zeros(0)
        else:
            row, col, val = zip(*data)
            self.row = np.array(row, dtype=np.int64)
            self.col = np.array(col, dtype=np.int64)
            self.val = np.array(val)

    @classmethod
    def from_arrays(cls, n: int, m: int, row: np.ndarray, col: np.ndarray, val: np.ndarray) -> 'MatCooOptim':
        """Create a matrix directly from the three arrays (no conversion)."""
        mat = cls(n, m)
        mat.row, mat.col, mat.val = row, col, val
        return mat

    @property
    def data(self) -> List[Tuple[int, int, float]]:
        return list(zip(self.row.tolist(), self.col.tolist(), self.val.tolist()))

    @property
    def nnz(self) -> int:
        """The number of stored elements."""
        return len(self.val)

    def fill_random(self, nnz: int, seed: int = None):
        """Fill with nnz random elements, sorted by (row, col). Not the same values as MatCooPure for a given seed."""
        rng = np.random.default_rng(seed)
        row = rng.integers(0, self.n, nnz)
        col = rng.integers(0, self.m, nnz)
        val = rng.uniform(1e-16, 1, nnz)
        ordre = np.lexsort((val, col, row))                 # tri par row, puis col, puis val
        self.row, self.col, self.val = row[ordre], col[ordre], val[ordre]

    @classmethod
    def random(cls, n: int, m: int, nnz: int, seed: int = None) -> 'MatCooOptim':
        """Create a random COO matrix."""
        mat = cls(n, m)
        mat.fill_random(nnz, seed)
        return mat

    @classmethod
    def identity(cls, n: int) -> 'MatCooOptim':
        """Create the n x n identity matrix."""
        i = np.arange(n)
        return cls.from_arrays(n, n, i, i.copy(), np.ones(n))

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> 'MatCooOptim':
        """
        Create a COO matrix from a NumPy array.

        Examples
        >>> A = np.array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> print(MatCooOptim.from_numpy(A))
        COO matrix ((3, 3)) with nnz=5
        data:
        (0, 0, 1)
        (0, 1, 2)
        (1, 2, 3)
        (2, 0, 4)
        (2, 2, 5)
        <BLANKLINE>
        """
        row, col = data.nonzero()
        return cls.from_arrays(*data.shape, row, col, data[row, col])

    def to_numpy(self) -> np.ndarray:
        """Convert to a dense NumPy array (duplicates are summed)."""
        arr = np.zeros((self.n, self.m))
        np.add.at(arr, (self.row, self.col), self.val)
        return arr

    def __str__(self) -> str:
        lignes = [f"COO matrix ({(self.n, self.m)}) with nnz={self.nnz}", "data:"]
        lignes += [f"({i}, {j}, {v})" for i, j, v in self.data]
        return "\n".join(lignes) + "\n"

    def _repr_pretty_(self, p: 'PrettyPrinter', cycle: bool) -> None:
        """Pretty display in IPython (first and last 5 elements)."""
        def ligne(k):
            return f"{self.row[k]:3d} {self.col[k]:3d}   {self.val[k]:.2e}\n"
        p.text(f"COO matrix ({self.n}, {self.m}) with nnz={self.nnz}\n")
        p.text("row col    value\n")
        if self.nnz < 10:
            for k in range(self.nnz):
                p.text(ligne(k))
        else:
            for k in range(5):
                p.text(ligne(k))
            p.text("...\n")
            for k in range(self.nnz - 5, self.nnz):
                p.text(ligne(k))

    def mat_vec_product(self, vec: np.ndarray) -> np.ndarray:
        """
        Product of the matrix and a vector, without Python loop.

        Examples:
        --------
        >>> I4 = MatCooOptim.identity(4)
        >>> I4.mat_vec_product(np.arange(4))
        array([0., 1., 2., 3.])
        """
        vec = np.asarray(vec)                               # pas de copie si c'est déjà un ndarray
        if self.m != vec.size:
            raise ValueError(f"Number of columns m={self.m} does not match size of vector {vec.size}")
        # chaque élément apporte val * vec[col] à la ligne row ; bincount somme par ligne
        return np.bincount(self.row, weights=self.val * vec[self.col], minlength=self.n)

    def dot(self, vec: np.ndarray) -> np.ndarray:
        """Alias for mat_vec_product."""
        return self.mat_vec_product(vec)

    def mat_mat_product(self, other: 'MatCooOptim') -> 'MatCooOptim':
        """
        Product of two COO matrices, without Python loop.

        Examples:
        --------
        >>> data1 = [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)]
        >>> mat1 = MatCooOptim(3, 3, data1)
        >>> mat2 = MatCooOptim(3, 3, data1)
        >>> mat1.mat_mat_product(mat2).data
        [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 24.0), (2, 1, 8.0), (2, 2, 25.0)]
        """
        (n2, m2), row2, col2, val2 = _triplets(other)
        if self.m != n2:
            raise ValueError(f"Number of columns in first matrix m={self.m} does not match number of rows in second matrix n={n2}")
        # On trie B par ligne : les éléments de la ligne j de B sont alors contigus.
        ordre = np.argsort(row2, kind='stable')
        row2, col2, val2 = row2[ordre], col2[ordre], val2[ordre]
        # Pour chaque élément (i, j, a) de A : ses partenaires dans B sont row2[debut : debut + nb]
        debut = np.searchsorted(row2, self.col, side='left')
        nb = np.searchsorted(row2, self.col, side='right') - debut
        # Un indice par produit a*b : ia = élément de A, ib = élément de B
        ia = np.repeat(np.arange(self.nnz), nb)
        rang = np.arange(nb.sum()) - np.repeat(np.cumsum(nb) - nb, nb)   # 0, 1, 2... dans chaque groupe
        ib = np.repeat(debut, nb) + rang
        # contribution a*b à la case (row de a, col de b) ; on additionne les doublons
        row, col, val = _fusionner(self.row[ia], col2[ib], self.val[ia] * val2[ib], m2)
        return MatCooOptim.from_arrays(self.n, m2, row, col, val)

    def __mul__(self, vec: np.ndarray) -> np.ndarray:
        """The * operator: product with a vector."""
        if isinstance(vec, (np.ndarray, list, tuple)):
            return self.mat_vec_product(vec)
        raise TypeError(f"Unsupported operand type for *: {type(vec)}")

    def __matmul__(self, other: 'MatCooOptim') -> 'MatCooOptim':
        """
        The @ operator: matrix-matrix product.

        Examples:
        --------
        >>> data1 = [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)]
        >>> mat1 = MatCooOptim(3, 3, data1)
        >>> result = mat1 @ mat1
        >>> print(result)
        COO matrix ((3, 3)) with nnz=5
        data:
        (0, 0, 1.0)
        (0, 1, 2.0)
        (2, 0, 24.0)
        (2, 1, 8.0)
        (2, 2, 25.0)
        <BLANKLINE>
        >>> print(np.allclose(result.to_numpy(), mat1.to_numpy() @ mat1.to_numpy()))
        True
        """
        return self.mat_mat_product(other)

    def __neg__(self) -> 'MatCooOptim':
        """Negation of the matrix."""
        return MatCooOptim.from_arrays(self.n, self.m, self.row, self.col, -self.val)

    def __sub__(self, other: 'MatCooOptim') -> 'MatCooOptim':
        """Subtraction of another matrix."""
        return self + (-other)

    def __add__(self, other):
        """
        Term by term addition. Values with an absolute value below 1e-15 are dropped.

        Examples
        --------
        >>> mat1 = MatCooOptim(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])
        >>> mat2 = MatCooOptim(2, 2, [(0, 1, 1.0), (1, 1, -2.0)])
        >>> print(mat1 + mat2)
        COO matrix ((2, 2)) with nnz=2
        data:
        (0, 0, 1.0)
        (0, 1, 1.0)
        <BLANKLINE>
        """
        forme, row2, col2, val2 = _triplets(other)
        if forme != (self.n, self.m):
            raise ValueError(f"Shapes {(self.n, self.m)} and {forme} do not match")
        row, col, val = _fusionner(np.concatenate((self.row, row2)),
                                   np.concatenate((self.col, col2)),
                                   np.concatenate((self.val, val2)), self.m)
        garde = np.abs(val) > 1e-15
        return MatCooOptim.from_arrays(self.n, self.m, row[garde], col[garde], val[garde])

    def __eq__(self, other) -> bool:
        """
        Equality up to 1e-15, whatever the order of the elements (duplicates are summed).
        Also works with MatCooPure and scipy.sparse.coo_matrix.

        Example:
        --------
        >>> print(MatCooOptim(2, 2, [(0, 0, 1.0), (1, 1, 2.0)]) == MatCooOptim(2, 2, [(1, 1, 2.0), (0, 0, 1.0)]))
        True
        """
        try:
            f1, r1, c1, v1 = _triplets(self)
            f2, r2, c2, v2 = _triplets(other)
        except AttributeError:
            return NotImplemented
        if f1 != f2:
            return False
        # égalité <=> la différence des deux matrices est nulle
        _, _, diff = _fusionner(np.concatenate((r1, r2)), np.concatenate((c1, c2)),
                                np.concatenate((v1, -v2)), f1[1])
        return bool(np.all(np.abs(diff) < 1e-15))

    def dev_show(self):
        """
        Display a summary of the matrix and its memory footprint (the three arrays).

        Example:
        --------
        >>> mat = MatCooOptim(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])
        >>> mat.dev_show()
        COO matrix ((2, 2)) with nnz=2
          0   0   1.0
          1   1   2.0
        ...
          0   0   1.0
          1   1   2.0
        <BLANKLINE>
        Memory footprint: 48 bytes.
        """
        print(f"COO matrix ({(self.n, self.m)}) with nnz={self.nnz}")
        for k in range(min(2, self.nnz)):
            print(f"  {self.row[k]}   {self.col[k]}   {self.val[k]}")
        print("...")
        for k in range(max(0, self.nnz - 2), self.nnz):
            print(f"  {self.row[k]}   {self.col[k]}   {self.val[k]}")
        print()
        print(f"Memory footprint: {self.row.nbytes + self.col.nbytes + self.val.nbytes} bytes.")
