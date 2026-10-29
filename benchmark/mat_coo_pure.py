import random
import numpy as np
from IPython.display import display, Markdown
from typing import List, Tuple
from pympler import asizeof  # pip install pympler
from pprint import PrettyPrinter
import sys


def _vers_dict(M):
    if isinstance(M, MatCooPure):
        forme, triplets = (M.n, M.m), M.data
    else:  # scipy.sparse.coo_matrix
        forme, triplets = M.shape, zip(M.row, M.col, M.data)
    d = {}
    for i, j, v in triplets:
        cle = (int(i), int(j))
        d[cle] = d.get(cle, 0.0) + v
    return forme, d

class MatCooPure():
    """
    COO matrix class (see Wikipedia for more information).

    Attributes:
    ----------
    n : int
        The number of rows.
    m : int
        The number of columns.
    data : list of tuple
        List containing a tuple of coordinates (i: int, j: int) and a float value.

    
    nnz : int
        The number of non-zero elements (can be accessed but not modified).
    """
    
    def __init__(self, n: int, m: int, data: List[Tuple[int, int, float]]=None):
        """
        Initialize the COO matrix.

        Parameters:
        -----------
        n : int
            The number of rows.
        m : int
            The number of columns.
        data : list of tuple, optional
            List containing a tuple of coordinates i: int, j: int and a float value.

        Returns:
        --------
        None
        """
        self.n=n
        self.m=m
        ### A faire vide et laisser nnz en implicite par len(values)
        if data is None:
            self.data=[]
        else:
            self.data=data


    def fill_random(self, nnz: int, seed: int=None):
        """
        Fill the COO matrix with random values.

        Parameters:
        -----------
        nnz : int
            The number of non-zero elements to fill.
        seed : int, optional
            Seed for the random number generator.

        Returns:
        --------
        None
        """
        if seed is not None:
            random.seed(seed)

        # Generate nnz random coordinates and values
        data = []
        for i in range(nnz):
            row = random.randint(0, self.n - 1)
            col = random.randint(0, self.m - 1)
            val = random.uniform(1e-16, 1)
            data.append((row, col, val))

        # Sort the data by row and column indices
        data.sort()

        # Update the COO matrix with the generated data
        self.data = data

    @classmethod
    def random(cls, n: int, m: int, nnz: int, seed: int = None) -> 'MatCooPure':
        """
        Create a random COO matrix.

        Parameters:
        -----------
        n : int
            The number of rows.
        m : int
            The number of columns.
        nnz : int
            The number of non-zero elements to fill.
        seed : int, optional
            Seed for the random number generator.

        Returns:
        --------
        MatCooPure
            A new random COO matrix.
        """
        mat=cls(n,m)
        mat.fill_random(nnz,seed)
        return mat
        
         
    @classmethod
    def identity(cls,n: int) -> 'MatCooPure':
        """
        Create an identity matrix of size n x n.

        Parameters:
        -----------
        n : int
            The size of the identity matrix.

        Returns:
        --------
        MatCooPure
            A new COO matrix representing the identity matrix.
        """
              
        data=[(i,i,1.) for i in range(n)]
        return cls(n,n,data)

    @classmethod
    def from_numpy(cls,data: np.ndarray) -> 'MatCooPure':
        """  
        Create a new COO matrix from a NumPy array.

        Parameters:
        -----------
        arr : numpy.ndarray
            The NumPy array to convert to a COO matrix.

        Returns:
        --------
        MatCooPure
            A new COO matrix representing the input array.
        
        Examples
        >>> import numpy as np
        >>> A=np.array([[1,2,0],[0,0,3],[4,0,5]])
        >>> mat=MatCooPure.from_numpy(A)
        >>> print(mat)
        COO matrix ((3, 3)) with nnz=5
        data:
        (0, 0, 1)
        (0, 1, 2)
        (1, 2, 3)
        (2, 0, 4)
        (2, 2, 5)
        <BLANKLINE>
        """
        coords=np.transpose(data.nonzero())
        dat=[(i,j,data[i,j]) for i,j in coords]
        return cls(*data.shape,dat)

    def to_numpy(self) -> np.ndarray:
        """
        Convert the COO matrix to a dense NumPy array.

        Returns:
        --------
        numpy.ndarray
            A dense NumPy array representing the COO matrix.
        """
        arr = np.zeros((self.n, self.m))
        for i, j, value in self.data:
            arr[i, j] = value

        return arr
        
    def __str__(self) -> str:
        s=f"COO matrix ({self.n,self.m}) with nnz={self.nnz}\n"
        s+=f"data:\n"
        lignes = [f"COO matrix ({(self.n, self.m)}) with nnz={self.nnz}", "data:"]
        lignes += [f"({i}, {j}, {v})" for i, j, v in self.data]
        return "\n".join(lignes) + "\n"

    def _repr_pretty_(self, p: 'PrettyPrinter', cycle: bool) -> None:
        """
        Display the matrix in a pretty format when using IPython.

        Parameters:
        -----------
        p : IPython.lib.pretty.RepresentationPrinter
            The printer object used to display the matrix.
        cycle : bool
            Whether the matrix contains a cycle (ignored).

        Returns:
        --------
        None
        """
        # Print the matrix as a table
        p.text(f"COO matrix ({self.n}, {self.m}) with nnz={self.nnz}\n")
        p.text(f"row col    value\n")
        if len(self.data)<10:
            for d in self.data : 
                p.text(f"{d[0]:3d} {d[1]:3d}   {d[2]:.2e}\n")
        else:
            for d in self.data[:5] : 
                p.text(f"{d[0]:3d} {d[1]:3d}   {d[2]:.2e}\n")
            p.text("...\n")
            for d in self.data[-5:] :
                p.text(f"{d[0]:3d} {d[1]:3d}   {d[2]:.2e}\n")
        return

    @property
    def nnz(self) -> int:
        """
        The number of non-zero elements in the matrix.
        """
        return len(self.data)
    
    def mat_vec_product(self, vec: np.ndarray) -> np.ndarray:
        """
        Calculate the product of the COO matrix and a vector.

        Parameters:
        ----------
        vec : numpy.ndarray
            The vector to multiply with the matrix.

        Returns:
        --------
        numpy.ndarray
            The result of the matrix-vector multiplication.

        Examples:
        --------
        >>> I4 = MatCooPure.identity(4)
        >>> I4.mat_vec_product(np.arange(4))
        array([0., 1., 2., 3.])
        """
        vec=np.array(vec)
        if self.m != vec.size:
            raise ValueError(f"Number of columns m={self.m} does not match size of vector {vec.size}")

        res = np.zeros(self.n)
        for i, j, val in self.data:
            res[i] += val * vec[j]
        return res

    def dot(self, vec: np.ndarray) -> np.ndarray:
        """
        Calculate the product of the COO matrix and a vector.
        
        Is just an alias for mat_vec_product.
        
        Parameters:
        ----------
        vec : numpy.ndarray
            The vector to multiply with the matrix.
            
        Returns:
        --------
        numpy.ndarray
            The result of the matrix-vector multiplication.
        """
        return self.mat_vec_product(vec)

    def mat_mat_product(self, other: 'MatCooPure') -> 'MatCooPure':
        """
        Calculate the product of two COO matrices.

        Parameters:
        ----------
        other : MatCooPure
            The matrix to multiply with the matrix.

        Returns:
        --------
        MatCooPure
            The result of the matrix-matrix multiplication.

        Examples:
        --------
        >>> data1 = [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)]
        >>> mat1 = MatCooPure(3, 3, data1)
        >>> data2 = [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)]
        >>> mat2 = MatCooPure(3, 3, data2)
        >>> result = mat1.mat_mat_product(mat2)
        >>> result.data
        [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 24.0), (2, 1, 8.0), (2, 2, 25.0)]
        """
        if self.m != other.n:
            raise ValueError(f"Number of columns in first matrix m={self.m} does not match number of rows in second matrix n={other.n}")

        # Initialize an empty dictionary for the resulting matrix data
        result_data = {}
        for i, j, value in self.data:
                for k, l, other_value in other.data:
                    if j == k: 
                        # when I can add the two values, I check if the key is already in the dictionary and if not I add it
                        if (i, l) not in result_data:
                            result_data[(i, l)] = 0
                        result_data[(i, l)] += value * other_value
        # Convert the dictionary to a list of tuples
        result_data = [(i, l, value) for (i, l), value in result_data.items()]
        return MatCooPure(self.n, other.m, result_data)

    def __mul__(self, vec: np.ndarray) -> np.ndarray:
        """
        Overload the * operator to calculate the product of the COO matrix and a vector.

        Parameters:
        ----------
        vec : arraylike
            The vector to multiply with the matrix.

        Returns:
        --------
        numpy.ndarray or MatCooPure
            The result of the matrix-vector or matrix-matrix multiplication.
        """
        if isinstance(vec, (np.ndarray, list, tuple)):
            return self.mat_vec_product(vec)
        else:
            raise TypeError(f"Unsupported operand type for *: {type(vec)}")

    def __matmul__(self, other: 'MatCooPure') -> 'MatCooPure':
        """
        Overload the @ operator for matrix multiplication.

        Parameters:
        ----------
        other : MatCooPure
            The matrix to multiply with the matrix.

        Returns:
        --------
        MatCooPure
            The result of the matrix-matrix multiplication.

        Raises:
        ------
        TypeError
            If the other operand is not an instance of MatCooPure.

        Examples:
        --------
        >>> data1 = [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)]
        >>> mat1 = MatCooPure(3, 3, data1)
        >>> data2 = [(0, 0, 1.0), (0, 1, 2.0), (2, 0, 4.0), (2, 2, 5.0)]
        >>> mat2 = MatCooPure(3, 3, data2)
        >>> result = mat1 @ mat2
        >>> numpy_result = mat1.to_numpy()@mat2.to_numpy()
        >>> print(result)
        COO matrix ((3, 3)) with nnz=5
        data:
        (0, 0, 1.0)
        (0, 1, 2.0)
        (2, 0, 24.0)
        (2, 1, 8.0)
        (2, 2, 25.0)
        <BLANKLINE>
        >>> print(numpy_result)
        [[ 1.  2.  0.]
         [ 0.  0.  0.]
         [24.  8. 25.]]
        >>> print(np.allclose(result.to_numpy(), numpy_result))
        True
        """
        #may not be the safest call, but it works better with notebooks imports
        try: 
            return self.mat_mat_product(other)
        except Exception as e:
            print(f"An error occurred during matrix multiplication: {e}")


    def __neg__(self) -> 'MatCooPure':
        """
        Negates the matrix.

        Returns
        -------
        MatCooPure
            A new matrix that is the negation of the current matrix.
        """
        data = [(i, j, -val) for i, j, val in self.data]
        return MatCooPure(self.n, self.m, data)
    
    def __sub__(self, other: 'MatCooPure') -> 'MatCooPure':
        """
        Subtracts another matrix from this matrix.

        Parameters
        ----------
        other : MatCooPure
            The other matrix to subtract from this matrix.

        Returns
        -------
        MatCooPure
            A new matrix that is the result of the subtraction.
        """
        return self + (-other)

    ##############################
    # Student work zone          #
    # Replace path by some       #
    # meaningful code            # 
    ##############################

    

    def __eq__(self, other: 'MatCooPure') -> bool:
        """
        Checks if this matrix is equal to another matrix.

        Parameters:
        -----------
        other (MatCooPure): The other matrix to compare with.

        Returns:
        --------
        bool: True if the matrices are equal, False otherwise.

        Example:
        --------
        >>> print(MatCooPure(2, 2, [(0, 0, 1.0), (1, 1, 2.0)]) == MatCooPure(2, 2, [(0, 0, 1.0), (1, 1, 2.0)]))
        True
        """

        try:
            f1, d1 = _vers_dict(self)
            f2, d2 = _vers_dict(other)
        except AttributeError:
            return NotImplemented
        if f1 != f2:
            return False
        return all(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) < 1e-15
                   for k in d1.keys() | d2.keys())
                

    def dev_show(self):
        """
        Displays a summary of the matrix and its memory footprint.

        This method prints a summary of the matrix, including its size, the number of non-zero elements,
        the first two and last two non-zero elements (if they exist), and the memory footprint of the data.

        Example:
        --------
        >>> mat = MatCooPure(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])
        >>> mat.dev_show()
        COO matrix ((2, 2)) with nnz=2
          0   0   1.00e+00
          1   1   2.00e+00
        ...
          0   0   1.00e+00
          1   1   2.00e+00
        <BLANKLINE>
        Memory footprint: 72 bytes.
        """
        print(f"COO matrix ({(self.n, self.m)}) with nnz={self.nnz}")
        for i, j, v in self.data[:2]:
            print(f"  {i}   {j}   {v}")
        print("...")
        for i, j, v in self.data[-2:]:
            print(f"  {i}   {j}   {v}")
        print()
        if self.data == [] or self.data == None:
            print(f"Memory footprint: 0 bytes.")
        else:
            print(f"Memory footprint: {sys.getsizeof(self.data)} bytes.")

    def __add__(self, other):
        """
        Adds this matrix to another matrix. It is performed by adding the values of the two matrices term by term.

        Parameters
        ----------
        other : MatCooPure
            The other matrix to add to this matrix.

        Returns
        -------
        MatCooPure
            A new matrix that is the result of the addition.

        Examples
        --------
        >>> mat1 = MatCooPure(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])
        >>> mat2 = MatCooPure(2, 2, [(0, 1, 1.0), (1, 1, -2.0)])
        >>> print(mat1 + mat2)
        COO matrix ((2, 2)) with nnz=2
        data:
        (0, 0, 1.0)
        (0, 1, 1.0)
        <BLANKLINE>
        """
        somme = {}
        for i, j, v in self.data + other.data:
            somme[(i, j)] = somme.get((i, j), 0.0) + v
        data = [(i, j, v) for (i, j), v in sorted(somme.items()) if abs(v) > 1e-15]
        return MatCooPure(self.n, self.m, data)
