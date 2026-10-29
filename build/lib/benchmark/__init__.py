"""Matrices creuses au format COO (version pure Python et version numpy) et outil de benchmark."""

from .mat_coo_pure import MatCooPure
from .mat_coo_optim import MatCooOptim

__version__ = "1.0"
__all__ = ["MatCooPure", "MatCooOptim"]
