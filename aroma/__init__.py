"""AROMA - Augmented Reality Optimized Tone Mapping.

Python/PyTorch reimplementation of

    Chapiro, Wolski, Velazquez-Armendariz, McElvain, Ninan and Mantiuk.
    "AROMA: Augmented Reality Optimized Tone Mapping". SIGGRAPH Asia 2026.

Notation follows the paper: UPPER CASE symbols are absolute quantities in
cd/m^2 (``Y_amb``, ``Y_dmax``, ...), lower case are relative (``y_smin``,
``l_c``, ``m``, ...).  Every function names the equation it implements.
"""

from .params import AromaParams
from .pipeline import AromaPipeline

__all__ = ["AromaParams", "AromaPipeline"]
