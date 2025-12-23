from fastapi import FastAPI
import numpy as np
from scipy.integrate import solve_ivp
from rdkit import Chem  # Real chem
from rdkit.Chem import AllChem  # Force fields

