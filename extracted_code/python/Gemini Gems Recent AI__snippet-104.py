from flask import Flask, jsonify, request
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
import numpy as np
from sklearn.preprocessing import StandardScaler

