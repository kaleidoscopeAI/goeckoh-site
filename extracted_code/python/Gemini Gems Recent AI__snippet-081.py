from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from celery import Celery  # For asynchronous tasks
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
from sklearn.preprocessing import StandardScaler
from scipy.linalg import expm
import numpy as np

