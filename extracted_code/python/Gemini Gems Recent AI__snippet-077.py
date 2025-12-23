from flask import Flask, jsonify, request
from.models import Molecule  # SQLAlchemy/MongoDB models
from.tasks import calculate_similarity  # Celery task
from.utils import get_mol_from_smiles #RDKit utility functions

