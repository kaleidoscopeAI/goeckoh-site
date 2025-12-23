import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from typing import Dict, Any, List, Optional, Union
import requests
import io
import json
import os

class DataPipeline:
    def __init__(self):
        self.data = None
        self.logger = logging.getLogger(__name__)

    def ingest_data(self, data_source: Union[str, Dict]) -> bool:
        """
        Ingests data from various sources, including URLs, files (CSV, Excel, JSON), or raw text.
        Handles different data formats and checks for successful data ingestion.
        """
        try:
            if isinstance(data_source, str):
                if data_source.startswith("http"):  # Handle URLs
                    self.data = self._fetch_from_url(data_source)
                elif os.path.isfile(data_source):  # Handle files
                    self.data = self._read_from_file(data_source)
                else:
                    self.data = data_source  # Assume raw text input
            elif isinstance(data_source, dict):  # Handle dictionaries
                self.data = pd.DataFrame(data_source)
            else:
                raise ValueError("Unsupported data source type.")

            if self.data is None:
                raise ValueError("Data ingestion failed: No data loaded.")

            logging.info(f"Data successfully ingested from {data_source}.")
            return True

        except Exception as e:
            logging.error(f"Error ingesting data: {e}")
            self.data = None
            return False

    def _fetch_from_url(self, url: str) -> pd.DataFrame:
        """Fetches data from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            if url.endswith(".csv"):
                return pd.read_csv(io.StringIO(response.text))
            elif url.endswith(".xlsx") or url.endswith(".xls"):
                return pd.read_excel(io.BytesIO(response.content))
            elif url.endswith(".json"):
                return pd.DataFrame(json.loads(response.text))
            else:
                raise ValueError("Unsupported file format for URL.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from URL {url}: {e}")
            raise

    def _read_from_file(self, file_path: str) -> Union[pd.DataFrame, str]:
        """Reads data from a local file."""
        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                return pd.read_excel(file_path)
            elif file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    return pd.DataFrame(json.load(f))
            else:
                with open(file_path, "r") as f:
                    return f.read()  # Read as raw text
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error reading from file {file_path}: {e}")
            raise

    def preprocess(self):
        """
        Preprocesses the data.
        """
        if self.data is None:
            logging.warning("No data to preprocess.")
            return

        try:
            if isinstance(self.data, pd.DataFrame):
                if 'IC50_nM' in self.data.columns:
                    self.data['active'] = (self.data['IC50_nM'] < 1000).astype(int)
                if 'SMILES' in self.data.columns:
                    self.data['mol'] = self.data['SMILES'].apply(Chem.MolFromSmiles)
                    self.data['mw'] = self.data['mol'].apply(Descriptors.MolWt)
                    self.data['logp'] = self.data['mol'].apply(Descriptors.MolLogP)
                    self.data['hbd'] = self.data['mol'].apply(Descriptors.NumHDonors)
                    self.data['hba'] = self.data['mol'].apply(Descriptors.NumHAcceptors)
                    self.data['tpsa'] = self.data['mol'].apply(Descriptors.TPSA)
            elif isinstance(self.data, str):
                # Placeholder for text processing logic
                self.data = self.process_text_data(self.data)

            logging.info("Data preprocessing completed.")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def process_text_data(self, text_data):
        """
        Placeholder for processing raw text data.
        """
        # Implement text processing logic here
        raise NotImplementedError("Text data processing not yet implemented.")

    def get_data_for_quantum_engine(self):
        """
        Prepares data for the Quantum Engine or a classical ML model.
        """
        if self.data is None or not isinstance(self.data, pd.DataFrame):
            logging.error("Data not available or not in DataFrame format for Quantum Engine.")
            raise ValueError("Data not available or not in DataFrame format for Quantum Engine.")

        try:
            X = self.data[['mw', 'logp', 'hbd', 'hba', 'tpsa']].values
            y = self.data['active'].values
            return {"features": X, "labels": y}
        except Exception as e:
            logging.error(f"Error preparing data for Quantum Engine: {e}")
            raise

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits data into training and testing sets.
        """
        if not isinstance(self.data, pd.DataFrame):
            logging.error("Data is not in a suitable format for splitting.")
            raise ValueError("Data is not in a suitable format for splitting.")

        try:
            X = self.data[['mw', 'logp', 'hbd', 'hba', 'tpsa']].values
            y = self.data['active'].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logging.info(f"Data split into training and testing sets (test_size={test_size}).")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise


