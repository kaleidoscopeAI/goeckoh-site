import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os


    filename="quantum_engine.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"


class QuantumEngine:
    def __init__(self, model_type="LogisticRegression", model_path="quantum_engine_model.joblib", test_size=0.2, scaler_type="StandardScaler"):
        """
        Enhanced QuantumEngine for processing numerical and structured data.
        Args:
            model_type (str): Type of model to use ('LogisticRegression', 'SVM', 'RandomForest').
            model_path (str): Path to save/load the trained model.
            test_size (float): Proportion of data to use for testing.
            scaler_type (str): Type of scaler to use ('StandardScaler', 'MinMaxScaler').
        """
        self.model_type = model_type
        self.model_path = model_path
        self.test_size = test_size
        self.scaler_type = scaler_type
        self.model = None
        self.scaler = self._initialize_scaler()

    def _initialize_scaler(self):
        """
        Initializes the scaler based on the specified type.
        """
        if self.scaler_type == "StandardScaler":
            return StandardScaler()
        elif self.scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")

    def _initialize_model(self):
        """
        Initializes the model based on the specified type.
        """
        if self.model_type == "LogisticRegression":
            return LogisticRegression()
        elif self.model_type == "SVM":
            return SVC(probability=True)
        elif self.model_type == "RandomForest":
            return RandomForestClassifier()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def initialize(self):
        """
        Initializes the engine by loading an existing model or creating a new one.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logging.info(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                logging.error(f"Error loading model from {self.model_path}: {e}")
                self.model = self._initialize_model()
        else:
            self.model = self._initialize_model()
            logging.info(f"No existing model found. Initialized a new {self.model_type} model.")

    def process_data(self, data_chunk):
        """
        Processes a chunk of data to extract validated insights.
        Args:
            data_chunk (dict): Chunk of data to process (features and labels).
        Returns:
            list: List of validated insights.
        """
        try:
            X = data_chunk["features"]
            y = data_chunk["labels"]

            if self.model is None:
                raise ValueError("Model is not initialized. Call initialize() first.")

            # Split and scale the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # Train the model if it's untrained
            if not hasattr(self.model, "classes_"):
                self.model.fit(X_train, y_train)
                joblib.dump(self.model, self.model_path)
                logging.info(f"Trained and saved model to {self.model_path}")

            # Predict and evaluate
            predictions = self.model.predict(X_test)
            probabilities = self.model.predict_proba(X_test) if hasattr(self.model, "predict_proba") else None
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary")

            logging.info(f"Processed data chunk with accuracy: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, F1-score: {f1:.2f}")
            print(classification_report(y_test, predictions))

            # Generate insights
            insights = []
            for i, pred in enumerate(predictions):
                if pred == 1:
                    prob = probabilities[i][1] if probabilities is not None else "N/A"
                    insights.append(f"Sample {i} predicted as active (Probability: {prob}, Accuracy: {accuracy:.2f})")

            return insights

        except Exception as e:
            logging.error(f"Error processing data chunk: {e}")
            raise

    def shutdown(self):
        """
        Shuts down the quantum engine.
        """
        logging.info("Quantum engine shut down.")


