Advanced Features: You now have a framework to add many more molecular descriptors. The more relevant features you include, the richer the representation in the Molecular Cube.

3D Conformation Generation: Added code to generate 3D conformations using RDKit's EmbedMolecule. This is crucial because some descriptors (and potentially docking programs later) require 3D structures.

Feature Scaling: Implemented StandardScaler (you can also use MinMaxScaler) to scale the feature matrix before dimensionality reduction. This is essential to prevent features with larger values from dominating the PCA or t-SNE.

t-SNE Integration: Added the option to use t-SNE instead of PCA. t-SNE often does a better job of preserving local neighborhoods in high-dimensional space, but it's more computationally intensive. Experiment with both to see which works best for your data.

Mordred Descriptors (Commented Out): Included commented-out code showing how to use the mordred library to calculate a wide range of molecular descriptors. If you want to use these, you'll need to install mordred: pip install mordred.

