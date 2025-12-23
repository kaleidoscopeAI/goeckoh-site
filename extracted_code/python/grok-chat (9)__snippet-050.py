• The log write failure likely stems from attempting to write to a root-owned .run directory at the project root from the
  backend directory, indicating a permission conflict to resolve.

