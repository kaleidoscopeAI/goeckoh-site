Fix the import of crystal_core.

Remove or comment out the code that accesses missing fields in ChemicalFeatures and AdaptiveCoolingParams.

Change the direct field access of Metrics to method calls (e.g., .accuracy() instead of .accuracy).

For the total field in Metrics, we must see how to compute the total. The code is trying to sum the total field of each Metrics struct, but it's private. If there's a getter method, use it. Otherwise, we might need to change the visibility or use a different approach.

