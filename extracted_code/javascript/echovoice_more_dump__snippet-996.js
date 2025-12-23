We are given a RunConfig that contains all the parameters.

We need to load the dataset. The function load_or_process_dataset is provided for this.

Then, if evolution_generations is greater than 0, we run the evolutionary algorithm to optimize embedding parameters.

Then, we set up the annealing schedule and optional parameters (adaptive cooling, multi-objective, chemical validation).

Then, we run either k-fold or holdout validation.

Finally, we print and export the results.

