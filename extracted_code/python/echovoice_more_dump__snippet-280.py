    We note that the run_experiment function is called from the Run command.

    The run_experiment function should load the dataset (possibly by downloading if not present), then run either k-fold or holdout validation.

    We see that there are two functions for evaluation: evaluate_kfold_advanced and evaluate_holdout_advanced. We assume these are implemented elsewhere (in the optimizer module).

    We need to handle the case when the dataset is not in parquet format by using the UniversalMutagEngine to process it.

