    First, check if the file exists. If not, and if the extension is "parquet", then download from the URL.

    Otherwise, if the file exists, then process it according to its extension.

However, the ensure_dataset_exists function is already provided and is called in run_experiment (via load_or_process_dataset?).
But note: the ensure_dataset_exists function only downloads if the file doesn't exist and the URL is provided.

