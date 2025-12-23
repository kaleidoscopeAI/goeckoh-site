We also note that the Run command has an option to download the dataset from a URL if it doesn't exist (via ensure_dataset_exists).

We must adjust: if the file is not parquet, we don't want to download from the URL (because the URL is for the parquet version).
