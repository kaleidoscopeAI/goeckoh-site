We'll create a function ensure_parquet_dataset_exists that only downloads if the path doesn't exist and the URL is non-empty and the extension is parquet.

But in the Run command, the URL is provided only for parquet. For other formats, the URL is not used.

