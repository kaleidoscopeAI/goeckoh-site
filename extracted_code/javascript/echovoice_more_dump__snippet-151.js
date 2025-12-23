We'll adjust the load_or_process_dataset function to only attempt download for parquet.

Now, note that the ensure_dataset_exists function is called in the InspectDataset command. We want to keep that behavior.

We'll create a helper function that downloads only if the URL is not empty.

