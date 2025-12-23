Now, note that the ensure_dataset_exists function is defined to use blocking reqwest. We are in an async context? Actually, the run_experiment function is not async.

We are using reqwest::blocking in ensure_dataset_exists, so it's okay.

Now, let's implement the conversion from ProcessedGraph to Graph.

The Graph struct is from the cognitive_crystal library. We assume it has the following fields:
