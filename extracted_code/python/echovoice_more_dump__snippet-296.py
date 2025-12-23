    It seems we have both a universal_engine module declared via pub mod universal_engine; and then immediately redefined with a block. We should remove the block and use the module from the file.

    Similarly for validation.

    Multiple imports of Rng and Result: Remove duplicate imports.

    Unresolved import chrono: Add chrono to Cargo.toml.

    Unresolved import glob: Add glob to Cargo.toml.

    Stable channel features: Remove the #![feature(...)] lines since we are on stable.

    Various other errors: We'll address them one by one.

