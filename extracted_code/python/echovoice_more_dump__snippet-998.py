Duplicate dependency in Cargo.toml:
Remove the duplicate rayon entry from your Cargo.toml file.

Module redefinition errors:
The modules universal_engine and validation are being defined multiple times. You should either:

Remove the pub mod universal_engine; and pub mod validation; lines and keep the inline definitions

Or remove the inline module definitions and keep the external module declarations

Duplicate imports:
Remove the duplicate imports of Rng and Result from various files.

Missing dependencies:
Add these missing dependencies to your Cargo.toml:

