fn main() {
    // This generates the scaffolding code for UniFFI.
    // It looks for an interface definition (usually in lib.rs)
    // and creates the C-compatible entry points for Kotlin/Swift.
    uniffi_build::generate_scaffolding("./src/lib.rs").unwrap();
}
