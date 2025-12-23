println!("cargo:rustc-link-lib=log");
println!("cargo:rustc-link-lib=aaudio");

// Cross-compile instructions
if std::env::var("TARGET").unwrap().contains("android") {
    println!("cargo:rustc-env=ANDROID_NDK=/path/to/ndk");
}
