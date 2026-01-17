fn main() {
    println!("cargo:rerun-if-changed=sp.proto");
    if let Ok(path) = protoc_bin_vendored::protoc_bin_path() {
        std::env::set_var("PROTOC", path);
    }
    prost_build::Config::new()
        .compile_protos(&["sp.proto"], &["."])
        .expect("failed to compile sp.proto");
}
