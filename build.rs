#[allow(non_snake_case)]
fn main() {
	let LIBTORCH = "/home/spock/sw/libtorch/libtorch";

	let torch_include = format!("{LIBTORCH}/include");
	let torch_include_torch = format!("{LIBTORCH}/include/torch/csrc/api/include");

	cc::Build::new()
		.cpp(true)
		.file("cpp/cuda_shim.cpp")
		.flag(format!("-isystem{torch_include}"))
		.flag(format!("-isystem{torch_include_torch}"))
		.flag("-std=c++23")
		.compile("cuda_shim");

	println!("cargo:rustc-link-arg=-Wl,-rpath,{LIBTORCH}/lib");
	println!("cargo:rustc-link-search=native={LIBTORCH}/lib");
	println!("cargo:rustc-link-lib=dylib=torch");
	println!("cargo:rustc-link-lib=dylib=torch_global_deps");
	println!("cargo:rustc-link-lib=dylib=torch_cpu");
	println!("cargo:rustc-link-lib=dylib=torch_cuda");
	println!("cargo:rustc-link-lib=dylib=torch_cuda_linalg");
	println!("cargo:rustc-link-lib=dylib=c10");
	println!("cargo:rustc-link-lib=dylib=c10_cuda");
	println!("cargo:rustc-link-lib=cuda");
	println!("cargo:rustc-link-lib=cudart");
	println!("cargo:rustc-link-lib=nvrtc");

	println!("cargo:rerun-if-changed=build.rs");
	println!("cargo:rerun-if-changed=cpp/cuda_shim.cpp");
}

/*
//libaoti_custom_ops.so*
//libbackend_with_compiler.so*
/**/libc10_cuda.so*
//libc10d_cuda_test.so*
/**/libc10.so*
libcaffe2_nvrtc.so*
//libjitbackend_test.so*
//libnnapi_backend.so*
//libshm.so*
//libtorchbind_test.so*
/**/libtorch_cpu.so*
/**/libtorch_cuda_linalg.so*
/**/libtorch_cuda.so*
/**/libtorch_global_deps.so*
//libtorch_python.so*
/**/libtorch.so*
*/
