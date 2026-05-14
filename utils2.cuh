//------------------------------------------------------------------------------
//
// Copyright 2026 Jiri Bobek. All rights reserved.
// License: GPL 3.0 or later. See LICENSE.txt for details.
//
//------------------------------------------------------------------------------

#pragma once

#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include <string_view>
#include "cuda/utils.cuh"

struct HarnessCliOptions {
	std::string input_dir = "tmp/block_torch";
	bool use_torch_maxes = false;
};

inline std::string tensor_path(std::string const &dir, char const *filename) {
	return dir + "/" + filename;
}

inline std::string torch_tensor_path(char const *filename) {
	return tensor_path("tmp/block_torch", filename);
}

inline void print_harness_usage(char const *program, bool allow_use_torch_maxes) {
	printf("Usage: %s [--cuda-inputs]", program);
	if (allow_use_torch_maxes) {
		printf(" [--use-torch-maxes]");
	}
	printf("\n");
}

inline bool parse_harness_cli_args(int argc, char *argv[], bool allow_use_torch_maxes, HarnessCliOptions &options) {
	char const *program = argc > 0 ? argv[0] : "kernel_harness";
	for (int i = 1; i < argc; ++i) {
		std::string_view arg = argv[i];
		if (arg == "--cuda-inputs") {
			options.input_dir = "tmp/block_cuda";
			continue;
		}
		if (arg == "--use-torch-maxes") {
			if (!allow_use_torch_maxes) {
				printf("%s does not support --use-torch-maxes\n", program);
				print_harness_usage(program, allow_use_torch_maxes);
				return false;
			}
			options.use_torch_maxes = true;
			continue;
		}
		if (arg == "--help") {
			print_harness_usage(program, allow_use_torch_maxes);
			return false;
		}
		printf("Unknown option: %s\n", argv[i]);
		print_harness_usage(program, allow_use_torch_maxes);
		return false;
	}
	return true;
}

std::vector<bf16> load_tensor(std::string const &filename, usize rows, usize cols) {
	printf("Loading input from %s\n", filename.c_str());
	std::vector<bf16> data(rows * cols);
	std::ifstream a_in(filename, std::ios::binary);
	if (!a_in) {
		printf("Failed to open %s\n", filename.c_str());
		return {};
	}
	if (!a_in.read(
		reinterpret_cast<char *>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(bf16))
	)) {
		printf("Failed to read %s as [%u, %u]\n", filename.c_str(), rows, cols);
		return {};
	}
	return data;
}

std::vector<bf16> load_f8_tensor(std::string const &filename, usize rows, usize cols) {
	if (cols % 2 != 0) {
		printf("Expected even number of f8 columns in %s, got %u\n", filename.c_str(), cols);
		return {};
	}
	printf("Loading input from %s\n", filename.c_str());
	std::vector<bf16> data(rows * (cols / 2));
	std::ifstream a_in(filename, std::ios::binary);
	if (!a_in) {
		printf("Failed to open %s\n", filename.c_str());
		return {};
	}
	if (!a_in.read(
		reinterpret_cast<char *>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(bf16))
	)) {
		printf("Failed to read %s as packed f8 [%u, %u]\n", filename.c_str(), rows, cols);
		return {};
	}
	return data;
}

std::vector<f32> load_f32_tensor(std::string const &filename, usize rows, usize cols) {
	printf("Loading input from %s\n", filename.c_str());
	std::vector<f32> data(rows * cols);
	std::ifstream a_in(filename, std::ios::binary);
	if (!a_in) {
		printf("Failed to open %s\n", filename.c_str());
		return {};
	}
	if (!a_in.read(
		reinterpret_cast<char *>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(f32))
	)) {
		printf("Failed to read %s as [%u, %u]\n", filename.c_str(), rows, cols);
		return {};
	}
	return data;
}

void store_tensor(
	std::string const &filename,
	std::vector<bf16> const &data,
	[[maybe_unused]] usize rows, [[maybe_unused]] usize cols
) {
	std::ofstream out(filename, std::ios::binary);
	if (!out) {
		printf("Failed to open %s for writing\n", filename.c_str());
		return;
	}
	if (!out.write(
		reinterpret_cast<const char *>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(bf16))
	)) {
		printf("Failed to write data to %s\n", filename.c_str());
	}
	printf("Wrote output to %s\n", filename.c_str());
}

void store_f32_tensor(
	std::string const &filename,
	std::vector<f32> const &data,
	[[maybe_unused]] usize rows, [[maybe_unused]] usize cols
) {
	std::ofstream out(filename, std::ios::binary);
	if (!out) {
		printf("Failed to open %s for writing\n", filename.c_str());
		return;
	}
	if (!out.write(
		reinterpret_cast<const char *>(data.data()),
		static_cast<std::streamsize>(data.size() * sizeof(f32))
	)) {
		printf("Failed to write data to %s\n", filename.c_str());
	}
	printf("Wrote output to %s\n", filename.c_str());
}
