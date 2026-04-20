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
#include "cuda/utils.cuh"

std::vector<bf16> load_tensor(std::string const &filename, usize rows, usize cols) {
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

std::vector<f32> load_f32_tensor(std::string const &filename, usize rows, usize cols) {
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
