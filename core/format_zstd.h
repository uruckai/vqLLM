/**
 * @file format_zstd.h
 * @brief Zstd-based layer compression format for low-memory inference
 * 
 * This is an alternative to the rANS-based format, using Zstd compression
 * for faster GPU decode via nvCOMP library.
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace codec {

// Magic number for Zstd format
constexpr uint32_t ZSTD_MAGIC = 0x5A535444;  // "ZSTD"
constexpr uint32_t ZSTD_VERSION = 1;

/**
 * @brief Layer header for Zstd compressed format
 * 
 * Layout:
 * [LayerHeaderZstd][Zstd compressed data]
 */
struct LayerHeaderZstd {
    uint32_t magic;              // ZSTD_MAGIC (0x5A535444)
    uint32_t version;            // Format version (1)
    uint32_t rows;               // Original rows
    uint32_t cols;               // Original columns
    uint32_t uncompressed_size;  // Size in bytes (rows * cols)
    uint32_t compressed_size;    // Compressed data size
    uint32_t compression_level;  // Zstd level used (1-22)
    uint32_t checksum;           // XXH64 checksum of uncompressed data
    uint32_t reserved[4];        // Reserved for future use
} __attribute__((packed));

} // namespace codec

