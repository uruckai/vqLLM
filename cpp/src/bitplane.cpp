/**
 * @file bitplane.cpp
 * @brief Implementation of bitplane coding
 */

#include "wcodec/bitplane.h"
#include <cstring>
#include <algorithm>

namespace wcodec {

std::vector<std::vector<uint8_t>> BitplaneCoder::pack(
    const int16_t* data,
    size_t size,
    int num_planes
) {
    std::vector<std::vector<uint8_t>> bitplanes(num_planes);
    
    // Separate sign and magnitude first
    std::vector<uint8_t> signs;
    std::vector<uint16_t> magnitudes;
    separateSignMagnitude(data, size, signs, magnitudes);
    
    // Pack each bitplane (MSB first)
    for (int plane = num_planes - 1; plane >= 0; --plane) {
        size_t num_bytes = (size + 7) / 8;
        bitplanes[num_planes - 1 - plane].resize(num_bytes, 0);
        
        for (size_t i = 0; i < size; ++i) {
            uint16_t bit = (magnitudes[i] >> plane) & 1;
            if (bit) {
                size_t byte_idx = i / 8;
                size_t bit_idx = i % 8;
                bitplanes[num_planes - 1 - plane][byte_idx] |= (1 << bit_idx);
            }
        }
    }
    
    // Add sign plane
    bitplanes.insert(bitplanes.begin(), signs);
    
    return bitplanes;
}

void BitplaneCoder::unpack(
    const std::vector<std::vector<uint8_t>>& bitplanes,
    int16_t* data,
    size_t size
) {
    if (bitplanes.empty()) return;
    
    // Extract sign plane
    std::vector<uint8_t> signs = bitplanes[0];
    
    // Reconstruct magnitudes
    std::vector<uint16_t> magnitudes(size, 0);
    
    int num_planes = bitplanes.size() - 1;
    for (int plane = 0; plane < num_planes; ++plane) {
        const auto& bitplane = bitplanes[plane + 1];
        
        for (size_t i = 0; i < size; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            
            if (byte_idx < bitplane.size()) {
                uint8_t bit = (bitplane[byte_idx] >> bit_idx) & 1;
                magnitudes[i] |= (bit << (num_planes - 1 - plane));
            }
        }
    }
    
    // Combine sign and magnitude
    combineSignMagnitude(signs, magnitudes, data, size);
}

void BitplaneCoder::separateSignMagnitude(
    const int16_t* data,
    size_t size,
    std::vector<uint8_t>& signs,
    std::vector<uint16_t>& magnitudes
) {
    size_t num_bytes = (size + 7) / 8;
    signs.resize(num_bytes, 0);
    magnitudes.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        bool is_negative = data[i] < 0;
        
        // Set sign bit
        if (is_negative) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            signs[byte_idx] |= (1 << bit_idx);
        }
        
        // Store magnitude
        magnitudes[i] = static_cast<uint16_t>(std::abs(data[i]));
    }
}

void BitplaneCoder::combineSignMagnitude(
    const std::vector<uint8_t>& signs,
    const std::vector<uint16_t>& magnitudes,
    int16_t* data,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        
        bool is_negative = false;
        if (byte_idx < signs.size()) {
            is_negative = (signs[byte_idx] >> bit_idx) & 1;
        }
        
        int16_t value = static_cast<int16_t>(magnitudes[i]);
        data[i] = is_negative ? -value : value;
    }
}

size_t BitplaneCoder::countSignificant(const std::vector<uint8_t>& plane) {
    size_t count = 0;
    for (uint8_t byte : plane) {
        // Count set bits (popcount)
        while (byte) {
            count += byte & 1;
            byte >>= 1;
        }
    }
    return count;
}

} // namespace wcodec

