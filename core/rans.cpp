/**
 * @file rans.cpp
 * @brief rANS implementation
 */

#include "rans.h"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace codec {

// ============================================================================
// Encoder
// ============================================================================

RANSEncoder::RANSEncoder() : state_(RANS_L) {
    memset(freqs_, 0, sizeof(freqs_));
}

void RANSEncoder::buildFrequencies(const uint8_t* data, size_t size) {
    // Count frequencies
    memset(freqs_, 0, sizeof(freqs_));
    for (size_t i = 0; i < size; i++) {
        freqs_[data[i]]++;
    }
    
    // Ensure no zero frequencies
    for (int i = 0; i < 256; i++) {
        if (freqs_[i] == 0) freqs_[i] = 1;
    }
    
    normalizeFrequencies();
}

void RANSEncoder::normalizeFrequencies() {
    // Normalize to RANS_SCALE
    uint64_t total = 0;
    for (int i = 0; i < 256; i++) {
        total += freqs_[i];
    }
    
    // Scale frequencies
    uint32_t scaled_freqs[256];
    uint32_t sum = 0;
    
    for (int i = 0; i < 256; i++) {
        // Scale and ensure minimum frequency of 1
        scaled_freqs[i] = std::max(1u, static_cast<uint32_t>((freqs_[i] * RANS_SCALE) / total));
        sum += scaled_freqs[i];
    }
    
    // Adjust to exactly RANS_SCALE
    if (sum != RANS_SCALE) {
        // Find most frequent symbol and adjust
        int max_idx = 0;
        for (int i = 1; i < 256; i++) {
            if (scaled_freqs[i] > scaled_freqs[max_idx]) {
                max_idx = i;
            }
        }
        
        int32_t diff = static_cast<int32_t>(RANS_SCALE) - static_cast<int32_t>(sum);
        int32_t new_freq = static_cast<int32_t>(scaled_freqs[max_idx]) + diff;
        // Ensure frequency is at least 1 and doesn't overflow
        scaled_freqs[max_idx] = std::max(1u, std::min(static_cast<uint32_t>(new_freq), RANS_SCALE - 255u));
    }
    
    // Build cumulative frequency table
    uint32_t cumul = 0;
    for (int i = 0; i < 256; i++) {
        symbols_[i].start = cumul;
        symbols_[i].freq = scaled_freqs[i];
        cumul += scaled_freqs[i];
    }
}

void RANSEncoder::renormalize(std::vector<uint8_t>& output) {
    while (state_ >= RANS_L * RANS_SCALE) {
        output.push_back(state_ & 0xFF);
        state_ >>= 8;
    }
}

void RANSEncoder::put(uint8_t symbol, std::vector<uint8_t>& output) {
    const RANSSymbol& s = symbols_[symbol];
    
    // Safety check: frequency must be at least 1
    if (s.freq == 0) {
        fprintf(stderr, "ERROR: Symbol %d has zero frequency!\n", symbol);
        return; // Skip encoding this symbol
    }
    
    // Renormalize if needed
    uint32_t max_state = ((RANS_L >> RANS_SCALE_BITS) << 8) * s.freq;
    while (state_ >= max_state) {
        output.push_back(state_ & 0xFF);
        state_ >>= 8;
    }
    
    // Encode symbol
    state_ = ((state_ / s.freq) << RANS_SCALE_BITS) + (state_ % s.freq) + s.start;
}

void RANSEncoder::flush(std::vector<uint8_t>& output) {
    // Write final state (4 bytes, little-endian)
    output.push_back((state_ >> 0) & 0xFF);
    output.push_back((state_ >> 8) & 0xFF);
    output.push_back((state_ >> 16) & 0xFF);
    output.push_back((state_ >> 24) & 0xFF);
}

std::vector<uint8_t> RANSEncoder::encode(const uint8_t* data, size_t size) {
    std::vector<uint8_t> output;
    
    // Write size header (4 bytes)
    output.push_back((size >> 0) & 0xFF);
    output.push_back((size >> 8) & 0xFF);
    output.push_back((size >> 16) & 0xFF);
    output.push_back((size >> 24) & 0xFF);
    
    // Write frequency table (256 * 2 bytes = 512 bytes)
    for (int i = 0; i < 256; i++) {
        output.push_back((symbols_[i].freq >> 0) & 0xFF);
        output.push_back((symbols_[i].freq >> 8) & 0xFF);
    }
    
    // Reset state
    state_ = RANS_L;
    
    // Encode in reverse order (rANS requirement)
    for (int64_t i = static_cast<int64_t>(size) - 1; i >= 0; i--) {
        put(data[i], output);
    }
    
    // Flush final state
    flush(output);
    
    return output;
}

std::vector<uint8_t> RANSEncoder::encodeWithoutFreqTable(const uint8_t* data, size_t size) {
    std::vector<uint8_t> output;
    
    // Write size header (4 bytes)
    output.push_back((size >> 0) & 0xFF);
    output.push_back((size >> 8) & 0xFF);
    output.push_back((size >> 16) & 0xFF);
    output.push_back((size >> 24) & 0xFF);
    
    // NO frequency table written here - it's stored globally
    
    // Reset state
    state_ = RANS_L;
    
    // Encode in reverse order (rANS requirement)
    for (int64_t i = static_cast<int64_t>(size) - 1; i >= 0; i--) {
        put(data[i], output);
    }
    
    // Flush final state
    flush(output);
    
    return output;
}

// ============================================================================
// Decoder
// ============================================================================

RANSDecoder::RANSDecoder() : state_(0), ptr_(nullptr) {}

void RANSDecoder::initState(const uint8_t* data) {
    ptr_ = data;
    // Read 4 bytes for initial state (little-endian)
    state_ = (ptr_[0] << 0) | (ptr_[1] << 8) | (ptr_[2] << 16) | (ptr_[3] << 24);
    ptr_ += 4;
}

uint8_t RANSDecoder::get(const RANSSymbol* symbols) {
    // Find symbol
    uint32_t cum_freq = state_ & (RANS_SCALE - 1);
    
    // Linear search (could optimize with binary search)
    uint8_t symbol = 0;
    for (int i = 0; i < 256; i++) {
        if (symbols[i].start <= cum_freq && 
            cum_freq < symbols[i].start + symbols[i].freq) {
            symbol = i;
            break;
        }
    }
    
    const RANSSymbol& s = symbols[symbol];
    
    // Update state
    state_ = s.freq * (state_ >> RANS_SCALE_BITS) + (cum_freq - s.start);
    
    // Renormalize
    while (state_ < RANS_L) {
        state_ = (state_ << 8) | (*--ptr_);
    }
    
    return symbol;
}

std::vector<uint8_t> RANSDecoder::decode(const uint8_t* compressed, 
                                        size_t compressed_size,
                                        size_t output_size) {
    if (compressed_size < 4 + 512 + 4) {
        return std::vector<uint8_t>();  // Invalid data
    }
    
    // Read size header
    uint32_t stored_size = (compressed[0] << 0) | (compressed[1] << 8) |
                          (compressed[2] << 16) | (compressed[3] << 24);
    
    if (stored_size != output_size) {
        return std::vector<uint8_t>();  // Size mismatch
    }
    
    // Read frequency table
    RANSSymbol symbols[256];
    uint32_t cumul = 0;
    const uint8_t* freq_ptr = compressed + 4;
    
    for (int i = 0; i < 256; i++) {
        uint16_t freq = (freq_ptr[0] << 0) | (freq_ptr[1] << 8);
        symbols[i].start = cumul;
        symbols[i].freq = freq;
        cumul += freq;
        freq_ptr += 2;
    }
    
    // Initialize decoder state (at end of compressed data)
    ptr_ = compressed + compressed_size;
    initState(ptr_ - 4);
    ptr_ -= 4;  // Move back past the state bytes
    
    // Decode (forward order)
    std::vector<uint8_t> output(output_size);
    for (size_t i = 0; i < output_size; i++) {
        output[i] = get(symbols);
    }
    
    return output;
}

} // namespace codec

