/**
 * @file rans.cpp
 * @brief Implementation of rANS entropy coder
 */

#include "wcodec/rans.h"
#include <algorithm>
#include <cstring>
#include <cassert>

namespace wcodec {

constexpr uint32_t kFreqScale = 4096;  // Total frequency scale

// RansEncoder implementation

RansEncoder::RansEncoder() : state_(kRansL) {}

RansEncoder::~RansEncoder() {}

void RansEncoder::init(const uint32_t* freqs, size_t num_symbols) {
    symbols_.resize(num_symbols);
    
    uint32_t cum = 0;
    for (size_t i = 0; i < num_symbols; ++i) {
        symbols_[i].freq = static_cast<uint16_t>(freqs[i]);
        symbols_[i].cum_freq = static_cast<uint16_t>(cum);
        symbols_[i].scale = kFreqScale;
        cum += freqs[i];
    }
    
    assert(cum == kFreqScale);
}

void RansEncoder::encode(uint8_t symbol) {
    const RansSymbol& sym = symbols_[symbol];
    
    // Renormalize if needed
    uint32_t x_max = ((kRansL >> 16) << 16) * sym.scale;
    while (state_ >= x_max) {
        output_.push_back(static_cast<uint8_t>(state_ & 0xFF));
        state_ >>= 8;
    }
    
    // Encode symbol
    state_ = ((state_ / sym.scale) << 16) + 
             (state_ % sym.scale) + 
             sym.cum_freq;
}

void RansEncoder::finish(std::vector<uint8_t>& output) {
    // Flush final state
    for (int i = 0; i < 4; ++i) {
        output_.push_back(static_cast<uint8_t>(state_ & 0xFF));
        state_ >>= 8;
    }
    
    // Reverse output (rANS outputs backwards)
    std::reverse(output_.begin(), output_.end());
    
    output = std::move(output_);
}

void RansEncoder::reset() {
    state_ = kRansL;
    output_.clear();
}

// RansDecoder implementation

RansDecoder::RansDecoder() : state_(0), data_(nullptr), size_(0), pos_(0) {}

RansDecoder::~RansDecoder() {}

void RansDecoder::init(const uint8_t* data, size_t size,
                       const uint32_t* freqs, size_t num_symbols) {
    data_ = data;
    size_ = size;
    pos_ = 0;
    
    // Initialize symbol table
    symbols_.resize(num_symbols);
    uint32_t cum = 0;
    for (size_t i = 0; i < num_symbols; ++i) {
        symbols_[i].freq = static_cast<uint16_t>(freqs[i]);
        symbols_[i].cum_freq = static_cast<uint16_t>(cum);
        symbols_[i].scale = kFreqScale;
        cum += freqs[i];
    }
    
    // Read initial state (4 bytes, big-endian-ish)
    state_ = 0;
    for (int i = 0; i < 4 && pos_ < size_; ++i) {
        state_ = (state_ << 8) | data_[pos_++];
    }
}

uint8_t RansDecoder::decode() {
    // Find symbol by binary search on cumulative frequencies
    uint32_t cf = state_ % kFreqScale;
    
    size_t symbol = 0;
    for (size_t i = 0; i < symbols_.size(); ++i) {
        if (symbols_[i].cum_freq <= cf && 
            cf < symbols_[i].cum_freq + symbols_[i].freq) {
            symbol = i;
            break;
        }
    }
    
    const RansSymbol& sym = symbols_[symbol];
    
    // Decode
    state_ = sym.freq * (state_ / kFreqScale) + 
             (state_ % kFreqScale) - sym.cum_freq;
    
    // Renormalize
    while (state_ < kRansL && pos_ < size_) {
        state_ = (state_ << 8) | data_[pos_++];
    }
    
    return static_cast<uint8_t>(symbol);
}

bool RansDecoder::hasMore() const {
    return pos_ < size_;
}

// Utility functions

void buildFrequencyTable(const int8_t* data, size_t size, uint32_t* freqs) {
    // Initialize counts
    std::memset(freqs, 0, 256 * sizeof(uint32_t));
    
    // Count occurrences (treat int8_t as uint8_t for indexing)
    for (size_t i = 0; i < size; ++i) {
        uint8_t idx = static_cast<uint8_t>(data[i]);
        freqs[idx]++;
    }
}

void normalizeFrequencies(uint32_t* freqs, size_t num_symbols, uint32_t scale) {
    // Sum current frequencies
    uint64_t total = 0;
    for (size_t i = 0; i < num_symbols; ++i) {
        total += freqs[i];
    }
    
    if (total == 0) {
        // Uniform distribution if no data
        for (size_t i = 0; i < num_symbols; ++i) {
            freqs[i] = scale / num_symbols;
        }
        return;
    }
    
    // Scale frequencies
    uint32_t scaled_sum = 0;
    for (size_t i = 0; i < num_symbols; ++i) {
        if (freqs[i] > 0) {
            freqs[i] = std::max(1u, static_cast<uint32_t>(
                (freqs[i] * static_cast<uint64_t>(scale)) / total
            ));
        }
        scaled_sum += freqs[i];
    }
    
    // Adjust to exactly match scale
    if (scaled_sum != scale) {
        // Find largest frequency and adjust
        size_t max_idx = 0;
        for (size_t i = 1; i < num_symbols; ++i) {
            if (freqs[i] > freqs[max_idx]) {
                max_idx = i;
            }
        }
        
        int32_t diff = static_cast<int32_t>(scale) - static_cast<int32_t>(scaled_sum);
        freqs[max_idx] = std::max(1u, static_cast<uint32_t>(
            static_cast<int32_t>(freqs[max_idx]) + diff
        ));
    }
}

} // namespace wcodec

