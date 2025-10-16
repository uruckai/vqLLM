/**
 * @file rans.h
 * @brief rANS (range Asymmetric Numeral Systems) entropy coder
 * 
 * Simple, fast implementation for compressing differential-encoded residuals.
 */

#ifndef CODEC_RANS_H
#define CODEC_RANS_H

#include <cstdint>
#include <vector>

namespace codec {

/**
 * rANS configuration
 */
constexpr uint32_t RANS_L = (1u << 23);  // Lower bound for renormalization
constexpr uint32_t RANS_SCALE_BITS = 12; // Frequency scaling (12 bits = 4096 total)
constexpr uint32_t RANS_SCALE = (1u << RANS_SCALE_BITS);

/**
 * Symbol statistics for rANS
 */
struct RANSSymbol {
    uint16_t start;  // Cumulative frequency start
    uint16_t freq;   // Symbol frequency
};

/**
 * rANS Encoder
 */
class RANSEncoder {
public:
    RANSEncoder();
    
    /**
     * Build frequency table from data
     */
    void buildFrequencies(const uint8_t* data, size_t size);
    
    /**
     * Encode data using rANS
     * @return Compressed data (includes frequency table)
     */
    std::vector<uint8_t> encode(const uint8_t* data, size_t size);
    
    /**
     * Encode data WITHOUT frequency table (for use with global freq table)
     * @return Compressed data (size header + encoded data + state, NO freq table)
     */
    std::vector<uint8_t> encodeWithoutFreqTable(const uint8_t* data, size_t size);
    
    /**
     * Get symbol table for decoder
     */
    const RANSSymbol* getSymbolTable() const { return symbols_; }

private:
    uint32_t freqs_[256];      // Raw frequencies
    RANSSymbol symbols_[256];  // Normalized symbols
    uint32_t state_;           // rANS state
    
    void normalizeFrequencies();
    void renormalize(std::vector<uint8_t>& output);
    void put(uint8_t symbol, std::vector<uint8_t>& output);
    void flush(std::vector<uint8_t>& output);
};

/**
 * rANS Decoder
 */
class RANSDecoder {
public:
    RANSDecoder();
    
    /**
     * Decode rANS compressed data
     */
    std::vector<uint8_t> decode(const uint8_t* compressed, size_t compressed_size,
                               size_t output_size);

private:
    uint32_t state_;
    const uint8_t* ptr_;
    
    void initState(const uint8_t* data);
    uint8_t get(const RANSSymbol* symbols);
    void renormalize();
};

} // namespace codec

#endif // CODEC_RANS_H

