/**
 * @file rans.h
 * @brief rANS (range Asymmetric Numeral Systems) entropy coder
 */

#pragma once

#include "types.h"
#include <vector>
#include <cstdint>

namespace wcodec {

// rANS state
constexpr uint32_t kRansStateBits = 32;
constexpr uint32_t kRansL = 1u << 23;  // Lower bound

// Symbol with frequency
struct RansSymbol {
    uint16_t freq;      // Symbol frequency
    uint16_t cum_freq;  // Cumulative frequency (start)
    uint16_t scale;     // Total frequency scale
};

class RansEncoder {
public:
    RansEncoder();
    ~RansEncoder();

    /**
     * @brief Initialize encoder with symbol frequencies
     * @param freqs Symbol frequencies (256 symbols for int8)
     * @param num_symbols Number of symbols
     */
    void init(const uint32_t* freqs, size_t num_symbols);

    /**
     * @brief Encode a symbol
     * @param symbol Symbol value (0-255 for int8)
     */
    void encode(uint8_t symbol);

    /**
     * @brief Finalize encoding and get output
     * @param output Output buffer
     */
    void finish(std::vector<uint8_t>& output);

    /**
     * @brief Reset encoder state
     */
    void reset();

private:
    uint32_t state_;
    std::vector<uint8_t> output_;
    std::vector<RansSymbol> symbols_;
    
    void renormalize();
};

class RansDecoder {
public:
    RansDecoder();
    ~RansDecoder();

    /**
     * @brief Initialize decoder with encoded data and symbol frequencies
     * @param data Encoded data
     * @param size Size of encoded data
     * @param freqs Symbol frequencies (must match encoder)
     * @param num_symbols Number of symbols
     */
    void init(const uint8_t* data, size_t size, 
              const uint32_t* freqs, size_t num_symbols);

    /**
     * @brief Decode next symbol
     * @return Decoded symbol value
     */
    uint8_t decode();

    /**
     * @brief Check if more data available
     */
    bool hasMore() const;

private:
    uint32_t state_;
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
    std::vector<RansSymbol> symbols_;
    
    void renormalize();
};

/**
 * @brief Build frequency table from data
 * @param data Input data
 * @param size Size of data
 * @param freqs Output frequency table (must be size 256)
 */
void buildFrequencyTable(const int8_t* data, size_t size, uint32_t* freqs);

/**
 * @brief Normalize frequencies to given scale
 * @param freqs Input/output frequencies
 * @param num_symbols Number of symbols
 * @param scale Target scale (typically 4096 or 8192)
 */
void normalizeFrequencies(uint32_t* freqs, size_t num_symbols, uint32_t scale);

} // namespace wcodec

