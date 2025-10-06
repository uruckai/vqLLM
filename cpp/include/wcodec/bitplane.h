/**
 * @file bitplane.h
 * @brief Bitplane coding for progressive representation
 */

#pragma once

#include "types.h"
#include <vector>

namespace wcodec {

/**
 * @brief Bitplane representation of coefficients
 */
class BitplaneCoder {
public:
    /**
     * @brief Pack signed values into bitplanes
     * @param data Input data (int16 coefficients)
     * @param size Number of values
     * @param num_planes Number of bitplanes to encode
     * @return Packed bitplanes (MSB first)
     */
    static std::vector<std::vector<uint8_t>> pack(
        const int16_t* data,
        size_t size,
        int num_planes = 16
    );

    /**
     * @brief Unpack bitplanes to values
     * @param bitplanes Packed bitplanes
     * @param data Output data
     * @param size Number of values
     */
    static void unpack(
        const std::vector<std::vector<uint8_t>>& bitplanes,
        int16_t* data,
        size_t size
    );

    /**
     * @brief Encode sign and magnitude separately
     * @param data Input signed data
     * @param size Number of values
     * @param signs Output sign bits (1 bit per value)
     * @param magnitudes Output magnitude values
     */
    static void separateSignMagnitude(
        const int16_t* data,
        size_t size,
        std::vector<uint8_t>& signs,
        std::vector<uint16_t>& magnitudes
    );

    /**
     * @brief Reconstruct signed values from sign and magnitude
     * @param signs Sign bits
     * @param magnitudes Magnitude values
     * @param data Output signed data
     * @param size Number of values
     */
    static void combineSignMagnitude(
        const std::vector<uint8_t>& signs,
        const std::vector<uint16_t>& magnitudes,
        int16_t* data,
        size_t size
    );

    /**
     * @brief Count significant coefficients in a bitplane
     * @param plane Bitplane data
     * @return Number of non-zero bits
     */
    static size_t countSignificant(const std::vector<uint8_t>& plane);
};

} // namespace wcodec

