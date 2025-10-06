/**
 * @file transform.h
 * @brief Integer DCT and ADST transforms for weight compression
 */

#pragma once

#include "types.h"

namespace wcodec {

/**
 * @brief Transform operations on 8x8 blocks
 */
class Transform {
public:
    /**
     * @brief Forward 8x8 integer DCT-II
     * @param input Input block (8x8, row-major)
     * @param output Output coefficients (8x8, row-major)
     */
    static void forwardDCT8x8(const int8_t* input, int16_t* output);

    /**
     * @brief Inverse 8x8 integer DCT-II
     * @param input Input coefficients (8x8)
     * @param output Output block (8x8)
     */
    static void inverseDCT8x8(const int16_t* input, int8_t* output);

    /**
     * @brief Forward 8x8 integer ADST
     * @param input Input block (8x8)
     * @param output Output coefficients (8x8)
     */
    static void forwardADST8x8(const int8_t* input, int16_t* output);

    /**
     * @brief Inverse 8x8 integer ADST
     * @param input Input coefficients (8x8)
     * @param output Output block (8x8)
     */
    static void inverseADST8x8(const int16_t* input, int8_t* output);

    /**
     * @brief Select best transform for a block via RD probe
     * @param block Input block (8x8)
     * @param lambda RD lambda parameter
     * @return Best transform type
     */
    static TransformType selectTransform(const int8_t* block, double lambda = 1.0);

    /**
     * @brief Zig-zag scan order for 8x8 block
     * @param input 2D block
     * @param output 1D array in zig-zag order
     */
    static void zigzagScan(const int16_t* input, int16_t* output);

    /**
     * @brief Inverse zig-zag scan
     * @param input 1D array in zig-zag order
     * @param output 2D block
     */
    static void inverseZigzagScan(const int16_t* input, int16_t* output);

private:
    // Integer DCT basis (scaled and rounded)
    static const int16_t kDCTBasis[64][64];
    
    // ADST basis
    static const int16_t kADSTBasis[64][64];
    
    // Zig-zag scan indices
    static const uint8_t kZigzagIndices[64];
};

} // namespace wcodec

