/**
 * @file transform.cpp
 * @brief Implementation of integer transforms
 */

#include "wcodec/transform.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace wcodec {

// Zig-zag scan order for 8x8 block
const uint8_t Transform::kZigzagIndices[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// Simplified integer DCT basis (scaled for integer math)
// In practice, would use separable 1D DCT for speed
void Transform::forwardDCT8x8(const int8_t* input, int16_t* output) {
    // Simplified DCT: use separable 1D transforms
    // DCT[i,j] = sum_x sum_y input[x,y] * cos((2x+1)*i*pi/16) * cos((2y+1)*j*pi/16)
    
    const double kScale = 1.0 / 8.0;
    const double kSqrt2 = 1.41421356237;
    
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            double sum = 0.0;
            
            for (int x = 0; x < 8; ++x) {
                for (int y = 0; y < 8; ++y) {
                    double cu = (u == 0) ? (1.0 / kSqrt2) : 1.0;
                    double cv = (v == 0) ? (1.0 / kSqrt2) : 1.0;
                    
                    double cos_u = std::cos((2 * x + 1) * u * M_PI / 16.0);
                    double cos_v = std::cos((2 * y + 1) * v * M_PI / 16.0);
                    
                    sum += input[x * 8 + y] * cu * cv * cos_u * cos_v;
                }
            }
            
            output[u * 8 + v] = static_cast<int16_t>(sum * kScale * 4.0);  // Scale for integers
        }
    }
}

void Transform::inverseDCT8x8(const int16_t* input, int8_t* output) {
    const double kScale = 0.25;  // Inverse scale
    const double kSqrt2 = 1.41421356237;
    
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            double sum = 0.0;
            
            for (int u = 0; u < 8; ++u) {
                for (int v = 0; v < 8; ++v) {
                    double cu = (u == 0) ? (1.0 / kSqrt2) : 1.0;
                    double cv = (v == 0) ? (1.0 / kSqrt2) : 1.0;
                    
                    double cos_u = std::cos((2 * x + 1) * u * M_PI / 16.0);
                    double cos_v = std::cos((2 * y + 1) * v * M_PI / 16.0);
                    
                    sum += input[u * 8 + v] * cu * cv * cos_u * cos_v;
                }
            }
            
            output[x * 8 + y] = static_cast<int8_t>(std::round(sum * kScale));
        }
    }
}

// Simplified ADST (Asymmetric DST)
void Transform::forwardADST8x8(const int8_t* input, int16_t* output) {
    // ADST[i,j] = sum_x sum_y input[x,y] * sin((2x+1)*(i+1)*pi/17) * sin((2y+1)*(j+1)*pi/17)
    
    const double kScale = 1.0 / 8.0;
    
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            double sum = 0.0;
            
            for (int x = 0; x < 8; ++x) {
                for (int y = 0; y < 8; ++y) {
                    double sin_u = std::sin((2 * x + 1) * (u + 1) * M_PI / 17.0);
                    double sin_v = std::sin((2 * y + 1) * (v + 1) * M_PI / 17.0);
                    
                    sum += input[x * 8 + y] * sin_u * sin_v;
                }
            }
            
            output[u * 8 + v] = static_cast<int16_t>(sum * kScale * 4.0);
        }
    }
}

void Transform::inverseADST8x8(const int16_t* input, int8_t* output) {
    const double kScale = 0.25;
    
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            double sum = 0.0;
            
            for (int u = 0; u < 8; ++u) {
                for (int v = 0; v < 8; ++v) {
                    double sin_u = std::sin((2 * x + 1) * (u + 1) * M_PI / 17.0);
                    double sin_v = std::sin((2 * y + 1) * (v + 1) * M_PI / 17.0);
                    
                    sum += input[u * 8 + v] * sin_u * sin_v;
                }
            }
            
            output[x * 8 + y] = static_cast<int8_t>(std::round(sum * kScale));
        }
    }
}

TransformType Transform::selectTransform(const int8_t* block, double lambda) {
    // Simple RD selection: try each transform, estimate bits
    int16_t coeffs_none[64];
    int16_t coeffs_dct[64];
    int16_t coeffs_adst[64];
    
    // None: just copy
    for (int i = 0; i < 64; ++i) {
        coeffs_none[i] = block[i];
    }
    
    // DCT
    forwardDCT8x8(block, coeffs_dct);
    
    // ADST
    forwardADST8x8(block, coeffs_adst);
    
    // Estimate cost (simple: sum of absolute values = rate proxy)
    auto estimateCost = [](const int16_t* coeffs) {
        int64_t cost = 0;
        for (int i = 0; i < 64; ++i) {
            cost += std::abs(coeffs[i]);
        }
        return cost;
    };
    
    int64_t cost_none = estimateCost(coeffs_none);
    int64_t cost_dct = estimateCost(coeffs_dct);
    int64_t cost_adst = estimateCost(coeffs_adst);
    
    // Select minimum
    if (cost_dct <= cost_none && cost_dct <= cost_adst) {
        return TransformType::DCT;
    } else if (cost_adst <= cost_none) {
        return TransformType::ADST;
    } else {
        return TransformType::NONE;
    }
}

void Transform::zigzagScan(const int16_t* input, int16_t* output) {
    for (int i = 0; i < 64; ++i) {
        output[i] = input[kZigzagIndices[i]];
    }
}

void Transform::inverseZigzagScan(const int16_t* input, int16_t* output) {
    for (int i = 0; i < 64; ++i) {
        output[kZigzagIndices[i]] = input[i];
    }
}

} // namespace wcodec

