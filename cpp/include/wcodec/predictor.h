/**
 * @file predictor.h
 * @brief Predictive coding for weight tiles (intra-prediction)
 */

#pragma once

#include "types.h"
#include <vector>

namespace wcodec {

class Predictor {
public:
    /**
     * @brief Select best predictor mode for a tile
     * @param tile Input tile data (row-major)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param left_col Left neighbor column (nullptr if not available)
     * @param top_row Top neighbor row (nullptr if not available)
     * @return Best predictor mode
     */
    static PredictorMode selectMode(
        const int8_t* tile,
        size_t rows,
        size_t cols,
        const int8_t* left_col,
        const int8_t* top_row
    );

    /**
     * @brief Predict tile values and compute residual
     * @param tile Input tile data
     * @param residual Output residual (tile - prediction)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param left_col Left neighbor column
     * @param top_row Top neighbor row
     * @param mode Predictor mode to use
     */
    static void predict(
        const int8_t* tile,
        int8_t* residual,
        size_t rows,
        size_t cols,
        const int8_t* left_col,
        const int8_t* top_row,
        PredictorMode mode
    );

    /**
     * @brief Reconstruct tile from residual and prediction
     * @param residual Input residual
     * @param tile Output reconstructed tile
     * @param rows Number of rows
     * @param cols Number of columns
     * @param left_col Left neighbor column
     * @param top_row Top neighbor row
     * @param mode Predictor mode used
     */
    static void reconstruct(
        const int8_t* residual,
        int8_t* tile,
        size_t rows,
        size_t cols,
        const int8_t* left_col,
        const int8_t* top_row,
        PredictorMode mode
    );

private:
    // Individual predictor implementations
    static void predictLeft(const int8_t* left_col, int8_t* pred, size_t rows, size_t cols);
    static void predictTop(const int8_t* top_row, int8_t* pred, size_t rows, size_t cols);
    static void predictAvg(const int8_t* left_col, const int8_t* top_row, 
                          int8_t* pred, size_t rows, size_t cols);
    static void predictPlanar(const int8_t* left_col, const int8_t* top_row,
                             int8_t* pred, size_t rows, size_t cols);
    
    // Estimate bits needed for residual (simple entropy estimate)
    static size_t estimateBits(const int8_t* residual, size_t size);
};

} // namespace wcodec

