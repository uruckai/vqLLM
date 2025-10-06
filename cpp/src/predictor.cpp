/**
 * @file predictor.cpp
 * @brief Implementation of predictive coding
 */

#include "wcodec/predictor.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>

namespace wcodec {

PredictorMode Predictor::selectMode(
    const int8_t* tile,
    size_t rows,
    size_t cols,
    const int8_t* left_col,
    const int8_t* top_row
) {
    // If no neighbors, no prediction
    if (!left_col && !top_row) {
        return PredictorMode::NONE;
    }

    size_t size = rows * cols;
    std::vector<int8_t> pred(size);
    std::vector<int8_t> residual(size);
    
    PredictorMode best_mode = PredictorMode::NONE;
    size_t best_cost = std::numeric_limits<size_t>::max();

    // Try each available mode
    std::vector<PredictorMode> modes = {PredictorMode::NONE};
    if (left_col) modes.push_back(PredictorMode::LEFT);
    if (top_row) modes.push_back(PredictorMode::TOP);
    if (left_col && top_row) {
        modes.push_back(PredictorMode::AVG);
        modes.push_back(PredictorMode::PLANAR);
    }

    for (auto mode : modes) {
        // Compute prediction
        predict(tile, residual.data(), rows, cols, left_col, top_row, mode);
        
        // Estimate cost (simple: sum of absolute residuals)
        size_t cost = estimateBits(residual.data(), size);
        
        if (cost < best_cost) {
            best_cost = cost;
            best_mode = mode;
        }
    }

    return best_mode;
}

void Predictor::predict(
    const int8_t* tile,
    int8_t* residual,
    size_t rows,
    size_t cols,
    const int8_t* left_col,
    const int8_t* top_row,
    PredictorMode mode
) {
    size_t size = rows * cols;
    std::vector<int8_t> pred(size);

    // Compute prediction based on mode
    switch (mode) {
        case PredictorMode::NONE:
            std::memset(pred.data(), 0, size);
            break;
        case PredictorMode::LEFT:
            predictLeft(left_col, pred.data(), rows, cols);
            break;
        case PredictorMode::TOP:
            predictTop(top_row, pred.data(), rows, cols);
            break;
        case PredictorMode::AVG:
            predictAvg(left_col, top_row, pred.data(), rows, cols);
            break;
        case PredictorMode::PLANAR:
            predictPlanar(left_col, top_row, pred.data(), rows, cols);
            break;
    }

    // Compute residual
    for (size_t i = 0; i < size; ++i) {
        residual[i] = tile[i] - pred[i];
    }
}

void Predictor::reconstruct(
    const int8_t* residual,
    int8_t* tile,
    size_t rows,
    size_t cols,
    const int8_t* left_col,
    const int8_t* top_row,
    PredictorMode mode
) {
    size_t size = rows * cols;
    std::vector<int8_t> pred(size);

    // Compute same prediction
    switch (mode) {
        case PredictorMode::NONE:
            std::memset(pred.data(), 0, size);
            break;
        case PredictorMode::LEFT:
            predictLeft(left_col, pred.data(), rows, cols);
            break;
        case PredictorMode::TOP:
            predictTop(top_row, pred.data(), rows, cols);
            break;
        case PredictorMode::AVG:
            predictAvg(left_col, top_row, pred.data(), rows, cols);
            break;
        case PredictorMode::PLANAR:
            predictPlanar(left_col, top_row, pred.data(), rows, cols);
            break;
    }

    // Reconstruct: tile = residual + prediction
    for (size_t i = 0; i < size; ++i) {
        tile[i] = residual[i] + pred[i];
    }
}

// Private implementations

void Predictor::predictLeft(const int8_t* left_col, int8_t* pred, 
                           size_t rows, size_t cols) {
    if (!left_col) {
        std::memset(pred, 0, rows * cols);
        return;
    }
    
    // Each row gets value from left neighbor
    for (size_t r = 0; r < rows; ++r) {
        int8_t val = left_col[r];
        for (size_t c = 0; c < cols; ++c) {
            pred[r * cols + c] = val;
        }
    }
}

void Predictor::predictTop(const int8_t* top_row, int8_t* pred,
                          size_t rows, size_t cols) {
    if (!top_row) {
        std::memset(pred, 0, rows * cols);
        return;
    }
    
    // Each column gets value from top neighbor
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            pred[r * cols + c] = top_row[c];
        }
    }
}

void Predictor::predictAvg(const int8_t* left_col, const int8_t* top_row,
                          int8_t* pred, size_t rows, size_t cols) {
    if (!left_col || !top_row) {
        std::memset(pred, 0, rows * cols);
        return;
    }
    
    // Average of left and top
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            int16_t left = left_col[r];
            int16_t top = top_row[c];
            pred[r * cols + c] = static_cast<int8_t>((left + top) / 2);
        }
    }
}

void Predictor::predictPlanar(const int8_t* left_col, const int8_t* top_row,
                             int8_t* pred, size_t rows, size_t cols) {
    if (!left_col || !top_row) {
        std::memset(pred, 0, rows * cols);
        return;
    }
    
    // Simple planar: linear interpolation from edges
    // pred[r,c] = (left[r] * (cols-c) + top[c] * (rows-r)) / (rows + cols - r - c)
    // Simplified version for speed:
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            int16_t left = left_col[r];
            int16_t top = top_row[c];
            // Weighted average based on position
            int16_t w_left = cols - c;
            int16_t w_top = rows - r;
            pred[r * cols + c] = static_cast<int8_t>(
                (left * w_left + top * w_top) / (w_left + w_top)
            );
        }
    }
}

size_t Predictor::estimateBits(const int8_t* residual, size_t size) {
    // Simple entropy estimate: sum of absolute values
    // (More residual energy = more bits needed)
    size_t cost = 0;
    for (size_t i = 0; i < size; ++i) {
        cost += std::abs(residual[i]);
    }
    return cost;
}

} // namespace wcodec

