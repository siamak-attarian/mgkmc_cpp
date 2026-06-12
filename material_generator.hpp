#ifndef MATERIAL_GENERATOR_HPP
#define MATERIAL_GENERATOR_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <complex>
#include <stdexcept>
#include <iostream>
#include "pocketfft_hdronly.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------------------------------------------
// 1. NPY file reader
// -------------------------------------------------------------
inline std::vector<double> read_npy_double(const std::string& filepath, std::vector<size_t>& shape) {
    std::ifstream fs(filepath, std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("Failed to open npy file: " + filepath);
    }
    
    char magic[6];
    fs.read(magic, 6);
    if (magic[0] != (char)0x93 || magic[1] != 'N' || magic[2] != 'U' || 
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        throw std::runtime_error("Invalid npy magic bytes");
    }
    
    char major = 0;
    char minor = 0;
    fs.read(&major, 1);
    fs.read(&minor, 1);
    
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        fs.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2) {
        fs.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported npy version: " + std::to_string((int)major));
    }
    
    std::string header(header_len, ' ');
    fs.read(&header[0], header_len);
    
    // Parse shape
    size_t shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        shape_pos = header.find("\"shape\"");
    }
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("Could not find 'shape' in npy header");
    }
    
    size_t colon_pos = header.find(":", shape_pos);
    size_t open_paren = header.find("(", colon_pos);
    size_t close_paren = header.find(")", open_paren);
    if (open_paren == std::string::npos || close_paren == std::string::npos) {
        throw std::runtime_error("Invalid shape format in npy header");
    }
    
    std::string shape_str = header.substr(open_paren + 1, close_paren - open_paren - 1);
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());
    
    shape.clear();
    std::stringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            shape.push_back(std::stoull(token));
        }
    }
    
    // Parse descr to identify dtype
    std::string descr = "";
    size_t descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) {
        descr_pos = header.find("\"descr\"");
    }
    if (descr_pos != std::string::npos) {
        size_t colon = header.find(":", descr_pos);
        size_t quote1 = header.find_first_of("'\"", colon);
        size_t quote2 = header.find_first_of("'\"", quote1 + 1);
        descr = header.substr(quote1 + 1, quote2 - quote1 - 1);
    }
    
    size_t num_elements = 1;
    for (size_t s : shape) {
        num_elements *= s;
    }
    
    std::vector<double> data(num_elements);
    if (descr == "<f4" || descr == "f4" || descr == "|f4") {
        // float32 conversion to double
        std::vector<float> temp(num_elements);
        fs.read(reinterpret_cast<char*>(temp.data()), num_elements * sizeof(float));
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = temp[i];
        }
    } else if (descr == "<f8" || descr == "f8" || descr == "|f8" || descr.empty()) {
        fs.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(double));
    } else {
        throw std::runtime_error("Unsupported dtype in npy header: " + descr);
    }
    
    return data;
}

// -------------------------------------------------------------
// 2. Correlated field generator (matching SciPy gaussian_filter wrap)
// -------------------------------------------------------------
inline double box_muller_normal(std::mt19937& rng) {
    double u1 = (rng() - rng.min()) / double(rng.max() - rng.min() + 1.0);
    double u2 = (rng() - rng.min()) / double(rng.max() - rng.min() + 1.0);
    while (u1 <= 1e-15) {
        u1 = (rng() - rng.min()) / double(rng.max() - rng.min() + 1.0);
    }
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

inline std::vector<double> generate_correlated_field_cpp(
    int nx, int ny, int nz, bool is_3d,
    double mean_target, double std_target, double corr_sigma,
    bool has_clip_min, double clip_min,
    bool has_clip_max, double clip_max,
    int seed
) {
    int N = nx * ny * nz;
    std::vector<double> field(N);
    
    // 1. Generate normal random noise
    std::mt19937 rng(seed);
    for (int i = 0; i < N; ++i) {
        field[i] = box_muller_normal(rng);
    }
    
    // 2. Perform Gaussian filtering in frequency space
    std::vector<std::complex<double>> complex_hat(N);
    
    if (!is_3d) {
        // 2D FFT
        std::vector<std::complex<double>> temp(N);
        for (int i = 0; i < N; ++i) temp[i] = std::complex<double>(field[i], 0.0);
        
        pocketfft::shape_t shape{(size_t)nx, (size_t)ny};
        pocketfft::stride_t strides{(ptrdiff_t)(ny * sizeof(std::complex<double>)), (ptrdiff_t)sizeof(std::complex<double>)};
        pocketfft::shape_t axes{0, 1};
        pocketfft::c2c(shape, strides, strides, axes, pocketfft::FORWARD, temp.data(), complex_hat.data(), 1.0 / std::sqrt(N));
        
        // Compute 2D frequencies
        std::vector<double> fx(nx);
        int lim_x = (nx + 1) / 2;
        for (int i = 0; i < nx; ++i) fx[i] = (i < lim_x ? i : i - nx) / double(nx);
        
        std::vector<double> fy(ny);
        int lim_y = (ny + 1) / 2;
        for (int i = 0; i < ny; ++i) fy[i] = (i < lim_y ? i : i - ny) / double(ny);
        
        // Apply Gaussian filter multiplier: exp(-2 * pi^2 * sigma^2 * f^2)
        double coeff = -2.0 * M_PI * M_PI * corr_sigma * corr_sigma;
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                int idx = x * ny + y;
                double f2 = fx[x] * fx[x] + fy[y] * fy[y];
                complex_hat[idx] *= std::exp(coeff * f2);
            }
        }
        
        // 2D IFFT
        pocketfft::c2c(shape, strides, strides, axes, pocketfft::BACKWARD, complex_hat.data(), temp.data(), 1.0 / std::sqrt(N));
        for (int i = 0; i < N; ++i) field[i] = temp[i].real();
        
    } else {
        // 3D FFT
        std::vector<std::complex<double>> temp(N);
        for (int i = 0; i < N; ++i) temp[i] = std::complex<double>(field[i], 0.0);
        
        pocketfft::shape_t shape{(size_t)nx, (size_t)ny, (size_t)nz};
        pocketfft::stride_t strides{
            (ptrdiff_t)(ny * nz * sizeof(std::complex<double>)), 
            (ptrdiff_t)(nz * sizeof(std::complex<double>)), 
            (ptrdiff_t)sizeof(std::complex<double>)
        };
        pocketfft::shape_t axes{0, 1, 2};
        pocketfft::c2c(shape, strides, strides, axes, pocketfft::FORWARD, temp.data(), complex_hat.data(), 1.0 / std::sqrt(N));
        
        // Compute 3D frequencies
        std::vector<double> fx(nx);
        int lim_x = (nx + 1) / 2;
        for (int i = 0; i < nx; ++i) fx[i] = (i < lim_x ? i : i - nx) / double(nx);
        
        std::vector<double> fy(ny);
        int lim_y = (ny + 1) / 2;
        for (int i = 0; i < ny; ++i) fy[i] = (i < lim_y ? i : i - ny) / double(ny);
        
        std::vector<double> fz(nz);
        int lim_z = (nz + 1) / 2;
        for (int i = 0; i < nz; ++i) fz[i] = (i < lim_z ? i : i - nz) / double(nz);
        
        // Apply filter
        double coeff = -2.0 * M_PI * M_PI * corr_sigma * corr_sigma;
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    int idx = x * ny * nz + y * nz + z;
                    double f2 = fx[x] * fx[x] + fy[y] * fy[y] + fz[z] * fz[z];
                    complex_hat[idx] *= std::exp(coeff * f2);
                }
            }
        }
        
        // 3D IFFT
        pocketfft::c2c(shape, strides, strides, axes, pocketfft::BACKWARD, complex_hat.data(), temp.data(), 1.0 / std::sqrt(N));
        for (int i = 0; i < N; ++i) field[i] = temp[i].real();
    }
    
    // 3. Normalize field to mean=0, std=1
    double mean = 0.0;
    for (double val : field) mean += val;
    mean /= N;
    
    double var = 0.0;
    for (double val : field) var += (val - mean) * (val - mean);
    double std_dev = std::sqrt(var / N);
    if (std_dev < 1e-15) std_dev = 1.0;
    
    for (int i = 0; i < N; ++i) {
        field[i] = (field[i] - mean) / std_dev;
    }
    
    // 4. Scale to target mean and std
    for (int i = 0; i < N; ++i) {
        field[i] = mean_target + std_target * field[i];
    }
    
    // 5. Apply clipping
    if (has_clip_min) {
        for (int i = 0; i < N; ++i) {
            if (field[i] < clip_min) field[i] = clip_min;
        }
    }
    if (has_clip_max) {
        for (int i = 0; i < N; ++i) {
            if (field[i] > clip_max) field[i] = clip_max;
        }
    }
    
    return field;
}

#endif // MATERIAL_GENERATOR_HPP
