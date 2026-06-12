#ifndef LINEAR_ELASTIC_SOLVER_HPP
#define LINEAR_ELASTIC_SOLVER_HPP

#include <vector>
#include <array>
#include <complex>
#include <string>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "pocketfft_hdronly.h"

// Set up M_PI constant if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 4th order isotropic Green operator tensor representation: [2][2][2][2]
using GreenTensor = std::array<std::array<std::array<std::array<double, 2>, 2>, 2>, 2>;

// Helper function to calculate standard DFT frequencies (matching np.fft.fftfreq)
inline std::vector<double> fftfreq(int n, double d) {
    std::vector<double> freqs(n);
    int limit = (n + 1) / 2;
    for (int i = 0; i < n; ++i) {
        if (i < limit) {
            freqs[i] = static_cast<double>(i) / (n * d);
        } else {
            freqs[i] = static_cast<double>(i - n) / (n * d);
        }
    }
    return freqs;
}

// Lamé parameters for 2D isotropic elasticity
inline std::pair<double, double> compute_lame_2d(double E, double nu, const std::string& plane_mode) {
    double mu = E / (2.0 * (1.0 + nu));
    double lam;
    if (plane_mode == "plane_stress") {
        lam = E * nu / (1.0 - nu * nu);
    } else { // plane_strain
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    }
    return {lam, mu};
}

// Compute stress field from strain field: σ = 2μ ε + λ tr(ε) I
inline std::vector<Eigen::Matrix2d> stress_from_strain_2d(
    const std::vector<Eigen::Matrix2d>& eps,
    const std::vector<double>& lam,
    const std::vector<double>& mu,
    const std::vector<Eigen::Matrix2d>& eps_plastic = {}) 
{
    int N = eps.size();
    std::vector<Eigen::Matrix2d> sig(N);
    if (eps_plastic.empty()) {
        for (int i = 0; i < N; ++i) {
            sig[i] = 2.0 * mu[i] * eps[i] + lam[i] * eps[i].trace() * Eigen::Matrix2d::Identity();
        }
    } else {
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix2d eps_eff = eps[i] - eps_plastic[i];
            sig[i] = 2.0 * mu[i] * eps_eff + lam[i] * eps_eff.trace() * Eigen::Matrix2d::Identity();
        }
    }
    return sig;
}

// 2D isotropic Green operator computation
inline std::vector<GreenTensor> green_operator_2d(
    int nx, int ny,
    const std::vector<double>& kx, const std::vector<double>& ky,
    double lam0, double mu0) 
{
    std::vector<GreenTensor> Gamma(nx * ny);
    double A = 1.0 / (4.0 * mu0);
    double B = (lam0 + mu0) / (mu0 * (lam0 + 2.0 * mu0));

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            int idx = x * ny + y;
            double kx_val = kx[idx];
            double ky_val = ky[idx];
            double k2 = kx_val * kx_val + ky_val * ky_val;

            if (x == 0 && y == 0) {
                // Gamma[0,0] = 0
                for (int k = 0; k < 2; ++k)
                    for (int h = 0; h < 2; ++h)
                        for (int i = 0; i < 2; ++i)
                            for (int j = 0; j < 2; ++j)
                                Gamma[idx][k][h][i][j] = 0.0;
                continue;
            }

            double q[2] = {kx_val, ky_val};

            for (int k = 0; k < 2; ++k) {
                for (int h = 0; h < 2; ++h) {
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            double term1 = 0.0;
                            if (k == i) term1 += q[h] * q[j];
                            if (h == i) term1 += q[k] * q[j];
                            if (k == j) term1 += q[h] * q[i];
                            if (h == j) term1 += q[k] * q[i];
                            term1 = A * term1 / k2;

                            double term2 = B * (q[k] * q[h] * q[i] * q[j]) / (k2 * k2);

                            Gamma[idx][k][h][i][j] = term1 - term2;
                        }
                    }
                }
            }
        }
    }
    return Gamma;
}

// 2D FFT helper using pocketfft (C2C with ortho-normalization)
inline void fft2d(int nx, int ny, const std::vector<double>& real_field, std::vector<std::complex<double>>& complex_hat) {
    std::vector<std::complex<double>> temp(nx * ny);
    for (int i = 0; i < nx * ny; ++i) {
        temp[i] = std::complex<double>(real_field[i], 0.0);
    }
    pocketfft::shape_t shape{(size_t)nx, (size_t)ny};
    pocketfft::stride_t strides{(ptrdiff_t)(ny * sizeof(std::complex<double>)), (ptrdiff_t)sizeof(std::complex<double>)};
    pocketfft::shape_t axes{0, 1};
    pocketfft::c2c(shape, strides, strides, axes, pocketfft::FORWARD, temp.data(), complex_hat.data(), 1.0 / std::sqrt(nx * ny));
}

// 2D IFFT helper using pocketfft (C2C with ortho-normalization)
inline void ifft2d(int nx, int ny, const std::vector<std::complex<double>>& complex_hat, std::vector<double>& real_field) {
    std::vector<std::complex<double>> temp(nx * ny);
    pocketfft::shape_t shape{(size_t)nx, (size_t)ny};
    pocketfft::stride_t strides{(ptrdiff_t)(ny * sizeof(std::complex<double>)), (ptrdiff_t)sizeof(std::complex<double>)};
    pocketfft::shape_t axes{0, 1};
    pocketfft::c2c(shape, strides, strides, axes, pocketfft::BACKWARD, complex_hat.data(), temp.data(), 1.0 / std::sqrt(nx * ny));
    for (int i = 0; i < nx * ny; ++i) {
        real_field[i] = temp[i].real();
    }
}

// 2D Spectral Lippmann-Schwinger solver
inline void spectral_solver_2d(
    int nx, int ny,
    const std::vector<double>& E,
    const std::vector<double>& nu,
    const Eigen::Matrix2d& eps_bar,
    const std::string& plane_mode,
    const std::vector<GreenTensor>& Gamma,
    std::vector<Eigen::Matrix2d>& eps,
    std::vector<Eigen::Matrix2d>& sig_out,
    Eigen::Matrix2d& eps_macro,
    Eigen::Matrix2d& sig_macro,
    int max_iter = 200,
    double tol = 1e-6,
    bool verbose = false,
    const std::vector<Eigen::Matrix2d>& eps_plastic = {}) 
{
    int N = nx * ny;
    double E_avg = 0.0, nu_avg = 0.0;
    for (int i = 0; i < N; ++i) {
        E_avg += E[i];
        nu_avg += nu[i];
    }
    E_avg /= N;
    nu_avg /= N;

    // Compute Lamé constants for the reference and actual media
    auto [lam0, mu0] = compute_lame_2d(E_avg, nu_avg, plane_mode);
    std::vector<double> lam(N), mu(N);
    for (int i = 0; i < N; ++i) {
        auto [l, m] = compute_lame_2d(E[i], nu[i], plane_mode);
        lam[i] = l;
        mu[i] = m;
    }

    // Initialize eps field if not already warm-started
    if (eps.size() == (size_t)N) {
        // Warm start: adjust mean to match eps_bar
        Eigen::Matrix2d eps_mean = Eigen::Matrix2d::Zero();
        for (int i = 0; i < N; ++i) {
            eps_mean += eps[i];
        }
        eps_mean /= N;
        Eigen::Matrix2d mean_diff = eps_bar - eps_mean;
        for (int i = 0; i < N; ++i) {
            eps[i] += mean_diff;
        }
    } else {
        eps.assign(N, eps_bar);
    }

    // Auxiliary buffers for FFT/IFFT
    std::vector<double> tau_comp(N);
    std::vector<std::complex<double>> tau_hat_xx(N), tau_hat_xy(N), tau_hat_yx(N), tau_hat_yy(N);
    std::vector<std::complex<double>> eps_tilde_hat_xx(N), eps_tilde_hat_xy(N), eps_tilde_hat_yx(N), eps_tilde_hat_yy(N);
    std::vector<double> eps_tilde_xx(N), eps_tilde_xy(N), eps_tilde_yx(N), eps_tilde_yy(N);

    // Lippmann-Schwinger loop
    for (int it = 0; it < max_iter; ++it) {
        // 1. Stress calculation: sig = C : eps
        std::vector<Eigen::Matrix2d> sig = stress_from_strain_2d(eps, lam, mu, eps_plastic);

        // 2. Reference stress: sig0 = C0 : eps
        // Since C0 is homogeneous (lam0, mu0), compute polarization stress: tau = sig - sig0
        std::vector<Eigen::Matrix2d> tau(N);
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix2d sig0 = 2.0 * mu0 * eps[i] + lam0 * eps[i].trace() * Eigen::Matrix2d::Identity();
            tau[i] = sig[i] - sig0;
        }

        // 3. FFT of each polarization stress component
        // Component xx
        for (int i = 0; i < N; ++i) tau_comp[i] = tau[i](0, 0);
        fft2d(nx, ny, tau_comp, tau_hat_xx);
        // Component xy
        for (int i = 0; i < N; ++i) tau_comp[i] = tau[i](0, 1);
        fft2d(nx, ny, tau_comp, tau_hat_xy);
        // Component yx
        for (int i = 0; i < N; ++i) tau_comp[i] = tau[i](1, 0);
        fft2d(nx, ny, tau_comp, tau_hat_yx);
        // Component yy
        for (int i = 0; i < N; ++i) tau_comp[i] = tau[i](1, 1);
        fft2d(nx, ny, tau_comp, tau_hat_yy);

        // 4. Multiply with Green operator in Fourier space
        // eps_tilde_hat = - Gamma : tau_hat
        for (int i = 0; i < N; ++i) {
            std::complex<double> tau_hat_node[2][2] = {
                {tau_hat_xx[i], tau_hat_xy[i]},
                {tau_hat_yx[i], tau_hat_yy[i]}
            };

            std::complex<double> res[2][2] = { {0.0, 0.0}, {0.0, 0.0} };
            const auto& G = Gamma[i];

            for (int k = 0; k < 2; ++k) {
                for (int h = 0; h < 2; ++h) {
                    std::complex<double> val = 0.0;
                    for (int r = 0; r < 2; ++r) {
                        for (int s = 0; s < 2; ++s) {
                            val += G[k][h][r][s] * tau_hat_node[r][s];
                        }
                    }
                    res[k][h] = -val;
                }
            }
            eps_tilde_hat_xx[i] = res[0][0];
            eps_tilde_hat_xy[i] = res[0][1];
            eps_tilde_hat_yx[i] = res[1][0];
            eps_tilde_hat_yy[i] = res[1][1];
        }

        // 5. Inverse FFT of eps_tilde_hat
        ifft2d(nx, ny, eps_tilde_hat_xx, eps_tilde_xx);
        ifft2d(nx, ny, eps_tilde_hat_xy, eps_tilde_xy);
        ifft2d(nx, ny, eps_tilde_hat_yx, eps_tilde_yx);
        ifft2d(nx, ny, eps_tilde_hat_yy, eps_tilde_yy);

        // 6. Subtract mean to enforce periodic boundary fluctuation zero-mean
        double mean_xx = 0.0, mean_xy = 0.0, mean_yx = 0.0, mean_yy = 0.0;
        for (int i = 0; i < N; ++i) {
            mean_xx += eps_tilde_xx[i];
            mean_xy += eps_tilde_xy[i];
            mean_yx += eps_tilde_yx[i];
            mean_yy += eps_tilde_yy[i];
        }
        mean_xx /= N; mean_xy /= N; mean_yx /= N; mean_yy /= N;

        // 7. Update strain field: eps_new = eps_bar + eps_tilde
        std::vector<Eigen::Matrix2d> eps_new(N);
        double diff_sq = 0.0;
        double norm_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            eps_new[i](0, 0) = eps_bar(0, 0) + (eps_tilde_xx[i] - mean_xx);
            eps_new[i](0, 1) = eps_bar(0, 1) + (eps_tilde_xy[i] - mean_xy);
            eps_new[i](1, 0) = eps_bar(1, 0) + (eps_tilde_yx[i] - mean_yx);
            eps_new[i](1, 1) = eps_bar(1, 1) + (eps_tilde_yy[i] - mean_yy);

            diff_sq += (eps_new[i] - eps[i]).squaredNorm();
            norm_sq += eps[i].squaredNorm();
        }

        double diff = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-20);
        eps = eps_new;

        if (verbose && it % 10 == 0) {
            std::cout << "  Iter " << it << ": diff = " << diff << std::endl;
        }

        if (diff < tol) {
            break;
        }
    }

    // Compute final stresses and macros
    sig_out = stress_from_strain_2d(eps, lam, mu, eps_plastic);
    eps_macro = Eigen::Matrix2d::Zero();
    sig_macro = Eigen::Matrix2d::Zero();
    for (int i = 0; i < N; ++i) {
        eps_macro += eps[i];
        sig_macro += sig_out[i];
    }
    eps_macro /= N;
    sig_macro /= N;
}

// 4th order isotropic Green operator tensor representation for 3D: [3][3][3][3]
using GreenTensor3D = std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 3>;

// Lamé parameters for 3D isotropic elasticity
inline std::pair<double, double> compute_lame_3d(double E, double nu) {
    double mu = E / (2.0 * (1.0 + nu));
    double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    return {lam, mu};
}

// Compute stress field from strain field: σ = 2μ ε + λ tr(ε) I
inline std::vector<Eigen::Matrix3d> stress_from_strain_3d(
    const std::vector<Eigen::Matrix3d>& eps,
    const std::vector<double>& lam,
    const std::vector<double>& mu) 
{
    int N = eps.size();
    std::vector<Eigen::Matrix3d> sig(N);
    for (int i = 0; i < N; ++i) {
        sig[i] = 2.0 * mu[i] * eps[i] + lam[i] * eps[i].trace() * Eigen::Matrix3d::Identity();
    }
    return sig;
}

// 3D isotropic Green operator computation
inline std::vector<GreenTensor3D> green_operator_3d(
    int nx, int ny, int nz,
    const std::vector<double>& kx, const std::vector<double>& ky, const std::vector<double>& kz,
    double lam0, double mu0) 
{
    int N = nx * ny * nz;
    std::vector<GreenTensor3D> Gamma(N);
    double A = 1.0 / (4.0 * mu0);
    double B = (lam0 + mu0) / (mu0 * (lam0 + 2.0 * mu0));

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                int idx = x * (ny * nz) + y * nz + z;
                double kx_val = kx[idx];
                double ky_val = ky[idx];
                double kz_val = kz[idx];
                double k2 = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                if (x == 0 && y == 0 && z == 0) {
                    for (int k = 0; k < 3; ++k)
                        for (int h = 0; h < 3; ++h)
                            for (int i = 0; i < 3; ++i)
                                for (int j = 0; j < 3; ++j)
                                    Gamma[idx][k][h][i][j] = 0.0;
                    continue;
                }

                double q[3] = {kx_val, ky_val, kz_val};

                for (int k = 0; k < 3; ++k) {
                    for (int h = 0; h < 3; ++h) {
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                double term1 = 0.0;
                                if (k == i) term1 += q[h] * q[j];
                                if (h == i) term1 += q[k] * q[j];
                                if (k == j) term1 += q[h] * q[i];
                                if (h == j) term1 += q[k] * q[i];
                                term1 = A * term1 / k2;

                                double term2 = B * (q[k] * q[h] * q[i] * q[j]) / (k2 * k2);

                                Gamma[idx][k][h][i][j] = term1 - term2;
                            }
                        }
                    }
                }
            }
        }
    }
    return Gamma;
}

// 3D FFT helper using pocketfft (C2C with ortho-normalization)
inline void fft3d(int nx, int ny, int nz, const std::vector<double>& real_field, std::vector<std::complex<double>>& complex_hat) {
    std::vector<std::complex<double>> temp(nx * ny * nz);
    for (int i = 0; i < nx * ny * nz; ++i) {
        temp[i] = std::complex<double>(real_field[i], 0.0);
    }
    pocketfft::shape_t shape{(size_t)nx, (size_t)ny, (size_t)nz};
    pocketfft::stride_t strides{
        (ptrdiff_t)(ny * nz * sizeof(std::complex<double>)),
        (ptrdiff_t)(nz * sizeof(std::complex<double>)),
        (ptrdiff_t)sizeof(std::complex<double>)
    };
    pocketfft::shape_t axes{0, 1, 2};
    pocketfft::c2c(shape, strides, strides, axes, pocketfft::FORWARD, temp.data(), complex_hat.data(), 1.0 / std::sqrt(nx * ny * nz));
}

// 3D IFFT helper using pocketfft (C2C with ortho-normalization)
inline void ifft3d(int nx, int ny, int nz, const std::vector<std::complex<double>>& complex_hat, std::vector<double>& real_field) {
    std::vector<std::complex<double>> temp(nx * ny * nz);
    pocketfft::shape_t shape{(size_t)nx, (size_t)ny, (size_t)nz};
    pocketfft::stride_t strides{
        (ptrdiff_t)(ny * nz * sizeof(std::complex<double>)),
        (ptrdiff_t)(nz * sizeof(std::complex<double>)),
        (ptrdiff_t)sizeof(std::complex<double>)
    };
    pocketfft::shape_t axes{0, 1, 2};
    pocketfft::c2c(shape, strides, strides, axes, pocketfft::BACKWARD, complex_hat.data(), temp.data(), 1.0 / std::sqrt(nx * ny * nz));
    for (int i = 0; i < nx * ny * nz; ++i) {
        real_field[i] = temp[i].real();
    }
}

// 3D Spectral Lippmann-Schwinger solver
inline void spectral_solver_3d(
    int nx, int ny, int nz,
    const std::vector<double>& E,
    const std::vector<double>& nu,
    const Eigen::Matrix3d& eps_bar,
    const std::vector<GreenTensor3D>& Gamma,
    std::vector<Eigen::Matrix3d>& eps,
    std::vector<Eigen::Matrix3d>& sig_out,
    Eigen::Matrix3d& eps_macro,
    Eigen::Matrix3d& sig_macro,
    int max_iter = 200,
    double tol = 1e-6,
    bool verbose = false) 
{
    int N = nx * ny * nz;
    double E_avg = 0.0, nu_avg = 0.0;
    for (int i = 0; i < N; ++i) {
        E_avg += E[i];
        nu_avg += nu[i];
    }
    E_avg /= N;
    nu_avg /= N;

    // Compute Lamé constants for the reference and actual media
    auto [lam0, mu0] = compute_lame_3d(E_avg, nu_avg);
    std::vector<double> lam(N), mu(N);
    for (int i = 0; i < N; ++i) {
        auto [l, m] = compute_lame_3d(E[i], nu[i]);
        lam[i] = l;
        mu[i] = m;
    }

    // Initialize eps field if not already warm-started
    if (eps.size() == (size_t)N) {
        // Warm start: adjust mean to match eps_bar
        Eigen::Matrix3d eps_mean = Eigen::Matrix3d::Zero();
        for (int i = 0; i < N; ++i) {
            eps_mean += eps[i];
        }
        eps_mean /= N;
        Eigen::Matrix3d mean_diff = eps_bar - eps_mean;
        for (int i = 0; i < N; ++i) {
            eps[i] += mean_diff;
        }
    } else {
        eps.assign(N, eps_bar);
    }

    // Auxiliary buffers for FFT/IFFT
    std::vector<double> tau_comp(N);
    std::vector<std::complex<double>> tau_hat[3][3];
    std::vector<std::complex<double>> eps_tilde_hat[3][3];
    std::vector<double> eps_tilde[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int s = 0; s < 3; ++s) {
            tau_hat[r][s].resize(N);
            eps_tilde_hat[r][s].resize(N);
            eps_tilde[r][s].resize(N);
        }
    }

    // Lippmann-Schwinger loop
    for (int it = 0; it < max_iter; ++it) {
        // 1. Stress calculation: sig = C : eps
        std::vector<Eigen::Matrix3d> sig = stress_from_strain_3d(eps, lam, mu);

        // 2. Reference stress: sig0 = C0 : eps
        std::vector<Eigen::Matrix3d> tau(N);
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix3d sig0 = 2.0 * mu0 * eps[i] + lam0 * eps[i].trace() * Eigen::Matrix3d::Identity();
            tau[i] = sig[i] - sig0;
        }

        // 3. FFT of each polarization stress component
        for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
                for (int i = 0; i < N; ++i) tau_comp[i] = tau[i](r, s);
                fft3d(nx, ny, nz, tau_comp, tau_hat[r][s]);
            }
        }

        // 4. Multiply with Green operator in Fourier space
        // eps_tilde_hat = - Gamma : tau_hat
        for (int i = 0; i < N; ++i) {
            std::complex<double> tau_hat_node[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int s = 0; s < 3; ++s) {
                    tau_hat_node[r][s] = tau_hat[r][s][i];
                }
            }

            const auto& G = Gamma[i];

            for (int k = 0; k < 3; ++k) {
                for (int h = 0; h < 3; ++h) {
                    std::complex<double> val = 0.0;
                    for (int r = 0; r < 3; ++r) {
                        for (int s = 0; s < 3; ++s) {
                            val += G[k][h][r][s] * tau_hat_node[r][s];
                        }
                    }
                    eps_tilde_hat[k][h][i] = -val;
                }
            }
        }

        // 5. Inverse FFT of eps_tilde_hat
        for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
                ifft3d(nx, ny, nz, eps_tilde_hat[r][s], eps_tilde[r][s]);
            }
        }

        // 6. Subtract mean
        double mean_comp[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
        for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
                double sum = 0.0;
                for (int i = 0; i < N; ++i) {
                    sum += eps_tilde[r][s][i];
                }
                mean_comp[r][s] = sum / N;
            }
        }

        // 7. Update strain field
        std::vector<Eigen::Matrix3d> eps_new(N);
        double diff_sq = 0.0;
        double norm_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int r = 0; r < 3; ++r) {
                for (int s = 0; s < 3; ++s) {
                    eps_new[i](r, s) = eps_bar(r, s) + (eps_tilde[r][s][i] - mean_comp[r][s]);
                }
            }
            diff_sq += (eps_new[i] - eps[i]).squaredNorm();
            norm_sq += eps[i].squaredNorm();
        }

        double diff = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-20);
        eps = eps_new;

        if (verbose && it % 10 == 0) {
            std::cout << "  Iter " << it << ": diff = " << diff << std::endl;
        }

        if (diff < tol) {
            break;
        }
    }

    // Compute final stresses and macros
    sig_out = stress_from_strain_3d(eps, lam, mu);
    eps_macro = Eigen::Matrix3d::Zero();
    sig_macro = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i) {
        eps_macro += eps[i];
        sig_macro += sig_out[i];
    }
    eps_macro /= N;
    sig_macro /= N;
}

#endif // LINEAR_ELASTIC_SOLVER_HPP
