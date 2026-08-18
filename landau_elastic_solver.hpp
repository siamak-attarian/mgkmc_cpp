#ifndef LANDAU_ELASTIC_SOLVER_HPP
#define LANDAU_ELASTIC_SOLVER_HPP

// Small-strain Landau strain-energy constitutive law and its FFT-based
// (Moulinec-Suquet Picard / Anderson-accelerated) field solver.
//
// This is a direct port of two Python functions:
//   - elasticity.py:stress_from_strain_landau_2d   (lines 487-639, the fast
//     "scalar block-diagonal" form, NOT the tensor _reference oracle)
//   - linear_elastic_simulator.py:spectral_solver_landau_2d, solver="dbfft"
//     branch (lines ~1120-1220)
//
// The dbfft solver here is a plain Lippmann-Schwinger fixed-point iteration
// (no tangent/Newton assembly) optionally accelerated by Anderson type-II
// mixing of the last `anderson_m` iterates — it is unrelated to, and much
// simpler than, the finite-strain dbfft Newton-CG solver.

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Dense>
#include "linear_elastic_solver.hpp"   // GreenTensor, fftfreq, compute_lame_2d,
                                        // green_operator_2d, fft2d, ifft2d

struct StrainCappingParams {
    bool enabled = false;
    double limit = -1.0;            // <=0 means "auto" (matches Python None/<=0)
    double tangent_ratio = 0.1;
    std::string type = "piecewise"; // "piecewise" or "smooth"
    double smooth_power = 1.0;
};

// Per-pixel capped-strain scalars needed for stress + (for plane_stress) the
// e33 Newton residual, mirroring elasticity.py's inner `_scalars(e33)`.
struct LandauScalars {
    double I1, I2;
    double c11, c22, c12, c21, c33;
    double w;   // capping_weight (0 if capping disabled/E_eq==0)
    double coeff_I, coeff_eps, coeff_eps2;
};

inline LandauScalars landau_scalars_2d(
    double a11, double a22, double a12, double a21, double e33,
    double lam, double mu,
    double v1, double v2, double v3, double g1, double g2, double g3, double g4,
    bool capping, double E_cap, const std::string& cap_type, double smooth_power)
{
    LandauScalars s;
    double I1_raw = a11 + a22 + e33;
    double m3 = I1_raw / 3.0;
    double d11 = a11 - m3, d22 = a22 - m3, d33 = e33 - m3;

    double f = 1.0;
    s.w = 0.0;
    if (capping) {
        double E_eq_sq = std::max(0.0, (2.0 / 3.0) *
            (d11 * d11 + d22 * d22 + d33 * d33 + 2.0 * a12 * a21));
        double E_eq = std::sqrt(E_eq_sq);
        if (cap_type == "smooth") {
            double p = smooth_power;
            double ratio_p = std::min(100.0, std::pow(E_eq / E_cap, p));
            s.w = std::pow(std::tanh(ratio_p), 2.0);
            if (E_eq > 1e-12) {
                f = (E_cap / E_eq) * std::pow(std::tanh(ratio_p), 1.0 / p);
            }
        } else {
            bool is_capped = E_eq > E_cap;
            s.w = is_capped ? 1.0 : 0.0;
            if (is_capped) f = E_cap / E_eq;
        }
    }

    s.c11 = f * d11 + m3;
    s.c22 = f * d22 + m3;
    s.c12 = f * a12;
    s.c21 = f * a21;
    s.c33 = f * d33 + m3;

    double cross = s.c12 * s.c21;
    double tr2_ip = s.c11 * s.c11 + s.c22 * s.c22 + 2.0 * cross;
    s.I1 = s.c11 + s.c22 + s.c33;
    s.I2 = tr2_ip + s.c33 * s.c33;
    double t_ip = s.c11 + s.c22;
    double det_ip = s.c11 * s.c22 - cross;
    double I3 = t_ip * (tr2_ip - det_ip) + s.c33 * s.c33 * s.c33;

    s.coeff_I = lam * s.I1 + 0.5 * v1 * s.I1 * s.I1 + v2 * s.I2
              + (1.0 / 6.0) * g1 * s.I1 * s.I1 * s.I1
              + g2 * s.I1 * s.I2 + (4.0 / 3.0) * g3 * I3;
    s.coeff_eps = 2.0 * (mu + v2 * s.I1 + 0.5 * g2 * s.I1 * s.I1 + g4 * s.I2);
    s.coeff_eps2 = 4.0 * (v3 + g3 * s.I1);
    return s;
}

// Cauchy stress for the 2D Landau small-strain model, with optional strain
// capping. e33_state holds the per-pixel plane-stress e33 field, warm-started
// across calls (empty vector = cold start, matching Python's e33_state=None
// then first-call linear-elastic initial guess).
inline std::vector<Eigen::Matrix2d> stress_from_strain_landau_2d(
    const std::vector<Eigen::Matrix2d>& eps,
    const std::vector<double>& lam,
    const std::vector<double>& mu,
    double v1, double v2, double v3, double g1, double g2, double g3, double g4,
    const std::string& plane_mode,
    const StrainCappingParams& cap,
    std::vector<double>& e33_state)
{
    int N = (int)eps.size();
    std::vector<Eigen::Matrix2d> sig(N);

    bool capping = cap.enabled;
    double E_cap = 999.0, G_tangent_ratio = cap.tangent_ratio;
    if (capping) {
        double mu_mean = 0.0;
        for (int i = 0; i < N; ++i) mu_mean += mu[i];
        mu_mean /= N;
        if (cap.limit <= 0.0) {
            E_cap = (g4 < 0.0) ? std::sqrt(0.2 * mu_mean / std::abs(g4)) : 999.0;
        } else {
            E_cap = cap.limit;
        }
    }

    bool plane_stress = (plane_mode == "plane_stress");
    if (plane_stress && (int)e33_state.size() != N) {
        e33_state.assign(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double a11 = eps[i](0, 0), a22 = eps[i](1, 1);
            e33_state[i] = -lam[i] / (lam[i] + 2.0 * mu[i]) * (a11 + a22);
        }
    }

    const double tol_e33 = 1e-9;
    bool any_not_converged = false;
    double max_final_delta = 0.0;

    for (int i = 0; i < N; ++i) {
        double a11 = eps[i](0, 0), a22 = eps[i](1, 1);
        double a12 = eps[i](0, 1), a21 = eps[i](1, 0);
        double G_tangent = G_tangent_ratio * mu[i];

        LandauScalars s;
        if (plane_stress) {
            double e33 = e33_state[i];
            bool converged = false;
            double last_delta = 0.0;
            for (int it = 0; it < 20; ++it) {
                s = landau_scalars_2d(a11, a22, a12, a21, e33, lam[i], mu[i],
                                       v1, v2, v3, g1, g2, g3, g4,
                                       capping, E_cap, cap.type, cap.smooth_power);

                double sigma33 = s.coeff_I + s.coeff_eps * s.c33 + s.coeff_eps2 * s.c33 * s.c33;
                double dcoeff_I = lam[i] + v1 * s.I1 + 0.5 * g1 * s.I1 * s.I1 + g2 * s.I2
                                 + 2.0 * (v2 + g2 * s.I1) * s.c33 + 4.0 * g3 * s.c33 * s.c33;
                double dcoeff_eps = 2.0 * (v2 + g2 * s.I1) + 4.0 * g4 * s.c33;
                double dsigma33 = dcoeff_I + dcoeff_eps * s.c33 + s.coeff_eps
                                 + 4.0 * g3 * s.c33 * s.c33 + 2.0 * s.coeff_eps2 * s.c33;
                if (capping) {
                    sigma33 += 2.0 * G_tangent * (e33 - s.c33) * s.w;
                    dsigma33 += 2.0 * G_tangent * s.w;
                }

                double delta = -sigma33 / (dsigma33 + 1e-12);
                if (!std::isfinite(delta)) {
                    throw std::runtime_error(
                        "stress_from_strain_landau_2d: non-finite e33 Newton update "
                        "(Landau energy likely unstable at this strain; consider "
                        "strain capping or smaller load steps).");
                }
                e33 += delta;
                last_delta = delta;
                if (std::abs(delta) < tol_e33) { converged = true; break; }
            }
            if (!converged) {
                any_not_converged = true;
                max_final_delta = std::max(max_final_delta, std::abs(last_delta));
            }
            e33_state[i] = e33;
            // NOTE: `s` intentionally holds the pre-final-update Newton state
            // (evaluated at e33 BEFORE the last accepted delta), matching the
            // reference/Python behavior — see elasticity.py:623-624 comment.
        } else {
            s = landau_scalars_2d(a11, a22, a12, a21, 0.0, lam[i], mu[i],
                                   v1, v2, v3, g1, g2, g3, g4,
                                   capping, E_cap, cap.type, cap.smooth_power);
        }

        double cross = s.c12 * s.c21;
        double t_ip = s.c11 + s.c22;
        double sxx = s.coeff_I + s.coeff_eps * s.c11 + s.coeff_eps2 * (s.c11 * s.c11 + cross);
        double syy = s.coeff_I + s.coeff_eps * s.c22 + s.coeff_eps2 * (s.c22 * s.c22 + cross);
        double sxy = s.coeff_eps * s.c12 + s.coeff_eps2 * (s.c12 * t_ip);
        double syx = s.coeff_eps * s.c21 + s.coeff_eps2 * (s.c21 * t_ip);

        if (capping) {
            double r = 2.0 * G_tangent * s.w;
            sxx += r * (a11 - s.c11);
            syy += r * (a22 - s.c22);
            sxy += r * (a12 - s.c12);
            syx += r * (a21 - s.c21);
        }
        sig[i](0, 0) = sxx; sig[i](1, 1) = syy;
        sig[i](0, 1) = sxy; sig[i](1, 0) = syx;
    }

    if (any_not_converged) {
        std::cerr << " [stress_from_strain_landau_2d] Warning: e33 Newton not "
                     "converged in 20 iterations on some pixels (max final |delta| = "
                  << max_final_delta << "); sigma33 = 0 only approximately satisfied."
                  << std::endl;
    }

    return sig;
}

// Nonlinear Lippmann-Schwinger (Moulinec-Suquet) solver for the 2D Landau
// small-strain model, "dbfft" (basic Picard, optionally Anderson-accelerated)
// scheme — port of linear_elastic_simulator.py:spectral_solver_landau_2d,
// solver="dbfft" branch. No tangent/Newton assembly.
inline void spectral_solver_landau_2d(
    int nx, int ny,
    const std::vector<double>& lam,
    const std::vector<double>& mu,
    double v1, double v2, double v3, double g1, double g2, double g3, double g4,
    const Eigen::Matrix2d& eps_bar,
    const std::string& plane_mode,
    const std::vector<GreenTensor>& Gamma,
    const StrainCappingParams& cap,
    std::vector<double>& e33_state,
    std::vector<Eigen::Matrix2d>& eps,   // in/out: warm-started strain field
    std::vector<Eigen::Matrix2d>& sig_out,
    Eigen::Matrix2d& eps_macro,
    Eigen::Matrix2d& sig_macro,
    int max_iter = 400,
    double tol = 1e-6,
    int anderson_m = 5,
    bool verbose = false,
    const std::vector<Eigen::Matrix2d>& eps_plastic = {})
{
    // eps_plastic (STZ eigenstrain, if any) is subtracted only when
    // evaluating the REAL constitutive stress; the reference/polarization
    // split below (sig0 = C0:eps, used for the Green's-operator step) still
    // acts on the raw kinematic field `eps`, matching spectral_solver_2d's
    // eps_plastic handling (linear_elastic_simulator.py:34-40) — the
    // fixed-point iteration solves for a compatible strain field `eps`;
    // the eigenstrain only shifts what stress that field produces.
    auto elastic_strain = [&](const std::vector<Eigen::Matrix2d>& e) {
        if (eps_plastic.empty()) return e;
        std::vector<Eigen::Matrix2d> e_el(e.size());
        for (size_t i = 0; i < e.size(); ++i) e_el[i] = e[i] - eps_plastic[i];
        return e_el;
    };
    int N = nx * ny;
    double lam_avg = 0.0, mu_avg = 0.0;
    for (int i = 0; i < N; ++i) { lam_avg += lam[i]; mu_avg += mu[i]; }
    lam_avg /= N; mu_avg /= N;

    // Initialize / warm-start strain field to match eps_bar mean.
    if ((int)eps.size() == N) {
        Eigen::Matrix2d eps_mean = Eigen::Matrix2d::Zero();
        for (int i = 0; i < N; ++i) eps_mean += eps[i];
        eps_mean /= N;
        Eigen::Matrix2d mean_diff = eps_bar - eps_mean;
        for (int i = 0; i < N; ++i) eps[i] += mean_diff;
    } else {
        eps.assign(N, eps_bar);
    }

    std::vector<double> comp(N);
    std::vector<std::complex<double>> tau_hat_xx(N), tau_hat_xy(N), tau_hat_yx(N), tau_hat_yy(N);
    std::vector<std::complex<double>> eps_tilde_hat_xx(N), eps_tilde_hat_xy(N), eps_tilde_hat_yx(N), eps_tilde_hat_yy(N);
    std::vector<double> eps_tilde_xx(N), eps_tilde_xy(N), eps_tilde_yx(N), eps_tilde_yy(N);

    // Anderson (type-II) mixing history: fixed-point images G(x) and
    // residuals f = G(x) - x of the last (anderson_m + 1) iterates, each
    // flattened to a 4N vector (xx, xy, yx, yy stacked per pixel).
    std::vector<Eigen::VectorXd> aa_g, aa_f;
    auto flatten = [&](const std::vector<Eigen::Matrix2d>& field) {
        Eigen::VectorXd v(4 * N);
        for (int i = 0; i < N; ++i) {
            v(4 * i + 0) = field[i](0, 0);
            v(4 * i + 1) = field[i](0, 1);
            v(4 * i + 2) = field[i](1, 0);
            v(4 * i + 3) = field[i](1, 1);
        }
        return v;
    };
    auto unflatten = [&](const Eigen::VectorXd& v) {
        std::vector<Eigen::Matrix2d> field(N);
        for (int i = 0; i < N; ++i) {
            field[i](0, 0) = v(4 * i + 0);
            field[i](0, 1) = v(4 * i + 1);
            field[i](1, 0) = v(4 * i + 2);
            field[i](1, 1) = v(4 * i + 3);
        }
        return field;
    };

    for (int it = 0; it < max_iter; ++it) {
        // 1. Landau stress (elastic strain = eps - eps_plastic)
        std::vector<Eigen::Matrix2d> sig = stress_from_strain_landau_2d(
            elastic_strain(eps), lam, mu, v1, v2, v3, g1, g2, g3, g4, plane_mode, cap, e33_state);

        // 2-3. Reference stress sig0 = C0:eps (raw eps, NOT elastic strain —
        // see note above), polarization tau = sig - sig0
        std::vector<Eigen::Matrix2d> tau(N);
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix2d sig0 = 2.0 * mu_avg * eps[i]
                                  + lam_avg * eps[i].trace() * Eigen::Matrix2d::Identity();
            tau[i] = sig[i] - sig0;
        }

        // 4. FFT of polarization
        for (int i = 0; i < N; ++i) comp[i] = tau[i](0, 0); fft2d(nx, ny, comp, tau_hat_xx);
        for (int i = 0; i < N; ++i) comp[i] = tau[i](0, 1); fft2d(nx, ny, comp, tau_hat_xy);
        for (int i = 0; i < N; ++i) comp[i] = tau[i](1, 0); fft2d(nx, ny, comp, tau_hat_yx);
        for (int i = 0; i < N; ++i) comp[i] = tau[i](1, 1); fft2d(nx, ny, comp, tau_hat_yy);

        // 5. Apply Green operator: eps_tilde_hat = -Gamma:tau_hat
        for (int i = 0; i < N; ++i) {
            std::complex<double> tau_node[2][2] = {
                {tau_hat_xx[i], tau_hat_xy[i]},
                {tau_hat_yx[i], tau_hat_yy[i]}
            };
            std::complex<double> res[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            const auto& G = Gamma[i];
            for (int k = 0; k < 2; ++k)
                for (int h = 0; h < 2; ++h) {
                    std::complex<double> val = 0.0;
                    for (int r = 0; r < 2; ++r)
                        for (int c = 0; c < 2; ++c)
                            val += G[k][h][r][c] * tau_node[r][c];
                    res[k][h] = -val;
                }
            eps_tilde_hat_xx[i] = res[0][0];
            eps_tilde_hat_xy[i] = res[0][1];
            eps_tilde_hat_yx[i] = res[1][0];
            eps_tilde_hat_yy[i] = res[1][1];
        }

        // 6. Inverse FFT
        ifft2d(nx, ny, eps_tilde_hat_xx, eps_tilde_xx);
        ifft2d(nx, ny, eps_tilde_hat_xy, eps_tilde_xy);
        ifft2d(nx, ny, eps_tilde_hat_yx, eps_tilde_yx);
        ifft2d(nx, ny, eps_tilde_hat_yy, eps_tilde_yy);

        // 7. Zero-mean correction, then G(eps) = eps_bar + eps_tilde
        double mxx = 0, mxy = 0, myx = 0, myy = 0;
        for (int i = 0; i < N; ++i) {
            mxx += eps_tilde_xx[i]; mxy += eps_tilde_xy[i];
            myx += eps_tilde_yx[i]; myy += eps_tilde_yy[i];
        }
        mxx /= N; mxy /= N; myx /= N; myy /= N;

        std::vector<Eigen::Matrix2d> eps_new(N);
        for (int i = 0; i < N; ++i) {
            eps_new[i](0, 0) = eps_bar(0, 0) + (eps_tilde_xx[i] - mxx);
            eps_new[i](0, 1) = eps_bar(0, 1) + (eps_tilde_xy[i] - mxy);
            eps_new[i](1, 0) = eps_bar(1, 0) + (eps_tilde_yx[i] - myx);
            eps_new[i](1, 1) = eps_bar(1, 1) + (eps_tilde_yy[i] - myy);
        }

        double diff_sq = 0.0, norm_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            diff_sq += (eps_new[i] - eps[i]).squaredNorm();
            norm_sq += eps[i].squaredNorm();
        }
        double diff = std::sqrt(diff_sq) / (std::sqrt(norm_sq) + 1e-20);
        if (!std::isfinite(diff)) {
            throw std::runtime_error(
                "spectral_solver_landau_2d: non-finite strain update in LS loop "
                "(Landau energy likely unstable at this strain; consider strain "
                "capping or smaller load steps).");
        }
        if (verbose && it % 10 == 0) {
            std::cout << "  [landau-LS] iter " << it << ": rel_diff = " << diff << std::endl;
        }

        if (diff < tol) { eps = eps_new; break; }

        // 8. Anderson type-II mixing of the last (anderson_m + 1) iterates.
        if (anderson_m > 0) {
            Eigen::VectorXd g_new = flatten(eps_new);
            Eigen::VectorXd f_new = g_new - flatten(eps);
            aa_g.push_back(g_new);
            aa_f.push_back(f_new);
            if ((int)aa_f.size() > anderson_m + 1) {
                aa_g.erase(aa_g.begin());
                aa_f.erase(aa_f.begin());
            }
        }
        if (anderson_m > 0 && (int)aa_f.size() >= 2) {
            int m = (int)aa_f.size() - 1;
            Eigen::MatrixXd dF(4 * N, m), dG(4 * N, m);
            for (int j = 0; j < m; ++j) {
                dF.col(j) = aa_f[j + 1] - aa_f[j];
                dG.col(j) = aa_g[j + 1] - aa_g[j];
            }
            Eigen::VectorXd gamma_aa = dF.colPivHouseholderQr().solve(aa_f.back());
            bool ok = gamma_aa.allFinite();
            if (ok) {
                Eigen::VectorXd eps_aa = aa_g.back() - dG * gamma_aa;
                if (eps_aa.allFinite()) {
                    eps = unflatten(eps_aa);
                } else {
                    ok = false;
                }
            }
            if (!ok) {
                // Ill-conditioned mixing: plain step and restart history.
                eps = eps_new;
                aa_g = {aa_g.back()};
                aa_f = {aa_f.back()};
            }
        } else {
            eps = eps_new;
        }
    }

    sig_out = stress_from_strain_landau_2d(
        elastic_strain(eps), lam, mu, v1, v2, v3, g1, g2, g3, g4, plane_mode, cap, e33_state);
    eps_macro = Eigen::Matrix2d::Zero();
    sig_macro = Eigen::Matrix2d::Zero();
    for (int i = 0; i < N; ++i) { eps_macro += eps[i]; sig_macro += sig_out[i]; }
    eps_macro /= N;
    sig_macro /= N;
}

#endif // LANDAU_ELASTIC_SOLVER_HPP
