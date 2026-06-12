#ifndef KMC_SOLVER_HPP
#define KMC_SOLVER_HPP

#include <vector>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>

#include "linear_elastic_solver.hpp"
#include "vtu_writer.hpp"

// Forward declaration of get_current_timestamp
std::string get_current_timestamp();

struct BarrierGenerator {
    std::string type;
    double mean = 2.0;
    double std_val = 0.6;
    double min_cutoff = 0.1;
    double max_cutoff = -1.0; // flag for no max cutoff if negative
    double loc = 1.0;
    double scale = 0.5;
    double epsilon = 0.1;
    double ratio = 0.8;

    double generate(std::mt19937& rng) const {
        double val = 0.0;
        if (type == "gaussian") {
            std::normal_distribution<double> dist(mean, std_val);
            val = dist(rng);
        } else if (type == "rayleigh") {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double u = dist(rng);
            val = loc + scale * std::sqrt(-2.0 * std::log(1.0 - u));
        } else if (type == "modified_rayleigh") {
            double a = std::max(0.0, mean - 6.0 * std_val);
            double b = mean + 6.0 * std_val;
            double x_max = (mean + std::sqrt(mean * mean + 4.0 * std_val * std_val)) / 2.0;
            double c_max = x_max * std::exp(-std::pow(x_max - mean, 2.0) / (2.0 * std_val * std_val));
            std::uniform_real_distribution<double> dist_x(a, b);
            std::uniform_real_distribution<double> dist_u(0.0, c_max);
            while (true) {
                double x = dist_x(rng);
                double u = dist_u(rng);
                double pdf = x * std::exp(-std::pow(x - mean, 2.0) / (2.0 * std_val * std_val));
                if (u < pdf) {
                    val = x;
                    break;
                }
            }
        } else if (type == "modified_rayleigh_with_exponential") {
            std::uniform_real_distribution<double> dist_u(0.0, 1.0);
            double u_mix = dist_u(rng);
            if (u_mix < ratio) {
                // Exponential
                std::uniform_real_distribution<double> dist_exp(0.0, 1.0);
                val = -epsilon * std::log(1.0 - dist_exp(rng));
            } else {
                // Modified Rayleigh
                double a = std::max(0.0, mean - 6.0 * std_val);
                double b = mean + 6.0 * std_val;
                double x_max = (mean + std::sqrt(mean * mean + 4.0 * std_val * std_val)) / 2.0;
                double c_max = x_max * std::exp(-std::pow(x_max - mean, 2.0) / (2.0 * std_val * std_val));
                std::uniform_real_distribution<double> dist_x(a, b);
                std::uniform_real_distribution<double> dist_u(0.0, c_max);
                while (true) {
                    double x = dist_x(rng);
                    double u = dist_u(rng);
                    double pdf = x * std::exp(-std::pow(x - mean, 2.0) / (2.0 * std_val * std_val));
                    if (u < pdf) {
                        val = x;
                        break;
                    }
                }
            }
        }
        
        // Clip
        if (min_cutoff >= 0.0 && val < min_cutoff) {
            val = min_cutoff;
        }
        if (max_cutoff >= 0.0 && val > max_cutoff) {
            val = max_cutoff;
        }
        return val;
    }
};

class KmcSimulation2D {
public:
    int nx, ny;
    int M;
    double gamma0;
    double pixel;
    double volume;
    std::string output_dir;
    
    // Physics parameters
    std::string softening_scheme;
    double softening_cap;
    double jp, jt;
    double neighbor_softening_fraction;
    double temperature;
    double strain_rate;
    double stability_threshold;
    double nu0;
    double q_act_temp;
    std::string plane_mode;
    
    // Fast Patching Flags
    bool fast_patching_enabled;
    int patch_radius;
    int sync_interval;
    int flips_since_sync;
    
    // Dynamics modes
    std::string instability_mode;
    std::string cascade_timing;
    bool scale_rate_by_volume;
    bool redraw_directions;
    bool redraw_barriers;
    double tau;
    
    // Fields
    std::vector<double> E_field;
    std::vector<double> nu_field;
    std::vector<Eigen::Matrix2d> eps_field;
    std::vector<Eigen::Matrix2d> sig_field;
    std::vector<Eigen::Matrix2d> eps_plastic;
    std::vector<Eigen::Vector2d> soft_prop; // [g_p, g_t]
    std::vector<double> last_event_time;
    Eigen::Matrix2d eps_macro;
    Eigen::Matrix2d sig_macro;
    double time;
    
    // Barriers and Catalog
    std::vector<double> Q; // nx * ny * M
    std::vector<double> Q0; // nx * ny * M
    std::vector<Eigen::Matrix2d> catalog; // nx * ny * M
    std::vector<Eigen::Matrix2d> prev_strain_dir; // nx * ny
    
    std::vector<GreenTensor> Gamma;
    BarrierGenerator barrier_gen;
    std::mt19937 rng;
    
    // Fast Patching data structures
    std::vector<std::vector<Eigen::Matrix2d>> patch_kernels; // 2 bases, each size (2*R+1)*(2*R+1)
    std::vector<Eigen::Matrix2d> patch_missing_mean; // 2 bases
    Eigen::Matrix2d sigma_macro_unit;
    bool sigma_macro_unit_initialized;
    
    // File streams
    std::ofstream f_summary;
    std::ofstream f_global;
    
    KmcSimulation2D(
        int nx_, int ny_, int M_, double gamma0_, 
        const std::vector<double>& E_field_, const std::vector<double>& nu_field_,
        double pixel_ = 1.0,
        const BarrierGenerator& barrier_gen_ = BarrierGenerator(),
        std::string softening_scheme_ = "isotropic",
        double softening_cap_ = 2.0, double jp_ = 10.0, double jt_ = 30.0,
        double neighbor_softening_fraction_ = 0.0,
        double q_act_temp_ = 0.37, std::string output_dir_ = "output",
        double temperature_ = 0.0, double strain_rate_ = 1.0,
        double stability_threshold_ = 0.0, double nu0_ = 1e13,
        std::string plane_mode_ = "plane_strain",
        bool fast_patching_enabled_ = false, int patch_radius_ = 5, int sync_interval_ = 100,
        std::string instability_mode_ = "cascade", std::string cascade_timing_ = "none",
        bool scale_rate_by_volume_ = true,
        bool redraw_directions_ = true, bool redraw_barriers_ = true,
        int seed_ = 42
    ) : nx(nx_), ny(ny_), M(M_), gamma0(gamma0_), pixel(pixel_), volume(pixel_ * pixel_ * pixel_),
        output_dir(output_dir_), softening_scheme(softening_scheme_), softening_cap(softening_cap_),
        jp(jp_), jt(jt_), neighbor_softening_fraction(neighbor_softening_fraction_),
        temperature(temperature_), strain_rate(strain_rate_), stability_threshold(stability_threshold_),
        nu0(nu0_), q_act_temp(q_act_temp_), plane_mode(plane_mode_),
        fast_patching_enabled(fast_patching_enabled_), patch_radius(patch_radius_), sync_interval(sync_interval_),
        flips_since_sync(0), instability_mode(instability_mode_), cascade_timing(cascade_timing_),
        scale_rate_by_volume(scale_rate_by_volume_),
        redraw_directions(redraw_directions_), redraw_barriers(redraw_barriers_),
        barrier_gen(barrier_gen_), rng(seed_), sigma_macro_unit_initialized(false)
    {
        E_field = E_field_;
        nu_field = nu_field_;
        
        int N = nx * ny;
        eps_field.assign(N, Eigen::Matrix2d::Zero());
        sig_field.assign(N, Eigen::Matrix2d::Zero());
        eps_plastic.assign(N, Eigen::Matrix2d::Zero());
        soft_prop.assign(N, Eigen::Vector2d::Zero());
        last_event_time.assign(N, -std::numeric_limits<double>::infinity());
        eps_macro = Eigen::Matrix2d::Zero();
        sig_macro = Eigen::Matrix2d::Zero();
        time = 0.0;
        
        // Physics param: tau
        if (temperature > 0.0) {
            double kB = 8.617e-5; // eV/K
            tau = 1.0 / (nu0 * std::exp(-q_act_temp / (kB * temperature)));
            std::cout << " [KmcSimulation2D] T=" << temperature << "K > 0: Calculated Dynamic Softening Decay (tau): " << tau << " s" << std::endl;
        } else {
            tau = std::numeric_limits<double>::infinity();
            std::cout << " [KmcSimulation2D] T=0K: Softening Decay (tau): Infinite (No decay)" << std::endl;
        }
        
        // Initialize barriers & catalog
        Q.assign(N * M, 0.0);
        Q0.resize(N * M);
        catalog.resize(N * M);
        prev_strain_dir.assign(N, Eigen::Matrix2d::Zero());
        
        // Populate base barriers Q0 and STZ catalog
        std::normal_distribution<double> standard_normal(0.0, 1.0);
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                int voxel_idx = x * ny + y;
                for (int m = 0; m < M; ++m) {
                    int mode_idx = voxel_idx * M + m;
                    Q0[mode_idx] = barrier_gen.generate(rng);
                    
                    // Generate M independent 2x2 STZ modes
                    double gxx = 0.5 * gamma0 * standard_normal(rng);
                    double gxy = 0.5 * gamma0 * standard_normal(rng);
                    catalog[mode_idx](0, 0) = gxx;
                    catalog[mode_idx](1, 1) = -gxx;
                    catalog[mode_idx](0, 1) = gxy;
                    catalog[mode_idx](1, 0) = gxy;
                }
            }
        }
        
        // Precompute Gamma
        double E_sum = 0.0;
        double nu_sum = 0.0;
        for (int i = 0; i < N; ++i) {
            E_sum += E_field[i];
            nu_sum += nu_field[i];
        }
        double E_avg = E_sum / N;
        double nu_avg = nu_sum / N;
        init_gamma(E_avg, nu_avg);
        
        // Precompute patches if fast patching enabled
        if (fast_patching_enabled) {
            precompute_patch_kernels();
        }
    }
    
    void init_gamma(double E_avg, double nu_avg) {
        auto [lam0, mu0] = compute_lame_2d(E_avg, nu_avg, plane_mode);
        double Lx = nx * pixel;
        double Ly = ny * pixel;
        std::vector<double> kx(nx * ny), ky(nx * ny);
        std::vector<double> kx_1d = fftfreq(nx, Lx / nx);
        std::vector<double> ky_1d = fftfreq(ny, Ly / ny);
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                int idx = x * ny + y;
                kx[idx] = 2.0 * M_PI * kx_1d[x];
                ky[idx] = 2.0 * M_PI * ky_1d[y];
            }
        }
        Gamma = green_operator_2d(nx, ny, kx, ky, lam0, mu0);
    }
    
    void precompute_patch_kernels() {
        std::cout << "\n [KmcSimulation2D] Pre-computing Stress Patches (Radius=" << patch_radius << ")..." << std::endl;
        int R = patch_radius;
        int nx_center = nx / 2;
        int ny_center = ny / 2;
        int N = nx * ny;
        
        // 2 Deviatoric bases
        std::vector<Eigen::Matrix2d> bases = {
            (Eigen::Matrix2d() << 1.0, 0.0, 0.0, -1.0).finished(), // xx = -yy
            (Eigen::Matrix2d() << 0.0, 1.0, 1.0, 0.0).finished()   // xy
        };
        
        patch_kernels.resize(2);
        patch_missing_mean.resize(2);
        
        for (int b = 0; b < 2; ++b) {
            std::vector<Eigen::Matrix2d> temp_eps_plas(N, Eigen::Matrix2d::Zero());
            temp_eps_plas[nx_center * ny + ny_center] = bases[b];
            
            std::vector<Eigen::Matrix2d> temp_eps;
            std::vector<Eigen::Matrix2d> temp_sig;
            Eigen::Matrix2d temp_eps_macro = Eigen::Matrix2d::Zero();
            Eigen::Matrix2d temp_sig_macro = Eigen::Matrix2d::Zero();
            
            spectral_solver_2d(
                nx, ny, E_field, nu_field, temp_eps_macro, plane_mode, Gamma,
                temp_eps, temp_sig, temp_eps_macro, temp_sig_macro, 200, 1e-6, false, temp_eps_plas
            );
            
            // Crop and roll so that (nx_center, ny_center) is at the center (R, R) of crop
            int side = 2 * R + 1;
            patch_kernels[b].assign(side * side, Eigen::Matrix2d::Zero());
            Eigen::Matrix2d sum_crop = Eigen::Matrix2d::Zero();
            
            for (int dx = -R; dx <= R; ++dx) {
                for (int dy = -R; dy <= R; ++dy) {
                    int sx = (nx_center + dx + nx) % nx;
                    int sy = (ny_center + dy + ny) % ny;
                    Eigen::Matrix2d val = temp_sig[sx * ny + sy];
                    
                    int crop_idx = (dx + R) * side + (dy + R);
                    patch_kernels[b][crop_idx] = val;
                    sum_crop += val;
                }
            }
            
            // Compute missing mean
            Eigen::Matrix2d sum_sig = Eigen::Matrix2d::Zero();
            for (int i = 0; i < N; ++i) {
                sum_sig += temp_sig[i];
            }
            Eigen::Matrix2d mean_sig = sum_sig / N;
            patch_missing_mean[b] = mean_sig - sum_crop / N;
        }
        std::cout << " [KmcSimulation2D] Fast Patching Kernels Ready." << std::endl;
    }
    
    void elastic_run() {
        spectral_solver_2d(
            nx, ny, E_field, nu_field, eps_macro, plane_mode, Gamma,
            eps_field, sig_field, eps_macro, sig_macro, 200, 1e-6, false, eps_plastic
        );
    }
    
    void update_barriers() {
        int scheme_idx = (softening_scheme == "directional" ? 1 : 0);
        double GPa_nm3_to_eV = 6.241509;
        
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                int voxel_idx = x * ny + y;
                double g_t = soft_prop[voxel_idx](1);
                double t_last = last_event_time[voxel_idx];
                
                double g_t_curr = 0.0;
                if (g_t > 0.0) {
                    if (std::isinf(tau)) {
                        g_t_curr = g_t;
                    } else if (t_last == -std::numeric_limits<double>::infinity()) {
                        g_t_curr = 0.0;
                    } else {
                        double dt = time - t_last;
                        if (dt < 0.0) dt = 0.0;
                        g_t_curr = g_t * std::exp(-dt / tau);
                    }
                }
                
                double g_p = soft_prop[voxel_idx](0);
                double g_base = g_p + g_t_curr;
                if (softening_cap > 0.0 && g_base > softening_cap) {
                    g_base = softening_cap;
                }
                
                for (int m = 0; m < M; ++m) {
                    int mode_idx = voxel_idx * M + m;
                    double modifier = 1.0;
                    if (scheme_idx == 1) { // Directional
                        double dot_prod = 0.0;
                        double norm_mode_sq = 0.0;
                        double norm_prev_sq = 0.0;
                        for (int i = 0; i < 2; ++i) {
                            for (int j = 0; j < 2; ++j) {
                                double val_m = catalog[mode_idx](i, j);
                                double val_p = prev_strain_dir[voxel_idx](i, j);
                                dot_prod += val_m * val_p;
                                norm_mode_sq += val_m * val_m;
                                norm_prev_sq += val_p * val_p;
                            }
                        }
                        double norm_prev = std::sqrt(norm_prev_sq);
                        double norm_mode = std::sqrt(norm_mode_sq);
                        if (norm_prev < 1e-12 || norm_mode < 1e-12) {
                            modifier = 1.0;
                        } else {
                            double cos_theta = dot_prod / (norm_mode * norm_prev);
                            modifier = std::pow(1.0 + cos_theta, 2.0) / 4.0;
                        }
                    }
                    
                    double g_eff = modifier * g_base;
                    
                    double w_sum = 0.0;
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            w_sum += sig_field[voxel_idx](i, j) * catalog[mode_idx](i, j);
                        }
                    }
                    
                    double w_val = 0.5 * volume * (w_sum / 1e9) * GPa_nm3_to_eV;
                    Q[mode_idx] = Q0[mode_idx] * std::exp(-g_eff) - w_val;
                }
            }
        }
    }
    
    std::vector<int> find_unstable_indices() {
        std::vector<int> unstable;
        int size = Q.size();
        for (int i = 0; i < size; ++i) {
            if (Q[i] < stability_threshold) {
                unstable.push_back(i);
            }
        }
        return unstable;
    }
    
    void apply_flip_soa_2d(int x, int y, int m) {
        int voxel_idx = x * ny + y;
        int mode_idx = voxel_idx * M + m;
        
        // 1. Update plastic strain
        eps_plastic[voxel_idx] += catalog[mode_idx];
        
        // 2. Update Softening (2D VM equivalent strain)
        double e11 = catalog[mode_idx](0, 0);
        double e22 = catalog[mode_idx](1, 1);
        double e12 = catalog[mode_idx](0, 1);
        double sum_sq = (e12 * e12) + (e22 * e22 + e11 * e11 + std::pow(e11 - e22, 2.0)) / 6.0;
        
        double gp_new = soft_prop[voxel_idx](0) + jp * sum_sq;
        if (softening_cap > 0.0 && gp_new > softening_cap) {
            gp_new = softening_cap;
        }
        soft_prop[voxel_idx](0) = gp_new;
        soft_prop[voxel_idx](1) = jt * sum_sq;
        last_event_time[voxel_idx] = time;
        
        // 3. Neighbor Softening (2D neighbors: 8-connectivity)
        if (neighbor_softening_fraction > 0.0) {
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    int nx_n = (x + dx + nx) % nx;
                    int ny_n = (y + dy + ny) % ny;
                    int neighbor_idx = nx_n * ny + ny_n;
                    
                    double gp_n = soft_prop[neighbor_idx](0) + neighbor_softening_fraction * jp * sum_sq;
                    if (softening_cap > 0.0 && gp_n > softening_cap) {
                        gp_n = softening_cap;
                    }
                    soft_prop[neighbor_idx](0) = gp_n;
                    soft_prop[neighbor_idx](1) += neighbor_softening_fraction * jt * sum_sq;
                }
            }
        }
    }
    
    std::pair<int, int> run_cascade(int step, std::pair<int, int> component) {
        int total_flips = 0;
        int local_step = 0;
        int limit = nx * ny;
        
        while (true) {
            update_barriers();
            std::vector<int> unstable = find_unstable_indices();
            int n_unstable = unstable.size();
            if (n_unstable == 0) break;
            
            // Sort by Q value (most negative first)
            std::sort(unstable.begin(), unstable.end(), [this](int a, int b) {
                return Q[a] < Q[b];
            });
            
            std::vector<bool> flipped_voxels_in_batch(nx * ny, false);
            int flips_in_this_batch = 0;
            
            for (int k = 0; k < n_unstable; ++k) {
                int mode_idx = unstable[k];
                int m = mode_idx % M;
                int voxel_idx = mode_idx / M;
                int y = voxel_idx % ny;
                int x = voxel_idx / ny;
                
                if (flipped_voxels_in_batch[voxel_idx]) continue;
                flipped_voxels_in_batch[voxel_idx] = true;
                flips_in_this_batch++;
                
                Eigen::Matrix2d C = catalog[mode_idx];
                apply_flip_soa_2d(x, y, m);
                prev_strain_dir[voxel_idx] = C;
                
                // Redraw catalog/barriers after flip
                std::normal_distribution<double> standard_normal(0.0, 1.0);
                if (redraw_directions || redraw_barriers) {
                    for (int mm = 0; mm < M; ++mm) {
                        int current_mode_idx = voxel_idx * M + mm;
                        if (redraw_directions) {
                            double gxx = 0.5 * gamma0 * standard_normal(rng);
                            double gxy = 0.5 * gamma0 * standard_normal(rng);
                            catalog[current_mode_idx](0, 0) = gxx;
                            catalog[current_mode_idx](1, 1) = -gxx;
                            catalog[current_mode_idx](0, 1) = gxy;
                            catalog[current_mode_idx](1, 0) = gxy;
                        }
                        if (redraw_barriers) {
                            Q0[current_mode_idx] = barrier_gen.generate(rng);
                        }
                    }
                } else {
                    double gxx = 0.5 * gamma0 * standard_normal(rng);
                    double gxy = 0.5 * gamma0 * standard_normal(rng);
                    catalog[mode_idx](0, 0) = gxx;
                    catalog[mode_idx](1, 1) = -gxx;
                    catalog[mode_idx](0, 1) = gxy;
                    catalog[mode_idx](1, 0) = gxy;
                    Q0[mode_idx] = barrier_gen.generate(rng);
                }
                
                if (fast_patching_enabled) {
                    // Update stress locally using precomputed kernels
                    double gxx = C(0, 0);
                    double gxy = C(0, 1);
                    int R = patch_radius;
                    int side = 2 * R + 1;
                    
                    for (int dx = -R; dx <= R; ++dx) {
                        for (int dy = -R; dy <= R; ++dy) {
                            int px = (x + dx + nx) % nx;
                            int py = (y + dy + ny) % ny;
                            int crop_idx = (dx + R) * side + (dy + R);
                            
                            Eigen::Matrix2d patch_val = gxx * patch_kernels[0][crop_idx] + gxy * patch_kernels[1][crop_idx];
                            sig_field[px * ny + py] += patch_val;
                        }
                    }
                    
                    Eigen::Matrix2d mean_shift = gxx * patch_missing_mean[0] + gxy * patch_missing_mean[1];
                    for (int i = 0; i < nx * ny; ++i) {
                        sig_field[i] += mean_shift;
                    }
                }
            }
            
            if (!fast_patching_enabled) {
                elastic_run();
            }
            
            total_flips += flips_in_this_batch;
            local_step++;
            if (local_step > limit) break;
        }
        
        if (fast_patching_enabled && total_flips > 0) {
            elastic_run();
        }
        
        return {local_step, total_flips};
    }
    
    void init_logs(std::string summary_filename, bool enable_summary_log, bool enable_global_log) {
        std::filesystem::create_directories(output_dir);
        std::string summary_path = output_dir + "/" + summary_filename;
        std::string global_path = output_dir + "/global_log.txt";
        
        if (enable_summary_log) {
            f_summary.open(summary_path);
            std::string header = "Timestamp              Elapsed(s)   Step     Type            Eps_xx       Sig_xx(GPa)     KMC      Cascade  Flips    SimTime(s)\n";
            f_summary << header;
            f_summary << std::string(header.length() - 1, '-') << "\n";
        }
        
        if (enable_global_log) {
            f_global.open(global_path);
            f_global << std::left << std::setw(10) << "GlobalStep" << " "
                     << std::setw(12) << "ElasticStep" << " "
                     << std::setw(10) << "KMCStep" << " "
                     << std::setw(14) << "Eps_xx" << " "
                     << std::setw(14) << "Eps_yy" << " "
                     << std::setw(14) << "Eps_zz" << " "
                     << std::setw(14) << "Eps_xy" << " "
                     << std::setw(14) << "Eps_xz" << " "
                     << std::setw(14) << "Eps_yz" << " "
                     << std::setw(14) << "Sig_xx(GPa)" << " "
                     << std::setw(14) << "Sig_yy(GPa)" << " "
                     << std::setw(14) << "Sig_zz(GPa)" << " "
                     << std::setw(14) << "Sig_xy(GPa)" << " "
                     << std::setw(14) << "Sig_xz(GPa)" << " "
                     << std::setw(14) << "Sig_yz(GPa)" << " "
                     << std::setw(13) << "CascadeSteps" << " "
                     << std::setw(16) << "TotalCascadeFlips" << " "
                     << "SimTime(s)\n";
        }
    }
    
    void close_logs() {
        if (f_summary.is_open()) f_summary.close();
        if (f_global.is_open()) f_global.close();
    }
    
    void do_logging(
        int global_step, int elastic_steps_done, int total_kmc_steps,
        std::string step_type, int cascade_steps, int cascade_flips,
        std::pair<int, int> component, bool enable_console_log,
        std::chrono::high_resolution_clock::time_point start_time_total,
        std::string vtk_interval_str, bool vtk_elastic_only
    ) {
        Eigen::Matrix2d epsM = eps_macro;
        Eigen::Matrix2d sigM = sig_macro;
        
        double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time_total).count();
        double curr_strain_val = epsM(component.first, component.second);
        double curr_stress_val = sigM(component.first, component.second);
        
        std::string now_str = get_current_timestamp();
        
        // Formatted summary line
        char buf[256];
        snprintf(
            buf, sizeof(buf),
            "%-22s %-12.2f %-8d %-15s %-12.6f %-15.3f %-8d %-8d %-8d %-15.6e\n",
            now_str.c_str(), elapsed, global_step, step_type.c_str(),
            curr_strain_val, curr_stress_val / 1e9, total_kmc_steps,
            cascade_steps, cascade_flips, time
        );
        std::string summary_line(buf);
        
        if (f_summary.is_open()) {
            f_summary << summary_line;
        }
        if (enable_console_log) {
            std::cout << summary_line;
        }
        
        // Write to global log
        if (f_global.is_open()) {
            double nu_avg = 0.3; // Default approximation for plane strain correction
            double sig_zz = 0.0;
            if (plane_mode == "plane_strain") {
                double nu_sum = 0.0;
                for (double nu_val : nu_field) nu_sum += nu_val;
                nu_avg = nu_sum / nu_field.size();
                sig_zz = nu_avg * (sigM(0, 0) + sigM(1, 1));
            }
            
            char g_buf[1024];
            snprintf(
                g_buf, sizeof(g_buf),
                "%-10d %-12d %-10d %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-14.3f %-14.3f %-14.3f %-14.3f %-14.3f %-14.3f %-13d %-16d %-15.6e\n",
                global_step, elastic_steps_done, total_kmc_steps,
                epsM(0, 0), epsM(1, 1), 0.0, epsM(0, 1), 0.0, 0.0,
                sigM(0, 0) / 1e9, sigM(1, 1) / 1e9, sig_zz / 1e9, sigM(0, 1) / 1e9, 0.0, 0.0,
                cascade_steps, cascade_flips, time
            );
            f_global << g_buf;
        }
        
        // VTK Export
        bool save_vtk = false;
        std::string vt_name = "";
        if (vtk_interval_str != "none") {
            if (vtk_interval_str == "current") {
                vt_name = output_dir + "/step.vtu";
                save_vtk = true;
            } else {
                try {
                    int interval = std::stoi(vtk_interval_str);
                    if (interval > 0) {
                        bool count_ok = false;
                        if (vtk_elastic_only) {
                            if ((step_type == "INIT" || step_type == "ELAST") && elastic_steps_done % interval == 0) {
                                count_ok = true;
                            }
                        } else {
                            if (global_step % interval == 0) {
                                count_ok = true;
                            }
                        }
                        if (count_ok) {
                            std::stringstream ss;
                            ss << output_dir << "/step_" << std::setfill('0') << std::setw(5) << global_step << ".vtu";
                            vt_name = ss.str();
                            save_vtk = true;
                        }
                    }
                } catch (...) {}
            }
        }
        
        if (save_vtk && !vt_name.empty()) {
            export_to_vtu_2d(vt_name, nx, ny, eps_field, sig_field, E_field, nu_field, pixel, {}, eps_plastic, soft_prop);
        }
    }
    
    void run_simulation(
        int n_global_steps, double step_size, std::pair<int, int> component,
        const std::map<std::pair<int, int>, double>& stress_targets, double mixed_tol = 1e-4, int mixed_max_iter = 50,
        std::string checkpoint_interval = "none", std::string checkpoint_path = "checkpoint",
        std::string vtk_interval = "none", bool vtk_elastic_only = true,
        bool track_cascades = false, bool enable_console_log = true,
        std::string summary_filename = "summary_log.txt", bool enable_summary_log = true,
        bool enable_global_log = true, double max_kmc_steps_pct = 0.3
    ) {
        init_logs(summary_filename, enable_summary_log, enable_global_log);
        auto start_time_total = std::chrono::high_resolution_clock::now();
        
        int elastic_steps_done = 0;
        int total_kmc_steps = 0;
        int cascade_event_count = 0;
        int step = 1;
        int sequential_kmc_steps = 0;
        int max_sequential_kmc = static_cast<int>(max_kmc_steps_pct * nx * ny);
        
        elastic_run();
        
        if (enable_console_log) {
            std::cout << "Timestamp              Elapsed(s)   Step     Type            Eps_xx       Sig_xx(GPa)     KMC      Cascade  Flips    SimTime(s)\n";
            std::cout << "-----------------------------------------------------------------------------------------------------------------------------\n";
        }
        
        do_logging(
            0, elastic_steps_done, total_kmc_steps, "INIT", 0, 0,
            component, enable_console_log, start_time_total, vtk_interval, vtk_elastic_only
        );
        
        Eigen::Matrix2d strain_unit = Eigen::Matrix2d::Zero();
        strain_unit(component.first, component.second) = 1.0;
        
        double dt_step = std::abs(step_size) / strain_rate;
        double remaining_time = 0.0;
        
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        
        while (elastic_steps_done < n_global_steps) {
            remaining_time += dt_step;
            while (remaining_time > 0.0) {
                update_barriers();
                
                // Instability Handling
                if (instability_mode == "cascade") {
                    std::vector<int> unstable = find_unstable_indices();
                    if (!unstable.empty()) {
                        auto [c_steps, c_flips] = run_cascade(step, component);
                        if (c_flips > 0) {
                            cascade_event_count++;
                            do_logging(
                                step, elastic_steps_done, total_kmc_steps, "CASCADE",
                                c_steps, c_flips, component, enable_console_log, start_time_total,
                                vtk_interval, vtk_elastic_only
                            );
                            step++;
                        }
                        continue;
                    }
                }
                
                // Compute rates
                double eff_volume = scale_rate_by_volume ? volume : 1.0;
                double kB = 8.617e-5;
                double beta = (temperature > 0.0) ? (1.0 / (kB * temperature)) : std::numeric_limits<double>::infinity();
                
                std::vector<double> rates;
                std::vector<int> indices;
                double total_rate = 0.0;
                
                int size = Q.size();
                for (int i = 0; i < size; ++i) {
                    double q = Q[i];
                    if (instability_mode == "kmc" || q > 0.0) {
                        double r = 0.0;
                        if (q <= 0.0) {
                            r = eff_volume * nu0;
                        } else {
                            r = eff_volume * nu0 * std::exp(-q * beta);
                        }
                        rates.push_back(r);
                        total_rate += r;
                        indices.push_back(i);
                    }
                }
                
                if (total_rate > 0.0) {
                    double r_val = uniform_dist(rng);
                    double t_wait = -std::log(r_val) / total_rate;
                    if (t_wait < remaining_time) {
                        time += t_wait;
                        eps_macro += strain_unit * (strain_rate * t_wait);
                        remaining_time -= t_wait;
                        
                        // Select event
                        double target_sum = uniform_dist(rng) * total_rate;
                        double current_sum = 0.0;
                        int selected_idx = rates.size() - 1;
                        for (size_t i = 0; i < rates.size(); ++i) {
                            current_sum += rates[i];
                            if (current_sum >= target_sum) {
                                selected_idx = i;
                                break;
                            }
                        }
                        
                        int idx_flat = indices[selected_idx];
                        int m = idx_flat % M;
                        int voxel_idx = idx_flat / M;
                        int y = voxel_idx % ny;
                        int x = voxel_idx / ny;
                        
                        bool is_instab = (Q[idx_flat] <= stability_threshold);
                        Eigen::Matrix2d C = catalog[idx_flat];
                        apply_flip_soa_2d(x, y, m);
                        prev_strain_dir[voxel_idx] = C;
                        
                        // Redraw catalog/barriers after flip
                        std::normal_distribution<double> standard_normal(0.0, 1.0);
                        if (redraw_directions || redraw_barriers) {
                            for (int mm = 0; mm < M; ++mm) {
                                int current_mode_idx = voxel_idx * M + mm;
                                if (redraw_directions) {
                                    double gxx = 0.5 * gamma0 * standard_normal(rng);
                                    double gxy = 0.5 * gamma0 * standard_normal(rng);
                                    catalog[current_mode_idx](0, 0) = gxx;
                                    catalog[current_mode_idx](1, 1) = -gxx;
                                    catalog[current_mode_idx](0, 1) = gxy;
                                    catalog[current_mode_idx](1, 0) = gxy;
                                }
                                if (redraw_barriers) {
                                    Q0[current_mode_idx] = barrier_gen.generate(rng);
                                }
                            }
                        } else {
                            double gxx = 0.5 * gamma0 * standard_normal(rng);
                            double gxy = 0.5 * gamma0 * standard_normal(rng);
                            catalog[idx_flat](0, 0) = gxx;
                            catalog[idx_flat](1, 1) = -gxx;
                            catalog[idx_flat](0, 1) = gxy;
                            catalog[idx_flat](1, 0) = gxy;
                            Q0[idx_flat] = barrier_gen.generate(rng);
                        }
                        
                        if (fast_patching_enabled) {
                            if (!sigma_macro_unit_initialized) {
                                std::vector<Eigen::Matrix2d> dummy_eps;
                                std::vector<Eigen::Matrix2d> dummy_sig;
                                Eigen::Matrix2d dummy_eps_macro = strain_unit;
                                Eigen::Matrix2d dummy_sig_macro = Eigen::Matrix2d::Zero();
                                spectral_solver_2d(
                                    nx, ny, E_field, nu_field, dummy_eps_macro, plane_mode, Gamma,
                                    dummy_eps, dummy_sig, dummy_eps_macro, dummy_sig_macro, 200, 1e-6, false
                                );
                                sigma_macro_unit = dummy_sig_macro;
                                sigma_macro_unit_initialized = true;
                            }
                            
                            // Predictor: update stress locally
                            if (t_wait > 0.0) {
                                for (int i = 0; i < nx * ny; ++i) {
                                    sig_field[i] += sigma_macro_unit * (strain_rate * t_wait);
                                }
                            }
                            
                            // Kernel superposition
                            double gxx = C(0, 0);
                            double gxy = C(0, 1);
                            int R = patch_radius;
                            int side = 2 * R + 1;
                            
                            for (int dx = -R; dx <= R; ++dx) {
                                for (int dy = -R; dy <= R; ++dy) {
                                    int px = (x + dx + nx) % nx;
                                    int py = (y + dy + ny) % ny;
                                    int crop_idx = (dx + R) * side + (dy + R);
                                    
                                    Eigen::Matrix2d patch_val = gxx * patch_kernels[0][crop_idx] + gxy * patch_kernels[1][crop_idx];
                                    sig_field[px * ny + py] += patch_val;
                                }
                            }
                            
                            Eigen::Matrix2d mean_shift = gxx * patch_missing_mean[0] + gxy * patch_missing_mean[1];
                            for (int i = 0; i < nx * ny; ++i) {
                                sig_field[i] += mean_shift;
                            }
                            
                            flips_since_sync++;
                            if (flips_since_sync >= sync_interval) {
                                elastic_run();
                                flips_since_sync = 0;
                            }
                        } else {
                            elastic_run();
                        }
                        
                        std::string log_type = is_instab ? "KMC_INSTAB" : "KMC";
                        if (is_instab) {
                            sequential_kmc_steps++;
                        } else {
                            sequential_kmc_steps = 0;
                        }
                        
                        if (sequential_kmc_steps > max_sequential_kmc) {
                            std::cout << "\n[TERMINATE] KMC instability sequence limit reached! " << sequential_kmc_steps << " steps." << std::endl;
                            close_logs();
                            return;
                        }
                        
                        total_kmc_steps++;
                        do_logging(
                            step, elastic_steps_done, total_kmc_steps, log_type,
                            0, 1, component, enable_console_log, start_time_total,
                            vtk_interval, vtk_elastic_only
                        );
                        step++;
                        continue;
                    }
                }
                
                if (fast_patching_enabled) {
                    if (sigma_macro_unit_initialized) {
                        for (int i = 0; i < nx * ny; ++i) {
                            sig_field[i] += sigma_macro_unit * (strain_rate * remaining_time);
                        }
                    }
                }
                
                eps_macro += strain_unit * (strain_rate * remaining_time);
                time += remaining_time;
                remaining_time = 0.0;
            }
            
            // Mixed boundary conditions iteration
            double E_sum = 0.0;
            double nu_sum = 0.0;
            for (double val : E_field) E_sum += val;
            for (double val : nu_field) nu_sum += val;
            double E_avg = E_sum / E_field.size();
            double nu_avg = nu_sum / nu_field.size();
            
            for (int it = 0; it < mixed_max_iter; ++it) {
                elastic_run();
                Eigen::Matrix2d stress_err = Eigen::Matrix2d::Zero();
                double err_max = 0.0;
                
                for (const auto& target : stress_targets) {
                    int mi = target.first.first;
                    int mj = target.first.second;
                    double err = target.second - sig_macro(mi, mj);
                    stress_err(mi, mj) = err;
                    err_max = std::max(err_max, std::abs(err));
                }
                
                if (err_max < mixed_tol) {
                    break;
                }
                
                double tr_sig = stress_err.trace();
                Eigen::Matrix2d d_eps = (stress_err - nu_avg * tr_sig * Eigen::Matrix2d::Identity()) / E_avg;
                for (const auto& target : stress_targets) {
                    int mi = target.first.first;
                    int mj = target.first.second;
                    eps_macro(mi, mj) += d_eps(mi, mj);
                }
            }
            
            elastic_steps_done++;
            do_logging(
                step, elastic_steps_done, total_kmc_steps, "ELAST",
                0, 0, component, enable_console_log, start_time_total,
                vtk_interval, vtk_elastic_only
            );
            step++;
        }
        
        double total_duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time_total).count();
        int h = static_cast<int>(total_duration / 3600.0);
        int m = static_cast<int>((total_duration - h * 3600.0) / 60.0);
        double s = total_duration - h * 3600.0 - m * 60.0;
        
        std::stringstream ss;
        ss << "\nSimulation Finish Time: " << get_current_timestamp() << "\n"
           << "Total Duration: " << std::fixed << std::setprecision(2) << total_duration << " seconds ("
           << h << "h " << std::setw(2) << std::setfill('0') << m << "m " << std::setw(2) << std::setfill('0') << static_cast<int>(s) << "s)\n";
        
        std::string finish_str = ss.str();
        if (f_summary.is_open()) {
            f_summary << finish_str;
        }
        if (enable_console_log) {
            std::cout << finish_str;
        }
        
        close_logs();
    }
};

#endif // KMC_SOLVER_HPP
