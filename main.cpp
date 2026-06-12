#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

#include "yaml_parser.hpp"
#include "linear_elastic_solver.hpp"
#include "material_generator.hpp"
#include "vtu_writer.hpp"
#include "kmc_solver.hpp"

// Utility function to format current local time
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

int main(int argc, char* argv[]) {
    // 1. Load Configuration
    std::string config_path = "cpp_config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    std::cout << "Loading configuration from " << config_path << std::endl;
    YAMLConfig config;
    if (!config.load(config_path)) {
        std::cerr << "Error: Failed to load config file." << std::endl;
        return 1;
    }
    
    std::string sim_type = config.getString("simulation_type", "kmc");
    std::cout << "Simulation Type selected: '" << sim_type << "'" << std::endl;
    std::cout << "Running small-strain linear elastic simulation in C++" << std::endl;
    
    std::string dimensionality = config.getString("system.dimensionality", "2d");
    std::transform(dimensionality.begin(), dimensionality.end(), dimensionality.begin(), ::tolower);
    bool is_3d = (dimensionality == "3d");
    
    std::string plane_mode = config.getString("system.plane_mode", "plane_strain");
    int nx = config.getInt("system.nx", 32);
    int ny = config.getInt("system.ny", 32);
    int nz = is_3d ? config.getInt("system.nz", 32) : 1;
    double pixel = config.getDouble("system.pixel", 1.0);
    
    int N = nx * ny * nz;
    std::vector<double> E_field(N);
    std::string E_mode = config.getString("material.E.mode", "constant");
    std::transform(E_mode.begin(), E_mode.end(), E_mode.begin(), ::tolower);
    
    if (E_mode == "constant") {
        double E_val = config.getDouble("material.E.value", 70.0);
        if (E_val < 1e6) {
            std::cout << " [main] Detected E in GPa (" << E_val << "). Converting to Pa (*1e9)." << std::endl;
            E_val *= 1e9;
        }
        std::fill(E_field.begin(), E_field.end(), E_val);
    } 
    else if (E_mode == "file") {
        std::string path = config.getString("material.E.parameters.path", "");
        if (path.empty()) {
            std::cerr << "Error: material.E.mode is 'file' but material.E.parameters.path is empty or missing." << std::endl;
            return 1;
        }
        std::cout << "Reading E field from file: " << path << std::endl;
        std::vector<size_t> file_shape;
        try {
            E_field = read_npy_double(path, file_shape);
        } catch (const std::exception& e) {
            std::cerr << "Error: Failed to read npy file: " << e.what() << std::endl;
            return 1;
        }
        // Verify shape
        bool shape_ok = false;
        if (!is_3d && file_shape.size() == 2 && file_shape[0] == (size_t)nx && file_shape[1] == (size_t)ny) {
            shape_ok = true;
        } else if (is_3d && file_shape.size() == 3 && file_shape[0] == (size_t)nx && file_shape[1] == (size_t)ny && file_shape[2] == (size_t)nz) {
            shape_ok = true;
        }
        if (!shape_ok) {
            std::cerr << "Error: Loaded npy shape does not match grid shape." << std::endl;
            return 1;
        }
        // GPa to Pa auto-conversion
        double sum = 0.0;
        for (double val : E_field) sum += val;
        double E_mean = sum / E_field.size();
        if (E_mean < 1e6) {
            std::cout << " [main] Detected E in GPa (mean: " << E_mean << "). Converting to Pa (*1e9)." << std::endl;
            for (double& val : E_field) val *= 1e9;
        }
    } 
    else if (E_mode == "generated") {
        double E_val = config.getDouble("material.E.value", 70.0);
        double mean = config.getDouble("material.E.parameters.mean", E_val);
        double std_val = config.getDouble("material.E.parameters.std", 0.0);
        double corr = config.getDouble("material.E.parameters.corr", 10.0);
        
        bool has_clip_min = false;
        double clip_min = 0.0;
        std::string clip_min_str = config.getString("material.E.parameters.clip_min", "");
        if (!clip_min_str.empty()) {
            has_clip_min = true;
            clip_min = std::stod(clip_min_str);
        }
        
        bool has_clip_max = false;
        double clip_max = 0.0;
        std::string clip_max_str = config.getString("material.E.parameters.clip_max", "");
        if (!clip_max_str.empty()) {
            has_clip_max = true;
            clip_max = std::stod(clip_max_str);
        }
        
        int seed = config.getInt("seed", 42);
        std::cout << "Generating random correlated E field (mean: " << mean 
                  << ", std: " << std_val << ", corr: " << corr << ", seed: " << seed << ")" << std::endl;
        
        E_field = generate_correlated_field_cpp(
            nx, ny, nz, is_3d,
            mean, std_val, corr,
            has_clip_min, clip_min,
            has_clip_max, clip_max,
            seed
        );
        
        // GPa to Pa auto-conversion
        double sum = 0.0;
        for (double val : E_field) sum += val;
        double E_mean = sum / E_field.size();
        if (E_mean < 1e6) {
            std::cout << " [main] Detected E in GPa (mean: " << E_mean << "). Converting to Pa (*1e9)." << std::endl;
            for (double& val : E_field) val *= 1e9;
        }
    } 
    else {
        std::cerr << "Error: Unsupported material E mode: " << E_mode << std::endl;
        return 1;
    }

    std::vector<double> nu_field(N);
    std::string nu_mode = config.getString("material.nu.mode", "constant");
    std::transform(nu_mode.begin(), nu_mode.end(), nu_mode.begin(), ::tolower);
    
    if (nu_mode == "constant") {
        double nu_val = config.getDouble("material.nu.value", 0.3);
        std::fill(nu_field.begin(), nu_field.end(), nu_val);
    } 
    else if (nu_mode == "file") {
        std::string path = config.getString("material.nu.parameters.path", "");
        if (path.empty()) {
            std::cerr << "Error: material.nu.mode is 'file' but material.nu.parameters.path is empty or missing." << std::endl;
            return 1;
        }
        std::cout << "Reading nu field from file: " << path << std::endl;
        std::vector<size_t> file_shape;
        try {
            nu_field = read_npy_double(path, file_shape);
        } catch (const std::exception& e) {
            std::cerr << "Error: Failed to read npy file: " << e.what() << std::endl;
            return 1;
        }
        // Verify shape
        bool shape_ok = false;
        if (!is_3d && file_shape.size() == 2 && file_shape[0] == (size_t)nx && file_shape[1] == (size_t)ny) {
            shape_ok = true;
        } else if (is_3d && file_shape.size() == 3 && file_shape[0] == (size_t)nx && file_shape[1] == (size_t)ny && file_shape[2] == (size_t)nz) {
            shape_ok = true;
        }
        if (!shape_ok) {
            std::cerr << "Error: Loaded npy shape does not match grid shape." << std::endl;
            return 1;
        }
    } 
    else if (nu_mode == "generated") {
        double nu_val = config.getDouble("material.nu.value", 0.3);
        double mean = config.getDouble("material.nu.parameters.mean", nu_val);
        double std_val = config.getDouble("material.nu.parameters.std", 0.0);
        double corr = config.getDouble("material.nu.parameters.corr", 10.0);
        
        bool has_clip_min = false;
        double clip_min = 0.0;
        std::string clip_min_str = config.getString("material.nu.parameters.clip_min", "");
        if (!clip_min_str.empty()) {
            has_clip_min = true;
            clip_min = std::stod(clip_min_str);
        }
        
        bool has_clip_max = false;
        double clip_max = 0.0;
        std::string clip_max_str = config.getString("material.nu.parameters.clip_max", "");
        if (!clip_max_str.empty()) {
            has_clip_max = true;
            clip_max = std::stod(clip_max_str);
        }
        
        int seed = config.getInt("seed", 42) + 1; // offset seed for nu if generated
        std::cout << "Generating random correlated nu field (mean: " << mean 
                  << ", std: " << std_val << ", corr: " << corr << ", seed: " << seed << ")" << std::endl;
        
        nu_field = generate_correlated_field_cpp(
            nx, ny, nz, is_3d,
            mean, std_val, corr,
            has_clip_min, clip_min,
            has_clip_max, clip_max,
            seed
        );
    } 
    else {
        std::cerr << "Error: Unsupported material nu mode: " << nu_mode << std::endl;
        return 1;
    }

    double E_avg = 0.0;
    for (double val : E_field) E_avg += val;
    E_avg /= E_field.size();
    
    double nu_avg = 0.0;
    for (double val : nu_field) nu_avg += val;
    nu_avg /= nu_field.size();
    
    // Outputs & Logging Config
    std::string out_dir = config.getString("output.directory", "simulation_results");
    std::string summary_filename = config.getString("output.summary_filename", "summary_log.txt");
    std::string duplicate_dir_action = config.getString("output.duplicate_directory_action", "delete");
    bool enable_console = config.getBool("output.enable_console", true);
    bool enable_summary = config.getBool("output.enable_summary_log", true);
    bool enable_global = config.getBool("output.enable_global_log", true);
    
    std::string vtk_interval_str = config.getString("output.vtk_interval", "none");
    std::transform(vtk_interval_str.begin(), vtk_interval_str.end(), vtk_interval_str.begin(), ::tolower);
    int vtk_interval_int = -1;
    bool vtk_interval_is_int = false;
    if (vtk_interval_str != "none" && vtk_interval_str != "last" && vtk_interval_str != "current") {
        try {
            vtk_interval_int = std::stoi(vtk_interval_str);
            if (vtk_interval_int > 0) {
                vtk_interval_is_int = true;
            }
        } catch (...) {}
    }
    std::string vtk_path = out_dir + "/step";
    
    std::string checkpoint_interval = config.getString("output.checkpoint_interval", "none");
    std::transform(checkpoint_interval.begin(), checkpoint_interval.end(), checkpoint_interval.begin(), ::tolower);
    if (checkpoint_interval != "none") {
        std::cout << " [main] Warning: output.checkpoint_interval is set to '" << checkpoint_interval 
                  << "', but HDF5 checkpointing is not implemented in C++ due to library dependency constraints. "
                  << "Continuing simulation without checkpoints." << std::endl;
    }
    
    try {
        if (duplicate_dir_action == "delete" && std::filesystem::exists(out_dir)) {
            std::filesystem::remove_all(out_dir);
        }
        std::filesystem::create_directories(out_dir);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Warning: Directory access issue for '" << out_dir << "': " << e.what() 
                  << "\nProceeding with existing directory and overwriting files." << std::endl;
    }
    
    // Loading parameters
    double eps_target = config.getDouble("loading.eps_target", 0.05);
    double step_size = config.getDouble("loading.step_size", 1e-4);
    int n_steps = static_cast<int>(std::round(eps_target / step_size));
    
    // Setup boundary conditions
    std::string driving_raw = config.getString("boundary_conditions.driving_component", "xx");
    
    if (sim_type == "kmc") {
        // 1. Initialize KMC simulator parameters
        double mixed_tol_mpa = config.getDouble("boundary_conditions.mixed_tol", 1.0);
        BarrierGenerator bg;
        bg.type = config.getString("barriers.type", "gaussian");
        bg.mean = config.getDouble("barriers.kwargs.mean", 2.0);
        bg.std_val = config.getDouble("barriers.kwargs.std", 0.6);
        bg.min_cutoff = config.getDouble("barriers.kwargs.min_cutoff", 0.1);
        bg.max_cutoff = config.getDouble("barriers.kwargs.max_cutoff", -1.0);
        bg.loc = config.getDouble("barriers.kwargs.loc", 1.0);
        bg.scale = config.getDouble("barriers.kwargs.scale", 0.5);
        bg.epsilon = config.getDouble("barriers.kwargs.epsilon", 0.1);
        bg.ratio = config.getDouble("barriers.kwargs.ratio", 0.8);
        
        std::string soft_scheme = config.getString("physics.softening_scheme", "isotropic");
        double soft_cap = config.getBool("physics.enable_softening", true) ? config.getDouble("physics.softening_cap", 2.0) : 0.0;
        double jp_val = config.getBool("physics.enable_softening", true) ? config.getDouble("physics.jp", 10.0) : 0.0;
        double jt_val = config.getBool("physics.enable_softening", true) ? config.getDouble("physics.jt", 30.0) : 0.0;
        double neigh_frac = config.getBool("physics.enable_softening", true) ? config.getDouble("physics.neighbor_softening_fraction", 0.0) : 0.0;
        double q_act = config.getDouble("physics.q_act_temp", 0.37);
        
        bool redraw_b = config.getBool("physics.redraw_barriers", true);
        bool redraw_d = config.getBool("physics.redraw_directions", true);
        double stab_thresh = config.getDouble("physics.stability_threshold", 0.0);
        
        std::string instab_mode = config.getString("dynamics.instability_mode", "cascade");
        std::string casc_timing = config.getString("dynamics.cascade_timing", "none");
        bool scale_vol = config.getBool("dynamics.scale_rate_by_volume", true);
        double nu0_val = config.getDouble("dynamics.nu0", 1e13);
        double strain_rate_val = config.getDouble("dynamics.physical_strain_rate", 1.0);
        double temp_val = config.getDouble("dynamics.temperature", 0.0);
        
        bool fp_enabled = config.getBool("dynamics.fast_patching.enabled", false);
        int fp_radius = config.getInt("dynamics.fast_patching.patch_radius", 5);
        int fp_sync = config.getInt("dynamics.fast_patching.sync_interval", 100);
        
        int M_val = config.getInt("system.M", 20);
        double gamma0_val = config.getDouble("system.gamma0", 0.14);
        int seed_val = config.getInt("seed", 42);
        bool use_3d_barriers = config.getBool("system.3d_barriers", false);
        
        // 2. Parse driving component and mixed targets
        int drv_i = 0, drv_j = 0;
        std::map<std::pair<int, int>, double> stress_targets;
        
        if (is_3d) {
            if (driving_raw == "xx") { drv_i = 0; drv_j = 0; }
            else if (driving_raw == "yy") { drv_i = 1; drv_j = 1; }
            else if (driving_raw == "zz") { drv_i = 2; drv_j = 2; }
            else if (driving_raw == "xy") { drv_i = 0; drv_j = 1; }
            else if (driving_raw == "yx") { drv_i = 1; drv_j = 0; }
            else if (driving_raw == "xz") { drv_i = 0; drv_j = 2; }
            else if (driving_raw == "zx") { drv_i = 2; drv_j = 0; }
            else if (driving_raw == "yz") { drv_i = 1; drv_j = 2; }
            else if (driving_raw == "zy") { drv_i = 2; drv_j = 1; }
            
            std::map<std::string, double> mixed_raw = config.getDict("boundary_conditions.mixed_targets");
            for (const auto& pair : mixed_raw) {
                int mi = 0, mj = 0;
                if (pair.first == "xx") { mi = 0; mj = 0; }
                else if (pair.first == "yy") { mi = 1; mj = 1; }
                else if (pair.first == "zz") { mi = 2; mj = 2; }
                else if (pair.first == "xy") { mi = 0; mj = 1; }
                else if (pair.first == "yx") { mi = 1; mj = 0; }
                else if (pair.first == "xz") { mi = 0; mj = 2; }
                else if (pair.first == "zx") { mi = 2; mj = 0; }
                else if (pair.first == "yz") { mi = 1; mj = 2; }
                else if (pair.first == "zy") { mi = 2; mj = 1; }
                else continue;
                
                double s_val = pair.second;
                if (s_val < 1e6) {
                    s_val *= 1e9;
                }
                stress_targets[{mi, mj}] = s_val;
            }
        } else {
            if (driving_raw == "xx") { drv_i = 0; drv_j = 0; }
            else if (driving_raw == "yy") { drv_i = 1; drv_j = 1; }
            else if (driving_raw == "xy") { drv_i = 0; drv_j = 1; }
            else if (driving_raw == "yx") { drv_i = 1; drv_j = 0; }
            
            std::map<std::string, double> mixed_raw = config.getDict("boundary_conditions.mixed_targets");
            for (const auto& pair : mixed_raw) {
                int mi = 0, mj = 0;
                if (pair.first == "xx") { mi = 0; mj = 0; }
                else if (pair.first == "yy") { mi = 1; mj = 1; }
                else if (pair.first == "xy") { mi = 0; mj = 1; }
                else if (pair.first == "yx") { mi = 1; mj = 0; }
                else continue;
                
                double s_val = pair.second;
                if (s_val < 1e6) {
                    s_val *= 1e9;
                }
                stress_targets[{mi, mj}] = s_val;
            }
        }
        
        double max_kmc_steps_pct = config.getDouble("detection.max_kmc_steps_pct", 0.3);
        
        if (is_3d) {
            KmcSimulation3D sim(
                nx, ny, nz, M_val, gamma0_val, E_field, nu_field, pixel,
                bg, soft_scheme, soft_cap, jp_val, jt_val, neigh_frac,
                q_act, out_dir, temp_val, strain_rate_val, stab_thresh, nu0_val,
                fp_enabled, fp_radius, fp_sync,
                instab_mode, casc_timing, scale_vol, redraw_d, redraw_b, use_3d_barriers, seed_val
            );
            
            sim.run_simulation(
                n_steps, step_size, {drv_i, drv_j}, stress_targets,
                mixed_tol_mpa * 1e6, 50, checkpoint_interval, "checkpoint",
                vtk_interval_str, config.getBool("output.vtk_elastic_only", true),
                config.getBool("output.track_cascades", false),
                enable_console, summary_filename, enable_summary, enable_global,
                max_kmc_steps_pct
            );
        } else {
            KmcSimulation2D sim(
                nx, ny, M_val, gamma0_val, E_field, nu_field, pixel,
                bg, soft_scheme, soft_cap, jp_val, jt_val, neigh_frac,
                q_act, out_dir, temp_val, strain_rate_val, stab_thresh, nu0_val,
                plane_mode, fp_enabled, fp_radius, fp_sync,
                instab_mode, casc_timing, scale_vol, redraw_d, redraw_b, seed_val
            );
            
            sim.run_simulation(
                n_steps, step_size, {drv_i, drv_j}, stress_targets,
                mixed_tol_mpa * 1e6, 50, checkpoint_interval, "checkpoint",
                vtk_interval_str, config.getBool("output.vtk_elastic_only", true),
                config.getBool("output.track_cascades", false),
                enable_console, summary_filename, enable_summary, enable_global,
                max_kmc_steps_pct
            );
        }
        
        return 0;
    }
    
    std::cout << "\n--- Starting execution ---" << std::endl;
    std::cout << "Grid: " << nx << "x" << ny << (is_3d ? "x" + std::to_string(nz) : "") << " (" << (is_3d ? "3D" : "2D") << ")" << std::endl;
    if (!is_3d) {
        std::cout << "Mode: " << (plane_mode == "plane_stress" ? "Plane Stress" : "Plane Strain") << std::endl;
    } else {
        std::cout << "Mode: 3D (Ignoring plane stress / plane strain)" << std::endl;
    }
    std::cout << "Driving Strain Component: " << driving_raw << std::endl;
    
    // Open summary log
    std::ofstream summary_log;
    if (enable_summary) {
        std::string path = out_dir + "/" + summary_filename;
        summary_log.open(path);
        summary_log << std::left << std::setw(20) << "Timestamp" << " "
                    << std::setw(12) << "Elapsed(s)" << " "
                    << std::setw(8) << "Step" << " "
                    << std::setw(14) << ("Eps_" + driving_raw) << " "
                    << std::setw(16) << ("Sig_" + driving_raw + "(GPa)") << "\n";
        summary_log << std::string(75, '-') << "\n";
    }
    
    int max_iter_macro = 20;
    int max_iter_ls = 200;
    double tol_ls = 1e-6;
    double mixed_tol_mpa = config.getDouble("boundary_conditions.mixed_tol", 1.0);
    double tol_macro = mixed_tol_mpa * 1e6;
    
    if (is_3d) {
        // ==========================================
        // 3D SOLVER PATHWAY
        // ==========================================
        int drv_i = 0, drv_j = 0;
        if (driving_raw == "xx") { drv_i = 0; drv_j = 0; }
        else if (driving_raw == "yy") { drv_i = 1; drv_j = 1; }
        else if (driving_raw == "zz") { drv_i = 2; drv_j = 2; }
        else if (driving_raw == "xy") { drv_i = 0; drv_j = 1; }
        else if (driving_raw == "yx") { drv_i = 1; drv_j = 0; }
        else if (driving_raw == "xz") { drv_i = 0; drv_j = 2; }
        else if (driving_raw == "zx") { drv_i = 2; drv_j = 0; }
        else if (driving_raw == "yz") { drv_i = 1; drv_j = 2; }
        else if (driving_raw == "zy") { drv_i = 2; drv_j = 1; }
        else {
            std::cerr << "Warning: Unknown driving component '" << driving_raw << "'. Defaulting to xx." << std::endl;
        }
        
        Eigen::Matrix3d target_values = Eigen::Matrix3d::Zero();
        Eigen::Matrix<bool, 3, 3> target_strain_mask = Eigen::Matrix<bool, 3, 3>::Constant(true); // true = strain-controlled
        
        target_values(drv_i, drv_j) = eps_target;
        target_strain_mask(drv_i, drv_j) = true;
        
        // Non-driven components in shear default to strain-controlled to 0
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i != j && !(i == drv_i && j == drv_j)) {
                    target_values(i, j) = 0.0;
                    target_strain_mask(i, j) = true;
                }
            }
        }
        
        // Read mixed boundary conditions (stresses)
        std::map<std::string, double> mixed_raw = config.getDict("boundary_conditions.mixed_targets");
        std::map<std::pair<int, int>, double> stress_targets;
        for (const auto& pair : mixed_raw) {
            int mi = 0, mj = 0;
            if (pair.first == "xx") { mi = 0; mj = 0; }
            else if (pair.first == "yy") { mi = 1; mj = 1; }
            else if (pair.first == "zz") { mi = 2; mj = 2; }
            else if (pair.first == "xy") { mi = 0; mj = 1; }
            else if (pair.first == "yx") { mi = 1; mj = 0; }
            else if (pair.first == "xz") { mi = 0; mj = 2; }
            else if (pair.first == "zx") { mi = 2; mj = 0; }
            else if (pair.first == "yz") { mi = 1; mj = 2; }
            else if (pair.first == "zy") { mi = 2; mj = 1; }
            else continue;
            
            double s_val = pair.second;
            if (s_val < 1e6) { // Convert GPa to Pa
                s_val *= 1e9;
            }
            stress_targets[{mi, mj}] = s_val;
            target_values(mi, mj) = s_val;
            target_strain_mask(mi, mj) = false; // false = stress-controlled
        }
        
        std::cout << "Target Stresses (Relaxed Components):" << std::endl;
        for (const auto& st : stress_targets) {
            std::string label = (st.first.first == 0 && st.first.second == 0) ? "xx" :
                                (st.first.first == 1 && st.first.second == 1) ? "yy" :
                                (st.first.first == 2 && st.first.second == 2) ? "zz" : "xy/xz/yz";
            std::cout << "  " << label << ": " << (st.second / 1e9) << " GPa" << std::endl;
        }
        std::cout << "Running 3D Small-Strain Mixed Solver for " << n_steps << " steps..." << std::endl;
        
        std::ofstream global_log;
        if (enable_global) {
            std::string path = out_dir + "/global_log.txt";
            global_log.open(path);
            global_log << std::left << std::setw(12) << "GlobalStep" << " "
                       << std::setw(14) << "Eps_xx" << " "
                       << std::setw(14) << "Eps_yy" << " "
                       << std::setw(14) << "Eps_zz" << " "
                       << std::setw(14) << "Eps_xy" << " "
                       << std::setw(14) << "Eps_xz" << " "
                       << std::setw(14) << "Eps_yz" << " "
                       << std::setw(16) << "Sig_xx(GPa)" << " "
                       << std::setw(16) << "Sig_yy(GPa)" << " "
                       << std::setw(16) << "Sig_zz(GPa)" << " "
                       << std::setw(16) << "Sig_xy(GPa)" << " "
                       << std::setw(16) << "Sig_xz(GPa)" << " "
                       << std::setw(16) << "Sig_yz(GPa)" << "\n";
            global_log << std::string(195, '-') << "\n";
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // 3D Grid Wavevectors and Gamma
        auto [lam0, mu0] = compute_lame_3d(E_avg, nu_avg);
        double Lx = nx * pixel;
        double Ly = ny * pixel;
        double Lz = nz * pixel;
        std::vector<double> kx(N), ky(N), kz(N);
        std::vector<double> kx_1d = fftfreq(nx, Lx / nx);
        std::vector<double> ky_1d = fftfreq(ny, Ly / ny);
        std::vector<double> kz_1d = fftfreq(nz, Lz / nz);
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    int idx = x * (ny * nz) + y * nz + z;
                    kx[idx] = 2.0 * M_PI * kx_1d[x];
                    ky[idx] = 2.0 * M_PI * ky_1d[y];
                    kz[idx] = 2.0 * M_PI * kz_1d[z];
                }
            }
        }
        std::vector<GreenTensor3D> Gamma = green_operator_3d(nx, ny, nz, kx, ky, kz, lam0, mu0);
        
        Eigen::Matrix3d current_eps_bar = Eigen::Matrix3d::Zero();
        std::vector<Eigen::Matrix3d> eps;
        std::vector<Eigen::Matrix3d> sig_field;
        
        for (int s = 0; s <= n_steps; ++s) {
            Eigen::Matrix3d target_s = target_values * (static_cast<double>(s) / n_steps);
            double max_err = 0.0;
            
            Eigen::Matrix3d epsM, sigM;
            
            for (int it_macro = 0; it_macro < max_iter_macro; ++it_macro) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (target_strain_mask(i, j)) {
                            current_eps_bar(i, j) = target_s(i, j);
                        }
                    }
                }
                
                spectral_solver_3d(
                    nx, ny, nz, E_field, nu_field, current_eps_bar, Gamma,
                    eps, sig_field, epsM, sigM,
                    max_iter_ls, tol_ls, false
                );
                
                Eigen::Matrix3d stress_err = Eigen::Matrix3d::Zero();
                max_err = 0.0;
                bool has_stress_control = false;
                
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (!target_strain_mask(i, j)) {
                            stress_err(i, j) = target_s(i, j) - sigM(i, j);
                            max_err = std::max(max_err, std::abs(stress_err(i, j)));
                            has_stress_control = true;
                        }
                    }
                }
                
                if (!has_stress_control || max_err < tol_macro) {
                    break;
                }
                
                double tr_sig = stress_err.trace();
                Eigen::Matrix3d d_eps = (stress_err - nu_avg * tr_sig * Eigen::Matrix3d::Identity()) / E_avg;
                
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (!target_strain_mask(i, j)) {
                            current_eps_bar(i, j) += d_eps(i, j);
                        }
                    }
                }
            }
            
            double eps_drv = epsM(drv_i, drv_j);
            double sig_drv = sigM(drv_i, drv_j);
            
            if (enable_console) {
                std::cout << "step " << s << "/" << n_steps << ": "
                          << "eps_" << driving_raw << "=" << std::fixed << std::setprecision(4) << eps_drv << ", "
                          << "sig_" << driving_raw << "=" << std::setprecision(2) << (sig_drv / 1e6) << " MPa" << std::endl;
            }
            
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t0).count();
            
            if (enable_summary && summary_log.is_open()) {
                summary_log << std::left << std::setw(20) << get_current_timestamp() << " "
                            << std::setw(12) << std::fixed << std::setprecision(3) << elapsed << " "
                            << std::setw(8) << s << " "
                            << std::setw(14) << std::setprecision(6) << eps_drv << " "
                            << std::setw(16) << (sig_drv / 1e9) << "\n";
            }
            
            if (enable_global && global_log.is_open()) {
                global_log << std::left << std::setw(12) << s << " "
                           << std::setw(14) << std::fixed << std::setprecision(6) << epsM(0, 0) << " "
                           << std::setw(14) << epsM(1, 1) << " "
                           << std::setw(14) << epsM(2, 2) << " "
                           << std::setw(14) << epsM(0, 1) << " "
                           << std::setw(14) << epsM(0, 2) << " "
                           << std::setw(14) << epsM(1, 2) << " "
                           << std::setw(16) << (sigM(0, 0) / 1e9) << " "
                           << std::setw(16) << (sigM(1, 1) / 1e9) << " "
                           << std::setw(16) << (sigM(2, 2) / 1e9) << " "
                           << std::setw(16) << (sigM(0, 1) / 1e9) << " "
                           << std::setw(16) << (sigM(0, 2) / 1e9) << " "
                           << std::setw(16) << (sigM(1, 2) / 1e9) << "\n";
            }
            
            // VTK Export
            bool save_vtk = false;
            std::string vt_name = "";
            if (vtk_interval_str != "none") {
                if (vtk_interval_str == "current") {
                    vt_name = vtk_path + ".vtu";
                    save_vtk = true;
                } else if (vtk_interval_is_int && s % vtk_interval_int == 0) {
                    std::stringstream ss_vt;
                    ss_vt << vtk_path << "_" << std::setfill('0') << std::setw(6) << s << ".vtu";
                    vt_name = ss_vt.str();
                    save_vtk = true;
                }
                
                if (s == n_steps) {
                    if (vtk_interval_str == "last" || (vtk_interval_is_int && n_steps % vtk_interval_int != 0)) {
                        vt_name = vtk_path + "_final.vtu";
                        save_vtk = true;
                    }
                }
            }
            if (save_vtk && !vt_name.empty()) {
                export_to_vtu(vt_name, nx, ny, nz, eps, sig_field, E_field, nu_field, pixel);
            }
        }
        
        if (global_log.is_open()) global_log.close();
        std::cout << "\n3D Elastic simulation completed. Data output to directory: " << out_dir << std::endl;
        
    } else {
        // ==========================================
        // 2D SOLVER PATHWAY
        // ==========================================
        int drv_i = 0, drv_j = 0;
        if (driving_raw == "xx") { drv_i = 0; drv_j = 0; }
        else if (driving_raw == "yy") { drv_i = 1; drv_j = 1; }
        else if (driving_raw == "xy") { drv_i = 0; drv_j = 1; }
        else if (driving_raw == "yx") { drv_i = 1; drv_j = 0; }
        else {
            std::cerr << "Warning: Unknown driving component '" << driving_raw << "'. Defaulting to xx." << std::endl;
        }
        
        Eigen::Matrix2d target_values = Eigen::Matrix2d::Zero();
        Eigen::Matrix<bool, 2, 2> target_strain_mask = Eigen::Matrix<bool, 2, 2>::Constant(true); // true = strain-controlled
        
        target_values(drv_i, drv_j) = eps_target;
        target_strain_mask(drv_i, drv_j) = true;
        
        // Read mixed boundary conditions (stresses)
        std::map<std::string, double> mixed_raw = config.getDict("boundary_conditions.mixed_targets");
        std::map<std::pair<int, int>, double> stress_targets;
        for (const auto& pair : mixed_raw) {
            int mi = 0, mj = 0;
            if (pair.first == "xx") { mi = 0; mj = 0; }
            else if (pair.first == "yy") { mi = 1; mj = 1; }
            else if (pair.first == "xy") { mi = 0; mj = 1; }
            else if (pair.first == "yx") { mi = 1; mj = 0; }
            else continue;
            
            double s_val = pair.second;
            if (s_val < 1e6) { // Convert GPa to Pa
                s_val *= 1e9;
            }
            stress_targets[{mi, mj}] = s_val;
            target_values(mi, mj) = s_val;
            target_strain_mask(mi, mj) = false; // false = stress-controlled
        }
        
        std::cout << "Target Stresses (Relaxed Components):" << std::endl;
        for (const auto& st : stress_targets) {
            std::string label = (st.first.first == 0 && st.first.second == 0) ? "xx" :
                                (st.first.first == 1 && st.first.second == 1) ? "yy" : "xy";
            std::cout << "  " << label << ": " << (st.second / 1e9) << " GPa" << std::endl;
        }
        std::cout << "Running 2D Small-Strain Mixed Solver for " << n_steps << " steps..." << std::endl;
        
        std::ofstream global_log;
        if (enable_global) {
            std::string path = out_dir + "/global_log.txt";
            global_log.open(path);
            global_log << std::left << std::setw(12) << "GlobalStep" << " "
                       << std::setw(14) << "Eps_xx" << " "
                       << std::setw(14) << "Eps_yy" << " "
                       << std::setw(14) << "Eps_xy" << " "
                       << std::setw(16) << "Sig_xx(GPa)" << " "
                       << std::setw(16) << "Sig_yy(GPa)" << " "
                       << std::setw(16) << "Sig_xy(GPa)" << "\n";
            global_log << std::string(110, '-') << "\n";
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // 2D Grid Wavevectors and Gamma
        auto [lam0, mu0] = compute_lame_2d(E_avg, nu_avg, plane_mode);
        double Lx = nx * pixel;
        double Ly = ny * pixel;
        std::vector<double> kx(N), ky(N);
        std::vector<double> kx_1d = fftfreq(nx, Lx / nx);
        std::vector<double> ky_1d = fftfreq(ny, Ly / ny);
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                int idx = x * ny + y;
                kx[idx] = 2.0 * M_PI * kx_1d[x];
                ky[idx] = 2.0 * M_PI * ky_1d[y];
            }
        }
        std::vector<GreenTensor> Gamma = green_operator_2d(nx, ny, kx, ky, lam0, mu0);
        
        Eigen::Matrix2d current_eps_bar = Eigen::Matrix2d::Zero();
        std::vector<Eigen::Matrix2d> eps;
        std::vector<Eigen::Matrix2d> sig_field;
        
        for (int s = 0; s <= n_steps; ++s) {
            Eigen::Matrix2d target_s = target_values * (static_cast<double>(s) / n_steps);
            double max_err = 0.0;
            
            Eigen::Matrix2d epsM, sigM;
            
            for (int it_macro = 0; it_macro < max_iter_macro; ++it_macro) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        if (target_strain_mask(i, j)) {
                            current_eps_bar(i, j) = target_s(i, j);
                        }
                    }
                }
                
                spectral_solver_2d(
                    nx, ny, E_field, nu_field, current_eps_bar, plane_mode, Gamma,
                    eps, sig_field, epsM, sigM,
                    max_iter_ls, tol_ls, false
                );
                
                Eigen::Matrix2d stress_err = Eigen::Matrix2d::Zero();
                max_err = 0.0;
                bool has_stress_control = false;
                
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        if (!target_strain_mask(i, j)) {
                            stress_err(i, j) = target_s(i, j) - sigM(i, j);
                            max_err = std::max(max_err, std::abs(stress_err(i, j)));
                            has_stress_control = true;
                        }
                    }
                }
                
                if (!has_stress_control || max_err < tol_macro) {
                    break;
                }
                
                double tr_sig = stress_err.trace();
                Eigen::Matrix2d d_eps = (stress_err - nu_avg * tr_sig * Eigen::Matrix2d::Identity()) / E_avg;
                
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        if (!target_strain_mask(i, j)) {
                            current_eps_bar(i, j) += d_eps(i, j);
                        }
                    }
                }
            }
            
            double eps_drv = epsM(drv_i, drv_j);
            double sig_drv = sigM(drv_i, drv_j);
            
            if (enable_console) {
                std::cout << "step " << s << "/" << n_steps << ": "
                          << "eps_" << driving_raw << "=" << std::fixed << std::setprecision(4) << eps_drv << ", "
                          << "sig_" << driving_raw << "=" << std::setprecision(2) << (sig_drv / 1e6) << " MPa" << std::endl;
            }
            
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t0).count();
            
            if (enable_summary && summary_log.is_open()) {
                summary_log << std::left << std::setw(20) << get_current_timestamp() << " "
                            << std::setw(12) << std::fixed << std::setprecision(3) << elapsed << " "
                            << std::setw(8) << s << " "
                            << std::setw(14) << std::setprecision(6) << eps_drv << " "
                            << std::setw(16) << (sig_drv / 1e9) << "\n";
            }
            
            if (enable_global && global_log.is_open()) {
                global_log << std::left << std::setw(12) << s << " "
                           << std::setw(14) << std::fixed << std::setprecision(6) << epsM(0, 0) << " "
                           << std::setw(14) << epsM(1, 1) << " "
                           << std::setw(14) << epsM(0, 1) << " "
                           << std::setw(16) << (sigM(0, 0) / 1e9) << " "
                           << std::setw(16) << (sigM(1, 1) / 1e9) << " "
                           << std::setw(16) << (sigM(0, 1) / 1e9) << "\n";
            }
            
            // VTK Export
            bool save_vtk = false;
            std::string vt_name = "";
            if (vtk_interval_str != "none") {
                if (vtk_interval_str == "current") {
                    vt_name = vtk_path + ".vtu";
                    save_vtk = true;
                } else if (vtk_interval_is_int && s % vtk_interval_int == 0) {
                    std::stringstream ss_vt;
                    ss_vt << vtk_path << "_" << std::setfill('0') << std::setw(6) << s << ".vtu";
                    vt_name = ss_vt.str();
                    save_vtk = true;
                }
                
                if (s == n_steps) {
                    if (vtk_interval_str == "last" || (vtk_interval_is_int && n_steps % vtk_interval_int != 0)) {
                        vt_name = vtk_path + "_final.vtu";
                        save_vtk = true;
                    }
                }
            }
            if (save_vtk && !vt_name.empty()) {
                export_to_vtu_2d(vt_name, nx, ny, eps, sig_field, E_field, nu_field, pixel);
            }
        }
        
        if (global_log.is_open()) global_log.close();
        std::cout << "\n2D Elastic simulation completed. Data output to directory: " << out_dir << std::endl;
    }
    
    if (summary_log.is_open()) summary_log.close();
    return 0;
}
