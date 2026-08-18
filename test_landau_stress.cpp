// Standalone check of stress_from_strain_landau_2d against hand-picked
// Python reference values (see scratchpad/check_landau_py.py). Not part of
// the production build.
#include <vector>
#include <iostream>
#include <iomanip>
#include "landau_elastic_solver.hpp"

int main() {
    double GPa = 1e9;
    std::vector<double> lam = { 80.92 * GPa };
    std::vector<double> mu  = { 23.75 * GPa };
    double v1 = -236.5 * GPa, v2 = -27.4 * GPa, v3 = 11.8 * GPa;
    double g1 = -5640.7 * GPa, g2 = 2207.1 * GPa, g3 = -332.5 * GPa, g4 = -305.2 * GPa;

    StrainCappingParams cap;
    cap.enabled = true;
    cap.limit = 0.10;
    cap.type = "smooth";
    cap.smooth_power = 4.0;
    cap.tangent_ratio = 0.1;

    double cases[4][3] = {
        {0.01, 0.0, 0.0},
        {0.05, -0.02, 0.0},
        {0.12, -0.03, 0.01},
        {0.20, -0.05, 0.02},
    };

    std::vector<double> e33_state; // cold start each time, matches Python cold-start per call

    std::cout << std::fixed << std::setprecision(6);
    for (auto& c : cases) {
        std::vector<Eigen::Matrix2d> eps(1);
        eps[0](0, 0) = c[0]; eps[0](1, 1) = c[1];
        eps[0](0, 1) = c[2]; eps[0](1, 0) = c[2];
        e33_state.clear();
        auto sig = stress_from_strain_landau_2d(
            eps, lam, mu, v1, v2, v3, g1, g2, g3, g4, "plane_stress", cap, e33_state);
        std::cout << "eps=(" << c[0] << "," << c[1] << "," << c[2] << ")  sig_GPa: "
                  << "xx=" << sig[0](0, 0) / GPa << " yy=" << sig[0](1, 1) / GPa
                  << " xy=" << sig[0](0, 1) / GPa << " yx=" << sig[0](1, 0) / GPa
                  << std::endl;
    }
    return 0;
}
