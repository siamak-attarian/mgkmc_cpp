#ifndef VTU_WRITER_HPP
#define VTU_WRITER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------------------------------------------
// 1. Element-centered displacement reconstruction
// -------------------------------------------------------------
inline std::vector<Eigen::Vector3d> reconstruct_displacement_element_centered(
    int nx, int ny, int nz,
    const std::vector<Eigen::Matrix3d>& eps,
    double pixel
) {
    int Nx = nx + 1;
    int Ny = ny + 1;
    int Nz = nz + 1;
    
    std::vector<Eigen::Vector3d> u(Nx * Ny * Nz, Eigen::Vector3d::Zero());
    
    auto node_idx = [&](int i, int j, int k) {
        return i * Ny * Nz + j * Nz + k;
    };
    
    auto elem_idx = [&](int i, int j, int k) {
        return i * ny * nz + j * nz + k;
    };
    
    // 1. Integrate normal strains
    // u_x from eps_xx: integrate along +x direction
    for (int i = 1; i < Nx; ++i) {
        int ix = i - 1;
        for (int j = 0; j < Ny; ++j) {
            int jy = std::min(j, ny - 1);
            for (int k = 0; k < Nz; ++k) {
                int kz = std::min(k, nz - 1);
                int idx_curr = node_idx(i, j, k);
                int idx_prev = node_idx(i - 1, j, k);
                u[idx_curr][0] = u[idx_prev][0] + eps[elem_idx(ix, jy, kz)](0, 0) * pixel;
            }
        }
    }
    
    // u_y from eps_yy: integrate along +y direction
    for (int j = 1; j < Ny; ++j) {
        int jy = j - 1;
        for (int i = 0; i < Nx; ++i) {
            int ix = std::min(i, nx - 1);
            for (int k = 0; k < Nz; ++k) {
                int kz = std::min(k, nz - 1);
                int idx_curr = node_idx(i, j, k);
                int idx_prev = node_idx(i, j - 1, k);
                u[idx_curr][1] = u[idx_prev][1] + eps[elem_idx(ix, jy, kz)](1, 1) * pixel;
            }
        }
    }
    
    // u_z from eps_zz: integrate along +z direction
    for (int k = 1; k < Nz; ++k) {
        int kz = k - 1;
        for (int i = 0; i < Nx; ++i) {
            int ix = std::min(i, nx - 1);
            for (int j = 0; j < Ny; ++j) {
                int jy = std::min(j, ny - 1);
                int idx_curr = node_idx(i, j, k);
                int idx_prev = node_idx(i, j, k - 1);
                u[idx_curr][2] = u[idx_prev][2] + eps[elem_idx(ix, jy, kz)](2, 2) * pixel;
            }
        }
    }
    
    // 2. Shear contributions (using γ = 2 * ε)
    // u_x shear contribution
    for (int j = 1; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int ix = std::min(i, nx - 1);
            for (int k = 0; k < Nz; ++k) {
                int kz = std::min(k, nz - 1);
                double g_xy = eps[elem_idx(ix, j - 1, kz)](0, 1);
                u[node_idx(i, j, k)][0] += 0.5 * g_xy * pixel;
            }
        }
    }
    for (int k = 1; k < Nz; ++k) {
        for (int i = 0; i < Nx; ++i) {
            int ix = std::min(i, nx - 1);
            for (int j = 0; j < Ny; ++j) {
                int jy = std::min(j, ny - 1);
                double g_xz = eps[elem_idx(ix, jy, k - 1)](0, 2);
                u[node_idx(i, j, k)][0] += 0.5 * g_xz * pixel;
            }
        }
    }
    
    // u_y shear contribution
    for (int i = 1; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int jy = std::min(j, ny - 1);
            for (int k = 0; k < Nz; ++k) {
                int kz = std::min(k, nz - 1);
                double g_xy = eps[elem_idx(i - 1, jy, kz)](0, 1);
                u[node_idx(i, j, k)][1] += 0.5 * g_xy * pixel;
            }
        }
    }
    for (int k = 1; k < Nz; ++k) {
        for (int i = 0; i < Nx; ++i) {
            int ix = std::min(i, nx - 1);
            for (int j = 0; j < Ny; ++j) {
                int jy = std::min(j, ny - 1);
                double g_yz = eps[elem_idx(ix, jy, k - 1)](1, 2);
                u[node_idx(i, j, k)][1] += 0.5 * g_yz * pixel;
            }
        }
    }
    
    // u_z shear contribution
    for (int i = 1; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int jy = std::min(j, ny - 1);
            for (int k = 0; k < Nz; ++k) {
                int kz = std::min(k, nz - 1);
                double g_xz = eps[elem_idx(i - 1, jy, kz)](0, 2);
                u[node_idx(i, j, k)][2] += 0.5 * g_xz * pixel;
            }
        }
    }
    for (int j = 1; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int ix = std::min(i, nx - 1);
            for (int k = 0; k < Nz; ++k) {
                int kz = std::min(k, nz - 1);
                double g_yz = eps[elem_idx(ix, j - 1, kz)](1, 2);
                u[node_idx(i, j, k)][2] += 0.5 * g_yz * pixel;
            }
        }
    }
    
    // 3. Fix Visual Anchor: left boundary (x=0)
    // u0 = mean of u[0, :, :, :]
    Eigen::Vector3d u0 = Eigen::Vector3d::Zero();
    int face_nodes = Ny * Nz;
    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            u0 += u[node_idx(0, j, k)];
        }
    }
    u0 /= face_nodes;
    
    for (int i = 0; i < Nx * Ny * Nz; ++i) {
        u[i] -= u0;
    }
    
    return u;
}

// -------------------------------------------------------------
// 2. Export 3D and 2D arrays to XML VTU
// -------------------------------------------------------------
inline void export_to_vtu(
    const std::string& filename,
    int nx, int ny, int nz,
    const std::vector<Eigen::Matrix3d>& eps,
    const std::vector<Eigen::Matrix3d>& sig,
    const std::vector<double>& E_field,
    const std::vector<double>& nu_field,
    double pixel
) {
    std::ofstream fs(filename);
    if (!fs.is_open()) {
        throw std::runtime_error("Failed to open VTU file for writing: " + filename);
    }
    
    int Nx = nx + 1;
    int Ny = ny + 1;
    int Nz = nz + 1;
    int num_points = Nx * Ny * Nz;
    int num_cells = nx * ny * nz;
    
    fs << "<?xml version=\"1.0\"?>\n";
    fs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    fs << "  <UnstructuredGrid>\n";
    fs << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\"" << num_cells << "\">\n";
    
    // 1. Point Data (Displacements)
    std::vector<Eigen::Vector3d> u = reconstruct_displacement_element_centered(nx, ny, nz, eps, pixel);
    fs << "      <PointData Vectors=\"displacement\">\n";
    fs << "        <DataArray type=\"Float64\" Name=\"displacement\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < num_points; ++i) {
        fs << u[i][0] << " " << u[i][1] << " " << u[i][2] << "\n";
    }
    fs << "        </DataArray>\n";
    fs << "      </PointData>\n";
    
    // 2. Cell Data
    fs << "      <CellData>\n";
    
    auto print_cell_scalar = [&](const std::string& name, const std::vector<double>& values) {
        fs << "        <DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (int i = 0; i < num_cells; ++i) {
            fs << values[i] << (i % 10 == 9 || i == num_cells - 1 ? "\n" : " ");
        }
        fs << "        </DataArray>\n";
    };
    
    print_cell_scalar("E", E_field);
    print_cell_scalar("nu", nu_field);
    
    // Invariant and component variables
    std::vector<double> eps_xx(num_cells), eps_yy(num_cells), eps_zz(num_cells);
    std::vector<double> eps_xy(num_cells), eps_xz(num_cells), eps_yz(num_cells);
    std::vector<double> eps_vm(num_cells);
    
    std::vector<double> sig_xx(num_cells), sig_yy(num_cells), sig_zz(num_cells);
    std::vector<double> sig_xy(num_cells), sig_xz(num_cells), sig_yz(num_cells);
    std::vector<double> sig_vm(num_cells);
    
    for (int idx = 0; idx < num_cells; ++idx) {
        eps_xx[idx] = eps[idx](0, 0);
        eps_yy[idx] = eps[idx](1, 1);
        eps_zz[idx] = eps[idx](2, 2);
        eps_xy[idx] = eps[idx](0, 1);
        eps_xz[idx] = eps[idx](0, 2);
        eps_yz[idx] = eps[idx](1, 2);
        
        double tr_eps = eps[idx].trace();
        Eigen::Matrix3d eps_dev = eps[idx] - Eigen::Matrix3d::Identity() * (tr_eps / 3.0);
        eps_vm[idx] = std::sqrt((2.0 / 3.0) * eps_dev.squaredNorm());
        
        sig_xx[idx] = sig[idx](0, 0);
        sig_yy[idx] = sig[idx](1, 1);
        sig_zz[idx] = sig[idx](2, 2);
        sig_xy[idx] = sig[idx](0, 1);
        sig_xz[idx] = sig[idx](0, 2);
        sig_yz[idx] = sig[idx](1, 2);
        
        double tr_sig = sig[idx].trace();
        Eigen::Matrix3d sig_dev = sig[idx] - Eigen::Matrix3d::Identity() * (tr_sig / 3.0);
        sig_vm[idx] = std::sqrt(1.5 * sig_dev.squaredNorm());
    }
    
    print_cell_scalar("eps_xx", eps_xx);
    print_cell_scalar("eps_yy", eps_yy);
    print_cell_scalar("eps_zz", eps_zz);
    print_cell_scalar("eps_xy", eps_xy);
    print_cell_scalar("eps_xz", eps_xz);
    print_cell_scalar("eps_yz", eps_yz);
    print_cell_scalar("eps_vm", eps_vm);
    
    print_cell_scalar("sig_xx", sig_xx);
    print_cell_scalar("sig_yy", sig_yy);
    print_cell_scalar("sig_zz", sig_zz);
    print_cell_scalar("sig_xy", sig_xy);
    print_cell_scalar("sig_xz", sig_xz);
    print_cell_scalar("sig_yz", sig_yz);
    print_cell_scalar("sig_vm", sig_vm);
    
    fs << "      </CellData>\n";
    
    // 3. Points
    fs << "      <Points>\n";
    fs << "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j <= ny; ++j) {
            for (int k = 0; k <= nz; ++k) {
                fs << i * pixel << " " << j * pixel << " " << k * pixel << "\n";
            }
        }
    }
    fs << "        </DataArray>\n";
    fs << "      </Points>\n";
    
    // 4. Cells
    fs << "      <Cells>\n";
    fs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    int NYZ = (ny + 1) * (nz + 1);
    int NZ = nz + 1;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int n0 = i * NYZ + j * NZ + k;
                int n1 = n0 + 1;
                int n2 = n0 + NZ;
                int n3 = n2 + 1;
                int n4 = n0 + NYZ;
                int n5 = n4 + 1;
                int n6 = n4 + NZ;
                int n7 = n6 + 1;
                fs << n0 << " " << n1 << " " << n3 << " " << n2 << " "
                   << n4 << " " << n5 << " " << n7 << " " << n6 << "\n";
            }
        }
    }
    fs << "        </DataArray>\n";
    
    fs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= num_cells; ++i) {
        fs << i * 8 << (i % 10 == 0 || i == num_cells ? "\n" : " ");
    }
    fs << "        </DataArray>\n";
    
    fs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 1; i <= num_cells; ++i) {
        fs << 12 << (i % 10 == 0 || i == num_cells ? "\n" : " "); // 12 = VTK_HEXAHEDRON
    }
    fs << "        </DataArray>\n";
    fs << "      </Cells>\n";
    
    fs << "    </Piece>\n";
    fs << "  </UnstructuredGrid>\n";
    fs << "</VTKFile>\n";
}

inline void export_to_vtu_2d(
    const std::string& filename,
    int nx, int ny,
    const std::vector<Eigen::Matrix2d>& eps_2d,
    const std::vector<Eigen::Matrix2d>& sig_2d,
    const std::vector<double>& E_field,
    const std::vector<double>& nu_field,
    double pixel
) {
    int N = nx * ny;
    std::vector<Eigen::Matrix3d> eps_3d(N);
    std::vector<Eigen::Matrix3d> sig_3d(N);
    for (int i = 0; i < N; ++i) {
        eps_3d[i] = Eigen::Matrix3d::Zero();
        eps_3d[i].block<2,2>(0,0) = eps_2d[i];
        
        sig_3d[i] = Eigen::Matrix3d::Zero();
        sig_3d[i].block<2,2>(0,0) = sig_2d[i];
    }
    export_to_vtu(filename, nx, ny, 1, eps_3d, sig_3d, E_field, nu_field, pixel);
}

#endif // VTU_WRITER_HPP
