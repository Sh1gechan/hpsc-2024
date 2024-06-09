#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;
typedef vector<vector<double>> matrix;

int main(int argc, char** argv) {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int begin = rank * (nx / size);
    int end = (rank + 1) * (nx / size);
    if (rank == size - 1) end = nx; // Last process takes any remaining rows

    matrix u(ny, vector<double>(nx, 0));
    matrix v(ny, vector<double>(nx, 0));
    matrix p(ny, vector<double>(nx, 0));
    matrix b(ny, vector<double>(nx, 0));
    matrix un(ny, vector<double>(nx, 0));
    matrix vn(ny, vector<double>(nx, 0));
    matrix pn(ny, vector<double>(nx, 0));

    ofstream ufile, vfile, pfile;
    if (rank == 0) {
        ufile.open("u.dat");
        vfile.open("v.dat");
        pfile.open("p.dat");
    }

    for (int n = 0; n < nt; n++) {
        // Compute b
        for (int j = 1; j < ny - 1; j++) {
            for (int i = begin; i < end; i++) {
                b[j][i] = rho * (1. / dt * 
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) - 
                    pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2) - 
                    2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx)) - 
                    pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2));
            }
        }

        // Pressure Poisson equation
        for (int it = 0; it < nit; it++) {
            for (int j = 0; j < ny; j++)
                for (int i = begin; i < end; i++)
                    pn[j][i] = p[j][i];

            for (int j = 1; j < ny - 1; j++) {
                for (int i = begin; i < end; i++) {
                    p[j][i] = (pow(dy, 2) * (pn[j][i+1] + pn[j][i-1]) + 
                               pow(dx, 2) * (pn[j+1][i] + pn[j-1][i]) - 
                               b[j][i] * pow(dx, 2) * pow(dy, 2)) / 
                              (2 * (pow(dx, 2) + pow(dy, 2)));
                }
            }

            // Boundary conditions for pressure
            for (int j = 0; j < ny; j++) {
                if (begin == 0) p[j][0] = p[j][1];
                if (end == nx) p[j][nx-1] = p[j][nx-2];
            }
            for (int i = begin; i < end; i++) {
                if (rank == 0) p[0][i] = p[1][i];
                if (rank == size - 1) p[ny-1][i] = 0;
            }

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &p[0][0], (end - begin) * ny, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        // Copy u, v to un, vn
        for (int j = 0; j < ny; j++) {
            for (int i = begin; i < end; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }

        // Update u, v
        for (int j = 1; j < ny - 1; j++) {
            for (int i = begin; i < end; i++) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                          un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                          dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]) +
                          nu * dt / pow(dx, 2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]) +
                          nu * dt / pow(dy, 2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);

                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                          vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                          dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]) +
                          nu * dt / pow(dx, 2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]) +
                          nu * dt / pow(dy, 2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &u[0][0], (end - begin) * ny, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &v[0][0], (end - begin) * ny, MPI_DOUBLE, MPI_COMM_WORLD);

        // Boundary conditions for u, v
        for (int j = 0; j < ny; j++) {
            if (begin == 0) u[j][0] = 0;
            if (end == nx) u[j][nx-1] = 0;
            v[j][0] = 0;
            v[j][nx-1] = 0;
        }
        for (int i = begin; i < end; i++) {
            if (rank == 0) u[0][i] = 0;
            if (rank == size - 1) u[ny-1][i] = 1;
            v[0][i] = 0;
            v[ny-1][i] = 0;
        }

        // Output results every 10 steps
        if (rank == 0 && n % 10 == 0) {
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    ufile << u[j][i] << " ";
            ufile << "\n";
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    vfile << v[j][i] << " ";
            vfile << "\n";
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    pfile << p[j][i] << " ";
            pfile << "\n";
        }
    }

    if (rank == 0) {
        ufile.close();
        vfile.close();
        pfile.close();
    }

    MPI_Finalize();
    return 0;
}

