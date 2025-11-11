#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

extern void x86_64_scalar_kernel(int n, int kmax, double*** trajectories);

/* ---------- C kernel ---------- */
void C_kernel(int n, int kmax, double*** trajectories)
{
    const double h = 1e-4;
    const double sigma = 10.0;
    const double r = 28.0;
    const double b = 8.0 / 3.0;

    double x, y, z, xp, yp, zp, k1_x, k1_y, k1_z, k2_x, k2_y, k2_z;
    for (int k = 1; k < kmax + 1; k++) {
        for (int i = 0; i < n; i++) {
            x = trajectories[k - 1][0][i];
            y = trajectories[k - 1][1][i];
            z = trajectories[k - 1][2][i];

            k1_x = sigma * (y - x);
            k1_y = r * x - y - x * z;
            k1_z = x * y - b * z;

            xp = x + h * k1_x;
            yp = y + h * k1_y;
            zp = z + h * k1_z;

            k2_x = sigma * (yp - xp);
            k2_y = r * xp - yp - xp * zp;
            k2_z = xp * yp - b * zp;

            trajectories[k][0][i] = x + (h / 2) * (k1_x + k2_x);
            trajectories[k][1][i] = y + (h / 2) * (k1_y + k2_y);
            trajectories[k][2][i] = z + (h / 2) * (k1_z + k2_z);
        }
    }
}

/* ---------- printing ---------- */
static void print_trajectories(double*** trajectories, int kmax) {
    for (int i = 0; i < 4; i++) {
        printf("------------ Trajectory %d ------------\n", i + 1);
        for (int j = 0; j < 3; j++) {
            if (j == 0) printf("x: [");
            if (j == 1) printf("y: [");
            if (j == 2) printf("z: [");

            for (int k = 0; k < 5; k++) {
                printf("%.6f", trajectories[k][j][i]);
                printf(", ");
            }
            printf("..., ");
            for (int k = kmax - 4; k < kmax + 1; k++) {
                printf("%.6f", trajectories[k][j][i]);
                if (k < kmax) printf(", ");
            }
            printf("]\n");
        }
        printf("\n\n");
    }
}

/* ---------- helpers to allocate / free (kmax+1) x 3 x n ---------- */
static double*** alloc_tr(int kmax, int n) {
    double*** tr = (double***)malloc((kmax + 1) * sizeof(double**));
    if (!tr) { fprintf(stderr, "alloc_tr: level-0 malloc failed\n"); exit(1); }
    for (int k = 0; k <= kmax; ++k) {
        tr[k] = (double**)malloc(3 * sizeof(double*));
        if (!tr[k]) { fprintf(stderr, "alloc_tr: level-1 malloc failed\n"); exit(1); }
        for (int j = 0; j < 3; ++j) {
            tr[k][j] = (double*)malloc(n * sizeof(double));
            if (!tr[k][j]) { fprintf(stderr, "alloc_tr: level-2 malloc failed\n"); exit(1); }
        }
    }
    return tr;
}

static void free_tr(int kmax, double*** tr) {
    if (!tr) return;
    for (int k = 0; k <= kmax; ++k) {
        for (int j = 0; j < 3; ++j) {
            free(tr[k][j]);
        }
        free(tr[k]);
    }
    free(tr);
}

/* ---------- set identical initial conditions at k=0 ---------- */
static void init_k0(double*** tr, int n) {
    for (int i = 0; i < n; ++i) {
        tr[0][0][i] = i * 0.0001;       // x0
        tr[0][1][i] = 1.0 + i * 0.0001; // y0
        tr[0][2][i] = i * 0.0001;       // z0
    }
}

/* ---------- compare two trajectories: report max |diff| ---------- */
static void compare_outputs(double*** A, double*** B, int kmax, int n) {
    double max_abs = 0.0;
    double max_abs_x = 0.0, max_abs_y = 0.0, max_abs_z = 0.0;
    int max_k = 0, max_i = 0, max_c = 0;

    for (int k = 0; k <= kmax; ++k) {
        for (int i = 0; i < n; ++i) {
            double dx = fabs(A[k][0][i] - B[k][0][i]);
            double dy = fabs(A[k][1][i] - B[k][1][i]);
            double dz = fabs(A[k][2][i] - B[k][2][i]);

            if (dx > max_abs_x) max_abs_x = dx;
            if (dy > max_abs_y) max_abs_y = dy;
            if (dz > max_abs_z) max_abs_z = dz;

            if (dx > max_abs) { max_abs = dx; max_k = k; max_i = i; max_c = 0; }
            if (dy > max_abs) { max_abs = dy; max_k = k; max_i = i; max_c = 1; }
            if (dz > max_abs) { max_abs = dz; max_k = k; max_i = i; max_c = 2; }
        }
    }

    printf("\n=== C vs ASM comparison ===\n");
    printf("Max |diff| overall   : %.12e at (k=%d, i=%d, comp=%c)\n",
        max_abs, max_k, max_i, "xyz"[max_c]);
    printf("Max |diff| per comp  : x=%.12e  y=%.12e  z=%.12e\n",
        max_abs_x, max_abs_y, max_abs_z);

    double tol = 1e-12;  // tune as needed
    if (max_abs <= tol) {
        printf("Status: PASS (<= %.1e)\n", tol);
    }
    else {
        printf("Status: WARN (> %.1e) — likely just rounding, but inspect.\n", tol);
    }
}

/* ---------- main: run C and ASM into separate buffers, compare ---------- */
int main(void) {
    int n = 10;
    int kmax = 20000;

    // allocate two independent buffers
    double*** tr_c = alloc_tr(kmax, n);
    double*** tr_asm = alloc_tr(kmax, n);

    // set identical initial conditions in both
    init_k0(tr_c, n);
    init_k0(tr_asm, n);

    printf("Running C kernel...\n");
    C_kernel(n, kmax, tr_c);
    print_trajectories(tr_c, kmax);

    printf("Running x86_64 scalar kernel...\n");
    x86_64_scalar_kernel(n, kmax, tr_asm);
    print_trajectories(tr_asm, kmax);

    // Compare full trajectories
    compare_outputs(tr_c, tr_asm, kmax, n);

    // cleanup
    free_tr(kmax, tr_c);
    free_tr(kmax, tr_asm);
    return 0;
}
