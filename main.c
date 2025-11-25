#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

extern void x86_64_scalar_kernel(int n, int kmax, double*** trajectories);

extern void x86_64_simd_kernel(int n, int kmax, double*** trajectories);

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
static void print_trajectories(double*** trajectories, int kmax, int n) {
    int tmax = kmax + 1;

    for (int i = 0; i < n && i < 4; i++) {   // print up to 4 trajectories
        printf("  ------------ Trajectory %d ------------\n", i + 1);

        for (int j = 0; j < 3; j++) {
            printf("  %c: [", (j == 0 ? 'x' : (j == 1 ? 'y' : 'z')));

            if (tmax <= 5) {
                // --- Print all values if short ---
                for (int k = 0; k < tmax; k++) {
                    printf("%.6f", trajectories[k][j][i]);
                    if (k < tmax - 1) printf(", ");
                }
            }
            else {
                // --- First 5 ---
                for (int k = 0; k < 5; k++) {
                    printf("%.6f", trajectories[k][j][i]);
                    printf(", ");
                }

                printf("..., ");

                // --- Last 5 ---
                for (int k = tmax - 5; k < tmax; k++) {
                    printf("%.6f", trajectories[k][j][i]);
                    if (k < tmax - 1) printf(", ");
                }
            }

            printf("]\n");
        }

        printf("\n");
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

/* Reset whole trajectory buffer + reapply initial conditions */
static void reset_trajectories(double*** tr, int kmax, int n) {
    // Zero-out everything
    for (int k = 0; k <= kmax; k++) {
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < n; i++) {
                tr[k][j][i] = 0.0;
            }
        }
    }

    // Reapply initial conditions at k = 0
    for (int i = 0; i < n; i++) {
        tr[0][0][i] = i * 0.0001;       // x0
        tr[0][1][i] = 1.0 + i * 0.0001; // y0
        tr[0][2][i] = i * 0.0001;       // z0
    }
}


/* ---------- compare two trajectories: report max |diff| ---------- */
static boolean compare_outputs(double*** A, double*** B, int kmax, int n) {
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
    
    double tol = 1e-12;  // tune as needed
    if (max_abs <= tol) {
        return TRUE;
    }
    
    return FALSE;
}

/* ---------- main: run C and ASM into separate buffers, compare ---------- */
int main(void) {
    printf("=================================================== RUN INFO ===================================================\n");

	const int n = 10;          // number of trajectories
	const int kmax = 1 << 5;  // number of time steps to be computed
	printf("Number of trajectories      : %d\n", n);
	printf("Number of time steps        : %d\n", kmax);

    // Declare test variables
    const int TEST_NUM = 30;
    int j;
	printf("Number of tests per kernel  : %d\n\n", TEST_NUM);

    // Declare the timer variable
    LARGE_INTEGER li;
    long long int start, end;
	double PCFreq, c_time, asm_time, simd_time;
	QueryPerformanceFrequency(&li);
    PCFreq = (double)(li.QuadPart);

	// Performance summary
    printf("============================================== CORRECTNESS CHECKS =============================================\n");

    // Allocate buffers and set initial conditions
    double*** tr_c = alloc_tr(kmax, n);
    printf("Allocated buffer 1\n");
    double*** tr_asm = alloc_tr(kmax, n);
    printf("Allocated buffer 2\n");
	double*** tr_simd = alloc_tr(kmax, n);
	printf("Allocated buffer 3\n");
    reset_trajectories(tr_c, kmax, n);
    reset_trajectories(tr_asm, kmax, n);
	reset_trajectories(tr_simd, kmax, n);

    // FIRST KERNEL: PROGRAM IN C
    c_time = 0.0;
	for (j = 0; j < TEST_NUM; j++) {
        // Reset
        reset_trajectories(tr_c, kmax, n);

        // Start timer
        QueryPerformanceCounter(&li);
        start = li.QuadPart;

        // Run C kernel
        C_kernel(n, kmax, tr_c);

        // Stop timer
        QueryPerformanceCounter(&li);
        end = li.QuadPart;
        c_time += ((double)(end - start)) * 1000.0 / PCFreq;
    }
    printf("[ C Reference Kernel ] ----------------------------------------------------------------------------------------\n");
    printf("  Output Status:      PASS");
    printf("\n\n  C Kernel Trajectories:\n");
    print_trajectories(tr_c, kmax, n);
    
	// SECOND KERNEL: x86_64 SCALAR KERNEL
	asm_time = 0.0;
    for (j = 0; j < TEST_NUM; j++) {
        // Reset
        reset_trajectories(tr_asm, kmax, n);

        // Start timer
        QueryPerformanceCounter(&li);
        start = li.QuadPart;

        // Run x86_64 scalar kernel
        x86_64_scalar_kernel(n, kmax, tr_asm);

        // Stop timer
        QueryPerformanceCounter(&li);
        end = li.QuadPart;
        asm_time += ((double)(end - start)) * 1000.0 / PCFreq;
    }
    printf("[ x86-64 Scalar Kernel ]----------------------------------------------------------------------------\n");
    if (compare_outputs(tr_c, tr_asm, kmax, n)) {
        printf("  Output Status:      PASS");
    } else {
        printf("  Output Status:      FAIL");
	}
    printf("\n\n  x86-64 Scalar Kernel Trajectories:\n");
    print_trajectories(tr_asm, kmax, n);
    
	// THIRD KERNEL: x86_64 SIMD KERNEL
	simd_time = 0.0;
    for (j = 0; j < TEST_NUM; j++) {
        // Reset
        reset_trajectories(tr_asm, kmax, n);

        // Start timer
        QueryPerformanceCounter(&li);
        start = li.QuadPart;

        // Run x86_64 SIMD kernel
        x86_64_simd_kernel(n, kmax, tr_asm);

        // Stop timer
        QueryPerformanceCounter(&li);
        end = li.QuadPart;
		simd_time += ((double)(end - start)) * 1000.0 / PCFreq;
	}
	printf("[ x86-64 SIMD Kernel ] ------------------------------------------------------------------------------\n");
	if (compare_outputs(tr_c, tr_asm, kmax, n)) {
		printf("  Output Status:      PASS");
    } else {
        printf("  Output Status:      FAIL");
    }
    printf("\n\n  x86-64 SIMD Trajectories:\n");
    print_trajectories(tr_asm, kmax, n);

    printf("\n");
    printf("============================================== PERFORMANCE SUMMARY =============================================\n\n");
    printf("  C Kernel Average Time:                   %f ms\n", c_time / TEST_NUM);
    printf("  x86-64 Scalar Kernel Average Time:       %f ms\n", asm_time / TEST_NUM);
    printf("  x86-64 SIMD Kernel Average Time:         %f ms\n", simd_time / TEST_NUM);

    // cleanup
    free_tr(kmax, tr_c);
    free_tr(kmax, tr_asm);
	free_tr(kmax, tr_simd);
    return 0;
}
