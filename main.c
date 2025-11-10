#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

extern void x86_64_scalar_kernel();

/* 
	trajectories is an (kmax+1) x 3 x n array
*/
void C_kernel(int n, int kmax, double*** trajectories) 
{
	double h = 0.0001;
	double sigma = 10;
	double r = 28;
	double b = 8 / 3;

	double x, y, z, k1_x, k1_y, k1_z, k2_x, k2_y, k2_z;
	for (int k = 1; k < kmax + 1; k++) {
		for (int i = 0; i < n; i++) {
			x = trajectories[k - 1][0][i];
			y = trajectories[k - 1][1][i];
			z = trajectories[k - 1][2][i];

			k1_x = sigma * (y - z);
			k1_y = r * x - y + x * z;
			k1_z = x * y - b * z;

			k2_x = x + h * k1_x;
			k2_y = y + h * k1_y;
			k2_z = z + h * k1_z;

			trajectories[k][0][i] = x + (h / 2) * (k1_x + k2_x);
			trajectories[k][1][i] = z + (h / 2) * (k1_y + k2_y);
			trajectories[k][2][i] = x + (h / 2) * (k1_z + k2_z);
		}
	}
}

// print first and last 5 steps of first 4 trajectories
void print_trajectories(double*** trajectories, int kmax) {
	for (int i = 0; i < 4; i++) {
		printf("------------ Trajectory %d ------------\n", i+1);
		for (int j = 0; j < 3; j++) {
			if (j == 0)
				printf("x: [");
			if (j == 1)
				printf("y: [");
			if (j == 2)
				printf("z: [");

			for (int k = 0; k < 5; k++) {
				printf("%.6f", trajectories[k][j][i]);
				printf(", ");
			}
			printf("..., ");
			for (int k = kmax - 4; k < kmax + 1; k++) {
				printf("%.6f", trajectories[k][j][i]);
				if (k < kmax)
					printf(", ");
			}
			printf("]\n");
		}
		printf("\n\n");
	}
}

int main() {
	int n = 10;
	int kmax = 20000;
	double*** trajectories;

	// allocate trajectories array
	trajectories = (double***)malloc((kmax+1) * sizeof(double**));
	for (int i = 0; i < kmax + 1; i++) {
		trajectories[i] = (double**)malloc(3 * sizeof(double*));
		for (int j = 0; j < 3; j++) {
			trajectories[i][j] = (double*)malloc(kmax * sizeof(double));
		}
	}

	// initialize initial values
	for (int i = 0; i < n; i++) {
		trajectories[0][0][i] = i * 0.0001;
		trajectories[0][1][i] = 1 + i * 0.0001;
		trajectories[0][2][i] = i * 0.0001;
	}

	C_kernel(n, kmax, trajectories);
	print_trajectories(trajectories, kmax);
	return 0;
}