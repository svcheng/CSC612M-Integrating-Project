#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

extern void x86_64_scalar_kernel();

int main() {
	x86_64_scalar_kernel();
	return 0;
}