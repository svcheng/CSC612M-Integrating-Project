# Accelerating Heun’s Method for the Lorenz System Using AVX2 SIMD on x86-64 CPUs

## 1. Project Overview

This project implements a high-performance numerical solver for the **Lorenz system of differential equations** using **Heun’s Method**, featuring three separate kernel implementations:

| Kernel | Language | Description |
|--------|----------|-------------|
| **C Reference Kernel** | C | Baseline implementation for correctness validation |
| **x86-64 Scalar Kernel** | Hand-written Assembly | Optimized non-SIMD implementation |
| **x86-64 SIMD Kernel (AVX2)** | Hand-written Assembly | Processes 4 trajectories at a time using `YMM` registers |

The goal of this project is to benchmark how computation speed improves when transitioning from high-level scalar code → low-level assembly → full SIMD vectorization.

<br>

### Repository Layout

```
/ (root)
├─ main.c                    # Baseline C kernel, timing, correctness checks
├─ x86_64_scalar_kernel.asm  # scalar x86-64 kernel
├─ x86_64_simd_kernel.asm    # YMM, 256-bit kernel
└─ README.md                 # this file
```

---

## 2. Project Description

An ordinary differential equation (ODE) is an equation of the form

$$u'=f(t,u)$$

where $u$ is an unknown, vector-valued function of time, and $f$ defines how $u$ changes with respect to $t$. ODEs are widely used to describe the behavior of real-world systems that evolve over time, such as population dynamics, electrical circuits, and weather patterns.

In most cases, ODEs cannot be solved analytically (Heath, 2018). Instead, we rely on numerical methods to obtain approximate solutions. The **Initial Value Problem (IVP)** seeks the function $u(t)$ satisfying a given initial condition $u(t_0)=u_0$. To approximate the continuous solution, we compute estimates of $u$ at discrete, evenly spaced time points $t_1, t_2, \dots, t_{max}$ using a chosen step size $h$.

One widely used method for solving IVPs is **Heun's algorithm**, also known as the improved Euler Method. It iteratively computes approximate solutions $u_k\approx u(t_k)$ according to the rule:

$$u_{k+1}=u_k+\frac{h}{2}(k_1+k_2)$$

where $u_k$ is the approximation of $u(t_k)$, $h$ is the gap between time points, $k_1=f(t_k,u_k)$ is the initial slope, and $k_2=f(t_k+h,u_k+hk_1)$ is the corrected slope based on a prediction at the next step. This two-stage process makes Heun’s method second-order accurate, providing a good balance between computational efficiency and precision for many practical systems.

A particularly interesting application of Heun’s method is the Lorenz system, a classic model of atmospheric convection introduced by Edward Lorenz in 1963 (University of Toronto, n.d.). It is defined by the nonlinear equations:

$$\begin{bmatrix}
x' \\ 
y' \\ 
z'\end{bmatrix}=\begin{bmatrix} 
\sigma(y-x) \\ 
rx-y-xz \\ 
xy-bz\end{bmatrix}$$

where $\sigma,r,b\in \mathbb R$ are parameters that determine the system's behavior. Despite its simple form, the Lorenz system exhibits chaotic dynamics, meaning small differences in initial conditions lead to drastically different trajectories over time.

This project presents a parallel implementation of Heun's algorithm for simulating an ensemble of $n$ trajectories of the Lorenz system. Each trajectory is initialized with slightly different starting conditions. We implement parallelization using the SIMD capabilities of x86-64 processors via the AVX2 instruction set. We measure the execution time of this implementation and compare its effectiveness relative to serial implementations in C and x86-64 assembly.

The program will accept as input an $n\times3$ array of initial values, the number of time steps $k_{max}$, and the integration step size $h$. Its output will be an $(n\times k_{max}+1)\times3$ array containing the approximate states of all trajectories at the time points $t_1, t_2, \dots, t_{max}$.

---

## 3. Program Implementation

The project contains three kernels that all implement the same numerical method (Heun’s method for the Lorenz system), but at different abstraction levels and with different performance characteristics:

1. A **C reference kernel** for correctness and clarity
2. A **hand-written x86-64 scalar kernel** that optimizes the same algorithm at the instruction level
3. A **hand-written x86-64 AVX2 SIMD kernel** that vectorizes the inner loop across multiple trajectories

All kernels operate on the same data structure:

* `double*** trajectories` is a 3D buffer of size `(kmax + 1) × 3 × n`, allocated in `main.c`.
* Indexing convention:

  * `trajectories[k][0][i]` → ( $x_i(t_k)$ )
  * `trajectories[k][1][i]` → ( $y_i(t_k)$ )
  * `trajectories[k][2][i]` → ( $z_i(t_k)$ ) 

This corresponds to a **structure-of-arrays (SoA)** layout across coordinates (`x`, `y`, `z`) and **contiguous arrays over trajectories**, which is friendly for SIMD. 

<br>

### 3.1 Baseline C Kernel (Reference Implementation)

The C reference kernel is implemented in `C_kernel` in `main.c`. It is a direct translation of Heun’s method into nested loops and serves as the ground truth for both the scalar and SIMD assembly implementations. 

Function signature:

```c
void C_kernel(int n, int kmax, double*** trajectories)
```

Core structure:

* **Outer loop over time steps** `k = 1 .. kmax`
* **Inner loop over trajectories** `i = 0 .. n-1`

For each trajectory ( i ) at time step ( k-1 ):

1. Load state:

   ```c
   x = trajectories[k - 1][0][i];
   y = trajectories[k - 1][1][i];
   z = trajectories[k - 1][2][i];
   ```

2. Compute **k₁** (slope at current state):

   ```c
   k1_x = sigma * (y - x);
   k1_y = r * x - y - x * z;
   k1_z = x * y - b * z;
   ```

3. Predictor step to provisional state ((x_p, y_p, z_p)):

   ```c
   xp = x + h * k1_x;
   yp = y + h * k1_y;
   zp = z + h * k1_z;
   ```

4. Compute **k₂** (slope at predictor state):

   ```c
   k2_x = sigma * (yp - xp);
   k2_y = r * xp - yp - xp * zp;
   k2_z = xp * yp - b * zp;
   ```

5. Apply Heun’s update (second-order correction):

   ```c
   trajectories[k][0][i] = x + (h / 2) * (k1_x + k2_x);
   trajectories[k][1][i] = y + (h / 2) * (k1_y + k2_y);
   trajectories[k][2][i] = z + (h / 2) * (k1_z + k2_z);
   ```

This version is easy to read and verify but does not exploit any low-level hardware features. It is used as:

* A **reference solution** for numerical correctness
* A **baseline** for performance comparisons printed in the “Performance Summary” section of `main`. 

<br>

### 3.2 x86-64 Scalar Kernel (Hand-Written Assembly)

The scalar kernel translates the same Heun update into x86-64 assembly using SSE scalar floating-point instructions, still processing **one trajectory at a time**, but with more control over registers and memory access. It is defined in `x86_64_scalar_kernel.asm` as: 

```asm
; extern "C" void x86_64_scalar_kernel(int n, int kmax, double*** trajectories)
; RCX = n, RDX = kmax, R8 = trajectories
global x86_64_scalar_kernel
```

#### Control Flow and Loop Structure

The overall algorithm mirrors the C version:

* `r14d` is used as the **time-step counter** `k`
* `ecx` is used as the **trajectory index** `i`

Outer loop (`.outer` label):

1. Computes `prev = trajectories[k-1]` and `curr = trajectories[k]` using pointer arithmetic:

   ```asm
   mov     eax, r14d
   dec     eax
   movsxd  rax, eax
   shl     rax, 3
   mov     r12, [r11+rax]    ; r12 = prev (double**)

   mov     eax, r14d
   movsxd  rax, eax
   shl     rax, 3
   mov     r13, [r11+rax]    ; r13 = curr (double**)
   ```

2. Extracts coordinate pointers for `prev` and `curr`:

   ```asm
   mov     r15, [r12+0]      ; prev_x
   mov     rbx, [r12+8]      ; prev_y
   mov     rdi, [r12+16]     ; prev_z

   mov     rsi, [r13+0]      ; curr_x
   mov     rax, [r13+8]      ; curr_y
   mov     rdx, [r13+16]     ; curr_z
   ```

Inner loop (`.inner` label):

* For each index `i`:

  * Compute byte offset `offset = i * 8` (double is 8 bytes)
  * Load `x, y, z` from `prev_x/prev_y/prev_z`
  * Compute `k1`, predictor, `k2`, and final update in SSE registers

#### Constant Handling and Arithmetic

The kernel loads constants once into `xmm` registers:

```asm
movsd   xmm8,  [rel C_h]        ; h = 1e-4
movsd   xmm9,  [rel C_sigma]    ; sigma = 10
movsd   xmm10, [rel C_r]        ; r = 28
movsd   xmm11, [rel C_b]        ; b = 8/3
movsd   xmm12, [rel C_half_h]   ; half_h = h/2
```

Example: computing `k1` and predictor state for one trajectory:

```asm
; load x, y, z
movsd   xmm0, [r15+r8]      ; x
movsd   xmm1, [rbx+r8]      ; y
movsd   xmm2, [rdi+r8]      ; z

; k1x = sigma*(y - x)
movapd  xmm3, xmm1
subsd   xmm3, xmm0
mulsd   xmm3, xmm9

; predictors xp, yp, zp = (x,y,z) + h*k1
movapd  xmm13, xmm3
mulsd   xmm13, xmm8
addsd   xmm13, xmm0         ; xp
```

Once `k2` is computed, Heun’s update follows:

```asm
; x_next = x + 0.5*h*(k1x + k2x)
movapd  xmm13, xmm3         ; k1x
addsd   xmm13, xmm5         ; + k2x
mulsd   xmm13, [rel C_half_h]
addsd   xmm13, xmm0         ; + x
movsd   [rsi+r8], xmm13     ; store x_next
```

#### Purpose of the Scalar Kernel

* **Same algorithm** as the C kernel, but with:

  * Fewer redundant loads/stores
  * Explicit control over register usage and instruction sequence
* Used to:

  * Measure the performance gain of low-level optimization **without** SIMD
  * Verify that numerical output matches the C kernel within a tight tolerance (via `compare_outputs` in `main.c`). 

<br>

### 3.3 x86-64 AVX2 SIMD Kernel (Vectorized Across Trajectories)

The AVX2 kernel is where the main **parallel algorithm** lives. It takes the innermost loop over trajectories and converts it into a **data-parallel SIMD loop** that processes **4 trajectories per iteration** using 256-bit `ymm` registers. The time-stepping remains sequential (in `k`), but each step advances multiple trajectories simultaneously. 

Function signature and calling convention:

```asm
; extern "C" void x86_64_simd_kernel(int n, int kmax, double*** trajectories)
; RCX = n, RDX = kmax, R8 = trajectories
global x86_64_simd_kernel
```

#### Parallel Algorithm: What Became Parallel?

In the C kernel, the core computation is:

```c
for (int k = 1; k < kmax + 1; k++) {
    for (int i = 0; i < n; i++) {
        // compute k1, predictor, k2, and update for trajectory i
    }
}
```

* The **outer loop over time steps `k`** is inherently sequential:
  Each step depends on the previous one (`trajectories[k]` uses `trajectories[k-1]`).
* The **inner loop over trajectories `i`**, however, operates on **independent trajectories**:
  There are no data dependencies between different `i`.

**The SIMD kernel parallelizes exactly this inner loop.**

Instead of computing one trajectory at a time (`i += 1`), the AVX2 kernel processes **4 trajectories in lockstep** (`i += 4`) by:

* Loading `x`, `y`, and `z` for indices `i, i+1, i+2, i+3` into `ymm` vectors
* Performing the entire Heun update (k₁, predictor, k₂, and final correction) on these vectors

This is a classic **data-parallel** scheme: multiple independent ODE solves in parallel, with the time integration still sequential in `k`.

#### Vectorized Inner Loop

The label `.inner_vec` implements the vectorized portion:

1. **Boundary check** for the vectorized chunk:

   ```asm
   mov     r8d, ecx
   add     r8d, 4
   cmp     r8d, r9d
   jg      .inner_scalar      ; if i+4 > n, switch to scalar tail
   ```

2. Compute byte offset `offset = i * 8` and load four `x`, `y`, `z` values:

   ```asm
   movsxd  r8, ecx
   shl     r8, 3              ; offset = i * 8

   vmovupd ymm0, [r15+r8]     ; x[i..i+3]
   vmovupd ymm1, [rbx+r8]     ; y[i..i+3]
   vmovupd ymm2, [rdi+r8]     ; z[i..i+3]
   ```

3. Compute **k₁** in vector form:

   ```asm
   ; k1x = sigma * (y - x)
   vsubpd  ymm1, ymm1, ymm0      ; y - x
   vbroadcastsd ymm3, [C_sigma]
   vmulpd  ymm3, ymm3, ymm1      ; k1x -> ymm3

   ; k1y, k1z similarly, using vmulpd / vsubpd
   ```

4. Store `k₁` on the stack for later reuse in the Heun update:

   ```asm
   vmovdqu [rsp+0],  ymm3      ; k1x
   vmovdqu [rsp+32], ymm4      ; k1y
   vmovdqu [rsp+64], ymm5      ; k1z
   ```

5. Compute **predictor state** ((x_p, y_p, z_p)) for all 4 trajectories:

   ```asm
   vbroadcastsd ymm1, [C_h]    ; h

   ; xp = x + h * k1x
   vmovupd ymm0, [r15+r8]      ; base x
   vmovdqu ymm3, [rsp+0]       ; k1x
   vmulpd  ymm3, ymm3, ymm1
   vaddpd  ymm0, ymm0, ymm3    ; xp in ymm0

   ; yp, zp similar with k1y, k1z
   ```

6. Compute **k₂** in vector form:

   ```asm
   ; k2x = sigma*(yp - xp)
   vsubpd  ymm3, ymm1, ymm0
   vbroadcastsd ymm4, [C_sigma]
   vmulpd  ymm3, ymm4, ymm3    ; k2x

   ; k2y, k2z similarly
   ```

7. Apply **Heun’s update** for 4 trajectories at once:

   ```asm
   ; x_next = x + 0.5*h*(k1x + k2x)
   vbroadcastsd ymm1, [C_half_h]
   vmovupd ymm0, [r15+r8]      ; base x
   vmovdqu ymm2, [rsp+0]       ; k1x
   vaddpd  ymm2, ymm2, ymm3    ; k1x + k2x
   vmulpd  ymm2, ymm2, ymm1
   vaddpd  ymm0, ymm0, ymm2
   vmovupd [rsi+r8], ymm0      ; store x_next

   ; y_next, z_next done similarly using k1y/k2y and k1z/k2z
   ```

8. Advance `i` by 4:

   ```asm
   add     ecx, 4
   jmp     .inner_vec
   ```

This loop is the **parallel counterpart** to the sequential inner `for (int i = 0; i < n; i++)` in the C kernel.

#### Scalar Tail for Remaining Elements

If `n` is not a multiple of 4, a scalar tail loop (`.inner_scalar`) handles the remaining 1–3 trajectories using scalar SSE instructions, closely resembling the scalar kernel’s logic. This guarantees:

* No out-of-bounds memory access
* Correct results for arbitrary `n`

```asm
.inner_scalar:
    cmp     ecx, r9d
    jge     .end_inner
    ; ... scalar (non-SIMD) version of k1, predictor, k2, update ...
    inc     ecx
    jmp     .inner_scalar
```

#### Additional Implementation Details

* **Stack scratch space**: The SIMD kernel reserves 128 bytes on the stack to spill intermediate vectors (`k1x`, `k1y`, `k1z`) for reuse.

  ```asm
  sub     rsp, 128
  ...
  add     rsp, 128
  ```
* **AVX–SSE transition cleanup**: It uses `vzeroupper` before returning to avoid penalties when mixing AVX and legacy SSE instructions:

  ```asm
  vzeroupper
  ```
* **Same numerical behavior**: Like the scalar kernel, the SIMD kernel’s output is compared against the C reference using `compare_outputs` with a tight tolerance (`tol = 1e-12`) in `main.c`. 

<br>

### 3.4 Orchestration and Benchmarking (`main.c`)

The three kernels are driven and benchmarked by `main.c`: 

* Allocates three independent trajectory buffers (`tr_c`, `tr_asm`, `tr_simd`)
* Initializes consistent initial conditions with `reset_trajectories`
* Times each kernel over `TEST_NUM` runs using `QueryPerformanceCounter`
* Prints an average runtime and a **PASS/FAIL** correctness check against the C kernel

This provides a simple but complete framework to:

* Verify that all kernels implement the **same numerical algorithm**, and
* Measure performance gains when moving from:

  * high-level C → scalar ASM → AVX2 SIMD (data-parallel)



