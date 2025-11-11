; ------------------------------------------------------------
; x86_64_scalar_kernel.asm
; extern "C" void x86_64_scalar_kernel(int n, int kmax, double*** trajectories)
; RCX = n, RDX = kmax, R8 = trajectories
; ------------------------------------------------------------

default rel

section .rodata align=8
C_h:        dq 1.0e-4
C_sigma:    dq 10.0
C_r:        dq 28.0
C_b:        dq 2.6666666666666665      ; 8/3
C_half_h:   dq 5.0e-5

section .text
global x86_64_scalar_kernel

x86_64_scalar_kernel:
    ; ---- Prologue (save non-volatiles). 8 pushes => keep 16B alignment ----
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

    ; ---- Args into handy regs ----
    mov     r9d, ecx          ; n
    mov     r10d, edx         ; kmax
    mov     r11, r8           ; trajectories (double***)

    ; ---- Load constants into XMM8..XMM12 ----
    movsd   xmm8,  [rel C_h]          ; h = 1e-4
    movsd   xmm9,  [rel C_sigma]      ; sigma = 10
    movsd   xmm10, [rel C_r]          ; r = 28
    movsd   xmm11, [rel C_b]          ; b = 8/3
    movsd   xmm12, [rel C_half_h]     ; half_h = h/2

    ; k = 1
    xor     r14d, r14d
    inc     r14d

.outer:
    ; if (k >= kmax+1) break;
    mov     eax, r10d
    inc     eax
    cmp     r14d, eax
    jge     .done

    ; prev = trajectories[k-1]
    mov     eax, r14d
    dec     eax
    movsxd  rax, eax
    shl     rax, 3
    mov     r12, [r11+rax]        ; r12 = prev (double**)

    ; curr = trajectories[k]
    mov     eax, r14d
    movsxd  rax, eax
    shl     rax, 3
    mov     r13, [r11+rax]        ; r13 = curr (double**)

    ; prev_x/y/z pointers
    mov     r15, [r12+0]          ; prev_x
    mov     rbx, [r12+8]          ; prev_y
    mov     rdi, [r12+16]         ; prev_z

    ; curr_x/y/z pointers
    mov     rsi, [r13+0]          ; curr_x
    mov     rax, [r13+8]          ; curr_y (keep in RAX)
    mov     rdx, [r13+16]         ; curr_z (keep in RDX)

    ; i = 0
    xor     ecx, ecx

.inner:
    cmp     ecx, r9d
    jge     .end_inner

    ; offset = i * 8
    movsxd  r8, ecx
    shl     r8, 3

    ; load x, y, z (previous step)
    movsd   xmm0, [r15+r8]        ; x
    movsd   xmm1, [rbx+r8]        ; y
    movsd   xmm2, [rdi+r8]        ; z

    ; -------- k1 = f(x,y,z) for Lorenz --------
    ; k1x = sigma*(y - x)
    movapd  xmm3, xmm1
    subsd   xmm3, xmm0
    mulsd   xmm3, xmm9            ; k1x

    ; k1y = r*x - y - x*z = x*(r - z) - y
    movapd  xmm4, xmm0
    mulsd   xmm4, xmm10           ; r*x
    subsd   xmm4, xmm1            ; r*x - y
    movapd  xmm5, xmm0
    mulsd   xmm5, xmm2            ; x*z
    subsd   xmm4, xmm5            ; k1y

    ; k1z = x*y - b*z
    movapd  xmm6, xmm0
    mulsd   xmm6, xmm1            ; x*y
    movapd  xmm7, xmm2
    mulsd   xmm7, xmm11           ; b*z
    subsd   xmm6, xmm7            ; k1z

    ; predictor state: (xp,yp,zp) = (x,y,z) + h*k1
    movapd  xmm13, xmm3
    mulsd   xmm13, xmm8
    addsd   xmm13, xmm0           ; xp

    movapd  xmm14, xmm4
    mulsd   xmm14, xmm8
    addsd   xmm14, xmm1           ; yp

    movapd  xmm15, xmm6
    mulsd   xmm15, xmm8
    addsd   xmm15, xmm2           ; zp

        ; -------- k2 = f(xp,yp,zp) --------
    ; k2x = sigma*(yp - xp)
    movapd  xmm5, xmm14
    subsd   xmm5, xmm13
    mulsd   xmm5, xmm9               ; k2x -> xmm5

    ; k2z = xp*yp - b*zp   (compute first to free zp for reuse later)
    movapd  xmm12, xmm13             ; xmm12 = xp
    mulsd   xmm12, xmm14             ; xp*yp
    movapd  xmm7,  xmm15             ; temp = zp
    mulsd   xmm7,  xmm11             ; temp = b*zp
    subsd   xmm12, xmm7              ; k2z -> xmm12

    ; k2y = r*xp - yp - xp*zp
    movapd  xmm7,  xmm13             ; xmm7 = xp
    mulsd   xmm7,  xmm10             ; r*xp
    subsd   xmm7,  xmm14             ; r*xp - yp
    mulsd   xmm15, xmm13             ; zp = zp * xp  (now xp*zp in xmm15; zp no longer needed)
    subsd   xmm7,  xmm15             ; k2y -> xmm7

    ; -------- Heun update on correct bases --------
    ; x_next = x + 0.5*h*(k1x + k2x)
    movapd  xmm13, xmm3              ; reuse xp reg as accumulator
    addsd   xmm13, xmm5
    mulsd   xmm13, [rel C_half_h]
    addsd   xmm13, xmm0
    movsd   [rsi+r8], xmm13

    ; y_next = y + 0.5*h*(k1y + k2y)
    movapd  xmm14, xmm4
    addsd   xmm14, xmm7
    mulsd   xmm14, [rel C_half_h]
    addsd   xmm14, xmm1
    movsd   [rax+r8], xmm14

    ; z_next = z + 0.5*h*(k1z + k2z)
    movapd  xmm15, xmm6
    addsd   xmm15, xmm12
    mulsd   xmm15, [rel C_half_h]
    addsd   xmm15, xmm2
    movsd   [rdx+r8], xmm15

    inc     ecx
    jmp     .inner

.end_inner:
    inc     r14d
    jmp     .outer

.done:
    ; ---- Epilogue ----
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    ret
