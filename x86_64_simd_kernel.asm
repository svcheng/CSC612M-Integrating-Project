; ------------------------------------------------------------
; x86_64_simd_kernel.asm  (NASM / SASM syntax, Windows x64)
; extern "C" void x86_64_simd_kernel(int n, int kmax, double*** trajectories)
; RCX = n, RDX = kmax, R8 = trajectories
; ------------------------------------------------------------

default rel

section .rodata align=32
C_h:        dq 1.0e-4
C_sigma:    dq 10.0
C_r:        dq 28.0
C_b:        dq 2.6666666666666665  ; 8/3
C_half_h:   dq 5.0e-5

section .text
global x86_64_simd_kernel

x86_64_simd_kernel:
    ; ---- Prologue (save non-volatiles). 8 pushes => keep 16B alignment ----
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    ; Reserve 128 bytes stack scratch (aligned for YMM stores)
    sub     rsp, 128

    ; ---- Args into handy regs ----
    mov     r9d, ecx          ; n
    mov     r10d, edx         ; kmax
    mov     r11, r8           ; trajectories (double***)

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
    mov     rax, [r13+8]          ; curr_y
    mov     rdx, [r13+16]         ; curr_z

    ; i = 0
    xor     ecx, ecx

.inner_vec:
    ; if (i+4 > n) go scalar tail
    mov     r8d, ecx
    add     r8d, 4
    cmp     r8d, r9d
    jg      .inner_scalar

    ; byte offset = i * 8
    movsxd  r8, ecx
    shl     r8, 3

    ; ---- Load x, y, z vectors (4 doubles each) ----
    vmovupd ymm0, [r15+r8]        ; x
    vmovupd ymm1, [rbx+r8]        ; y
    vmovupd ymm2, [rdi+r8]        ; z

    ; -------- k1 = f(x,y,z) for Lorenz --------
    ; k1x = sigma * (y - x)
    vsubpd  ymm1, ymm1, ymm0      ; y - x  (y destroyed)
    vbroadcastsd ymm3, [C_sigma]
    vmulpd  ymm3, ymm3, ymm1      ; k1x -> ymm3

    ; k1y = r*x - y - x*z = x*(r - z) - y
    vbroadcastsd ymm4, [C_r]
    vmulpd  ymm4, ymm4, ymm0      ; r*x
    ; reload y for this term
    vmovupd ymm1, [rbx+r8]        ; y
    vsubpd  ymm4, ymm4, ymm1      ; r*x - y
    vmulpd  ymm5, ymm0, ymm2      ; x*z
    vsubpd  ymm4, ymm4, ymm5      ; k1y -> ymm4

    ; k1z = x*y - b*z
    ; use mem operand for y and z to reduce reg pressure
    vmovupd ymm1, [rbx+r8]        ; y
    vmulpd  ymm5, ymm0, ymm1      ; x*y
    vbroadcastsd ymm1, [C_b]
    vmulpd  ymm1, ymm1, [rdi+r8]  ; b*z
    vsubpd  ymm5, ymm5, ymm1      ; k1z -> ymm5

    ; ---- Spill k1 vectors to stack scratch ----
    ; layout: [rsp+0]=k1x, [rsp+32]=k1y, [rsp+64]=k1z
    vmovdqu [rsp+0],  ymm3
    vmovdqu [rsp+32], ymm4
    vmovdqu [rsp+64], ymm5

    ; -------- Predictor: (xp,yp,zp) = (x,y,z) + h*k1 --------
    vbroadcastsd ymm1, [C_h]      ; reuse ymm1 as broadcast(h)

    ; xp
    vmovupd ymm0, [r15+r8]        ; x
    vmovdqu ymm3, [rsp+0]         ; k1x
    vmulpd  ymm3, ymm3, ymm1
    vaddpd  ymm0, ymm0, ymm3      ; xp -> ymm0

    ; yp
    vmovupd ymm2, [rbx+r8]        ; y  (use ymm2 temporarily)
    vmovdqu ymm4, [rsp+32]        ; k1y
    vmulpd  ymm4, ymm4, ymm1
    vaddpd  ymm1, ymm2, ymm4      ; yp -> ymm1  (reusing ymm1)

    ; zp
    vmovupd ymm2, [rdi+r8]        ; z
    vmovdqu ymm5, [rsp+64]        ; k1z
    vbroadcastsd ymm3, [C_h]
    vmulpd  ymm5, ymm5, ymm3
    vaddpd  ymm2, ymm2, ymm5      ; zp -> ymm2

    ; -------- k2 = f(xp,yp,zp) --------
    ; k2x = sigma*(yp - xp)
    vsubpd  ymm3, ymm1, ymm0      ; yp - xp
    vbroadcastsd ymm4, [C_sigma]
    vmulpd  ymm3, ymm4, ymm3      ; k2x -> ymm3

    ; k2y = r*xp - yp - xp*zp
    vbroadcastsd ymm4, [C_r]
    vmulpd  ymm4, ymm4, ymm0      ; r*xp
    vsubpd  ymm4, ymm4, ymm1      ; r*xp - yp
    vmulpd  ymm5, ymm0, ymm2      ; xp*zp
    vsubpd  ymm4, ymm4, ymm5      ; k2y -> ymm4

    ; k2z = xp*yp - b*zp
    vmulpd  ymm5, ymm0, ymm1      ; xp*yp
    vbroadcastsd ymm1, [C_b]
    vmulpd  ymm1, ymm1, ymm2      ; b*zp
    vsubpd  ymm5, ymm5, ymm1      ; k2z -> ymm5

    ; -------- Heun update --------
    ; x_next = x + 0.5*h*(k1x + k2x)
    vbroadcastsd ymm1, [C_half_h]
    vmovupd ymm0, [r15+r8]        ; base x
    vmovdqu ymm2, [rsp+0]         ; k1x
    vaddpd  ymm2, ymm2, ymm3      ; k1x + k2x
    vmulpd  ymm2, ymm2, ymm1
    vaddpd  ymm0, ymm0, ymm2
    vmovupd [rsi+r8], ymm0

    ; y_next = y + 0.5*h*(k1y + k2y)
    vmovupd ymm0, [rbx+r8]        ; base y
    vmovdqu ymm2, [rsp+32]        ; k1y
    vaddpd  ymm2, ymm2, ymm4      ; k1y + k2y
    vbroadcastsd ymm3, [C_half_h]
    vmulpd  ymm2, ymm2, ymm3
    vaddpd  ymm0, ymm0, ymm2
    vmovupd [rax+r8], ymm0

    ; z_next = z + 0.5*h*(k1z + k2z)
    vmovupd ymm0, [rdi+r8]        ; base z
    vmovdqu ymm2, [rsp+64]        ; k1z
    vaddpd  ymm2, ymm2, ymm5      ; k1z + k2z
    vbroadcastsd ymm3, [C_half_h]
    vmulpd  ymm2, ymm2, ymm3
    vaddpd  ymm0, ymm0, ymm2
    vmovupd [rdx+r8], ymm0

    ; advance i by 4
    add     ecx, 4
    jmp     .inner_vec

; ---- Scalar tail for remaining (n - i) elements ----
.inner_scalar:
    cmp     ecx, r9d
    jge     .end_inner

    ; offset = i * 8
    movsxd  r8, ecx
    shl     r8, 3

    ; Load base x,y,z into xmm0..xmm2
    movsd   xmm0, [r15+r8]          ; x
    movsd   xmm1, [rbx+r8]          ; y
    movsd   xmm2, [rdi+r8]          ; z

    ; ---- k1 ----
    ; k1x -> xmm3
    movapd  xmm3, xmm1
    subsd   xmm3, xmm0
    mulsd   xmm3, [rel C_sigma]

    ; k1y -> xmm4
    movapd  xmm4, xmm0
    mulsd   xmm4, [rel C_r]
    subsd   xmm4, xmm1
    movapd  xmm5, xmm0
    mulsd   xmm5, xmm2
    subsd   xmm4, xmm5

    ; k1z -> xmm5
    movapd  xmm5, xmm0
    mulsd   xmm5, xmm1
    movsd   xmm0, [rel C_b]
    mulsd   xmm0, xmm2
    subsd   xmm5, xmm0

    ; predictor: put xp,yp,zp back into xmm0..xmm2 (overwriting)
    movsd   xmm0, [rel C_h]
    mulsd   xmm0, xmm3
    addsd   xmm0, [r15+r8]          ; xp

    movsd   xmm1, [rel C_h]
    mulsd   xmm1, xmm4
    addsd   xmm1, [rbx+r8]          ; yp

    movsd   xmm2, [rel C_h]
    mulsd   xmm2, xmm5
    addsd   xmm2, [rdi+r8]          ; zp

    ; ---- k2 ----
    ; k2x -> xmm3
    movapd  xmm3, xmm1
    subsd   xmm3, xmm0
    mulsd   xmm3, [rel C_sigma]

    ; k2y -> xmm4
    movapd  xmm4, xmm0
    mulsd   xmm4, [rel C_r]
    subsd   xmm4, xmm1
    movapd  xmm5, xmm0
    mulsd   xmm5, xmm2
    subsd   xmm4, xmm5

    ; k2z -> xmm5
    movapd  xmm5, xmm0
    mulsd   xmm5, xmm1
    movsd   xmm0, [rel C_b]
    mulsd   xmm0, xmm2
    subsd   xmm5, xmm0

    ; ---- update ----
    movsd   xmm0, [rel C_half_h]

    ; x_next
    movsd   xmm2, [r15+r8]          ; base x
    ; k1x was overwritten; recompute quickly: sigma*(y - x)
    movsd   xmm1, [rbx+r8]
    subsd   xmm1, [r15+r8]
    mulsd   xmm1, [rel C_sigma]
    addsd   xmm1, xmm3              ; k1x + k2x
    mulsd   xmm1, xmm0
    addsd   xmm2, xmm1
    movsd   [rsi+r8], xmm2

    ; y_next
    movsd   xmm2, [rbx+r8]          ; base y
    ; k1y recompute: r*x - y - x*z
    movsd   xmm1, [r15+r8]
    mulsd   xmm1, [rel C_r]
    subsd   xmm1, [rbx+r8]
    movsd   xmm2, [r15+r8]
    mulsd   xmm2, [rdi+r8]
    subsd   xmm1, xmm2              ; k1y
    addsd   xmm1, xmm4              ; + k2y
    mulsd   xmm1, [rel C_half_h]
    movsd   xmm2, [rbx+r8]
    addsd   xmm2, xmm1
    movsd   [rax+r8], xmm2

    ; z_next
    movsd   xmm2, [rdi+r8]          ; base z
    ; k1z recompute: x*y - b*z
    movsd   xmm1, [r15+r8]
    mulsd   xmm1, [rbx+r8]          ; x*y
    movsd   xmm2, [rdi+r8]
    mulsd   xmm2, [rel C_b]         ; b*z
    subsd   xmm1, xmm2              ; k1z
    addsd   xmm1, xmm5              ; + k2z
    mulsd   xmm1, [rel C_half_h]
    movsd   xmm2, [rdi+r8]
    addsd   xmm2, xmm1
    movsd   [rdx+r8], xmm2

    inc     ecx
    jmp     .inner_scalar


.end_inner:
    inc     r14d
    jmp     .outer

.done:
    vzeroupper                 ; avoid AVX-SSE transition penalty
    add     rsp, 128
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    ret
