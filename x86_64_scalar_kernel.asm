section .data
msg db "Hello World", 0

section .text
bits 64 
default rel

global x86_64_scalar_kernel
extern printf
x86_64_scalar_kernel:
	sub rsp, 8*5
	lea rcx, [msg]
	call printf
	add rsp, 8*5
	ret