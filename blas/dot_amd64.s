//+build !noasm,!appengine,!safe

#include "textflag.h"

// func Dusdot(x []float64, indx []int, y []float64, incy int) (dot float64)
TEXT ·Dusdot(SB), NOSPLIT, $0
    MOVQ    x+0(FP), R8
    MOVQ    indx+24(FP), SI
    MOVQ    indx+32(FP), AX     // AX = len(indx)
    MOVQ    y+48(FP), DX
    MOVQ    incy+72(FP), CX

    XORL    R9, R9              // i = 0
    XORPS   X0, X0              // sum = 0.0

    SUBQ    $1, AX              // len(indx)--

loop:
    CMPQ    R9, AX              // for ;i < len(indx);
    JGE     tail

    MOVUPD  (R8)(R9*8), X1      // X1 := x[i : i+1]
    
    MOVQ    (SI)(R9*8), R10     // R10 := indx[i]
    MOVQ    8(SI)(R9*8), R11    // R11 := indx[i+1]

    IMULQ   CX, R10             // R10 *= incy
    IMULQ   CX, R11             // R11 *= incy

    MOVLPD  (DX)(R10*8), X2     // X2l = y[R10]
    MOVHPD  (DX)(R11*8), X2     // X2h = y[R11]

    MULPD   X2, X1              
    ADDPD   X1, X0              // X0 += X1 * X2

    ADDQ    $2, R9              // i += 2

    JMP     loop

tail:
    ADDQ    $1, AX
    CMPQ    R9, AX
    JGE     end

    // Sum product of last elements if odd number of elements
    MOVSD   (R8)(R9*8), X1      // X1 := x[i]

    MOVQ    (SI)(R9*8), R10
    IMULQ   CX, R10
    MULSD   (DX)(R10*8), X1
    ADDSD   X1, X0

end:
    // Add the two sums together.
    MOVSD   X0, X3
    UNPCKHPD X0, X0
    ADDSD   X3, X0

    MOVSD   X0, dot+80(FP)      // Return final sum.
    RET
 
