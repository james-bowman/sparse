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
    XORPS   X1, X1              // sum2 = 0.0

    SUBQ    $3, AX              // len(indx)--

loop:
    CMPQ    R9, AX              // for ;i < len(indx);
    JGE     tailstart

    MOVUPD  (R8)(R9*8), X2      // X1 := x[i : i+1]
    MOVUPD  16(R8)(R9*8), X3    // X1 := x[i+2 : i+3]
    
    MOVQ    (SI)(R9*8), R10     // R10 := indx[i]
    MOVQ    8(SI)(R9*8), R11    // R11 := indx[i+1]
    MOVQ    16(SI)(R9*8), R12   // R12 := indx[i+2]
    MOVQ    24(SI)(R9*8), R13   // R13 := indx[i+3]

    IMULQ   CX, R10             // R10 *= incy
    IMULQ   CX, R11             // R11 *= incy
    IMULQ   CX, R12             // R12 *= incy
    IMULQ   CX, R13             // R13 *= incy

    MOVLPD  (DX)(R10*8), X4     // X4l = y[R10]
    MOVHPD  (DX)(R11*8), X4     // X4h = y[R11]
    MOVLPD  (DX)(R12*8), X5     // X5l = y[R10]
    MOVHPD  (DX)(R13*8), X5     // X5h = y[R11]

    MULPD   X4, X2
    MULPD   X5, X3    

    ADDPD   X2, X0              // X0 += X4 * X2
    ADDPD   X3, X1              // X1 += X5 * X3

    ADDQ    $4, R9              // i += 4

    JMP     loop

tailstart:
    ADDQ    $3, AX

tail:
    CMPQ    R9, AX
    JGE     end

    // Sum product of last elements if odd number of elements
    MOVSD   (R8)(R9*8), X2      // X1 := x[i]

    MOVQ    (SI)(R9*8), R10
    IMULQ   CX, R10
    MULSD   (DX)(R10*8), X2
    ADDSD   X2, X0

    INCQ    R9

    JMP     tail

end:
    // Add the two sums together.
    MOVSD   X0, X7
    UNPCKHPD X0, X0
    ADDSD   X7, X0

    MOVSD   X1, X7
    UNPCKHPD X1, X1
    ADDSD   X7, X0
    ADDSD   X1, X0

    MOVSD   X0, dot+80(FP)      // Return final sum.
    RET
 
