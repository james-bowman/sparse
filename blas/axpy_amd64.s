//+build !noasm,!appengine,!safe

#include "textflag.h"

// func Dusaxpy(alpha float64, x []float64, indx []int, y []float64, incy int)
TEXT ·Dusaxpy(SB), NOSPLIT, $0
    MOVSD   alpha+0(FP), X0
    MOVQ    x+8(FP), SI
    MOVQ    indx+32(FP), R8
    MOVQ    indx+40(FP), AX
    MOVQ    y+56(FP), CX
    MOVQ    incy+80(FP), DI

    MOVAPS  X0, X1              // alpha in X0 and X1 for pipelining
    XORL    R9, R9

    SUBQ    $4, AX              // len(indx)-4
 
    LEAQ    (SI)(AX*8), R14     // R14 = &indx[len(indx)-4]
    LEAQ    (R8)(AX*8), R15     // R15 = &indx[len(indx)-4]

    SUBQ    AX, R9
    JG      tailstart

loop:
    MOVSD   (R14)(R9*8), X2     // X2 := x[i]
    MOVSD   8(R14)(R9*8), X3    // X3 := x[i+1]
    MOVSD   16(R14)(R9*8), X4   // X4 := x[i+2]
    MOVSD   24(R14)(R9*8), X5   // X5 := x[i+3]

    MOVQ    (R15)(R9*8), R10    // R10 := indx[i]
    MOVQ    8(R15)(R9*8), R11   // R11 := indx[i+1]
    MOVQ    16(R15)(R9*8), R12  // R12 := indx[i+2]
    MOVQ    24(R15)(R9*8), R13  // R13 := indx[i+3]

    IMULQ   DI, R10             // R10 *= incy
    IMULQ   DI, R11             // R11 *= incy
    IMULQ   DI, R12             // R12 *= incy
    IMULQ   DI, R13             // R13 *= incy

    MULSD   X0, X2
    MULSD   X1, X3
    MULSD   X0, X4
    MULSD   X1, X5

    ADDSD   (CX)(R10*8), X2
    ADDSD   (CX)(R11*8), X3
    ADDSD   (CX)(R12*8), X4
    ADDSD   (CX)(R13*8), X5

    MOVSD   X2, (CX)(R10*8)
    MOVSD   X3, (CX)(R11*8)
    MOVSD   X4, (CX)(R12*8)
    MOVSD   X5, (CX)(R13*8)

    ADDQ    $4, R9              // i += 4
    JLE     loop

tailstart:
    SUBQ    $4, R9
    JNS     end

tail:
    // one more y[indx[i]*incy] += alpha * x[i]
    MOVSD   32(R14)(R9*8), X2   // X1 := x[i : i+1]
    MOVQ    32(R15)(R9*8), R10  // R10 := indx[i]
    IMULQ   DI, R10             // R10 *= incy
    MULSD   X0, X2
    ADDSD   (CX)(R10*8), X2
    MOVSD   X2, (CX)(R10*8)
    
    ADDQ    $1, R9  
    JS      tail

end:
    RET
