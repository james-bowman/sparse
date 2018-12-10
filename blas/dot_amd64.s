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

    SUBQ    $4, AX              // len(indx)-4
 
    LEAQ    (SI)(AX*8), R14     // R14 = &indx[len(indx)-4]
    LEAQ    (R8)(AX*8), R15     // R15 = &indx[len(indx)-4]

    SUBQ    AX, R9
    JG      tailstart

loop:
    MOVUPD  (R15)(R9*8), X2     // X1 := x[i : i+1]
    MOVUPD  16(R15)(R9*8), X3   // X1 := x[i+2 : i+3]
   
    MOVQ    (R14)(R9*8), R10    // R10 := indx[i]
    IMULQ   CX, R10             // R10 *= incy
    MOVLPD  (DX)(R10*8), X4     // X4l = y[R10]

    MOVQ    8(R14)(R9*8), R11   // R11 := indx[i+1]
    IMULQ   CX, R11             // R11 *= incy
    MOVHPD  (DX)(R11*8), X4     // X4h = y[R11]

    MOVQ    16(R14)(R9*8), R12  // R12 := indx[i+2]
    IMULQ   CX, R12             // R12 *= incy
    MOVLPD  (DX)(R12*8), X5     // X5l = y[R10]

    MOVQ    24(R14)(R9*8), R13  // R13 := indx[i+3]
    IMULQ   CX, R13             // R13 *= incy
    MOVHPD  (DX)(R13*8), X5     // X5h = y[R11]

    MULPD   X4, X2
    MULPD   X5, X3    

    ADDPD   X2, X0              // X0 += X4 * X2
    ADDPD   X3, X1              // X1 += X5 * X3

    ADDQ    $4, R9              // i += 4
    JLE     loop

tailstart:
    SUBQ    $4, R9
    JNS     end

tail:
    // Sum product of last elements if odd number of elements
    MOVSD   32(R15)(R9*8), X2   // X1 := x[i]

    MOVQ    32(R14)(R9*8), R10
    IMULQ   CX, R10
    MULSD   (DX)(R10*8), X2
    ADDSD   X2, X0

    ADDQ    $1, R9  
    JS      tail

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
 