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

    SUBQ    $4, AX              // len(indx)-4

    SHUFPD  $0, X0, X0
    MOVUPD  X0, X1              // copy alpha for better pipelining

    XORL    R9, R9
 
    LEAQ    (SI)(AX*8), R15     // R14 = &indx[len(indx)-4]
    LEAQ    (R8)(AX*8), R14     // R15 = &indx[len(indx)-4]

    SUBQ    AX, R9
    JG      tailstart

loop:
    MOVUPD  (R15)(R9*8), X2     // X2 := x[i : i+1]
    MOVUPD  16(R15)(R9*8), X3   // X3 := x[i+2 : i+3]

    MOVQ    (R14)(R9*8), R10    // R10 := indx[i]
    MOVQ    8(R14)(R9*8), R11   // R11 := indx[i+1]
    MOVQ    16(R14)(R9*8), R12  // R12 := indx[i+2]
    MOVQ    24(R14)(R9*8), R13  // R13 := indx[i+3]

    IMULQ   DI, R10             // R10 *= incy
    IMULQ   DI, R11             // R11 *= incy
    IMULQ   DI, R12             // R12 *= incy
    IMULQ   DI, R13             // R13 *= incy

    MOVLPD  (CX)(R10*8), X4     // X4l = y[R10]
    MOVHPD  (CX)(R11*8), X4     // X4h = y[R11]
    MOVLPD  (CX)(R12*8), X5     // X5l = y[R12]
    MOVHPD  (CX)(R13*8), X5     // X5h = y[R13]

    MULPD   X0, X2              // X2 := alpha * x[i : i+1]
    MULPD   X1, X3              // X3 := alpha * x[i+2 : i+3]

    ADDPD   X4, X2
    ADDPD   X5, X3
    
    MOVLPD  X2, (CX)(R10*8)
    MOVHPD  X2, (CX)(R11*8)
    MOVLPD  X3, (CX)(R12*8)
    MOVHPD  X3, (CX)(R13*8)

    ADDQ    $4, R9              // i += 4
    JLE     loop

tailstart:
    SUBQ    $4, R9
    JNS     end

tail:
    // y[indx[i]*incy] += alpha * x[i] for remaining elements of x
    MOVSD   32(R15)(R9*8), X2   // X1 := x[i : i+1]

    MOVQ    32(R14)(R9*8), R10  // R10 := indx[i]
    IMULQ   DI, R10             // R10 *= incy

    MULSD   X0, X2
    ADDSD   (CX)(R10*8), X2
    MOVSD   X2, (CX)(R10*8)

    ADDQ    $1, R9  
    JS      tail

end:
    RET
