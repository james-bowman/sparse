//+build !noasm,!appengine,!safe

#include "textflag.h"

// func Dusaxpy(alpha float64, x []float64, indx []int, y []float64, incy int)
TEXT ·Dusaxpy(SB), NOSPLIT, $0
    MOVSD   alpha+0(FP), X0
    MOVQ    x+8(FP), SI
    MOVQ    x+16(FP), BX
    MOVQ    indx+32(FP), R8
    MOVQ    indx+40(FP), AX
    MOVQ    y+56(FP), CX
    MOVQ    y+64(FP), DX
    MOVQ    incy+80(FP), DI

    SHUFPD  $0, X0, X0

    XORL    R9, R9

    DECQ    AX                  // len(indx)--

loop:
    CMPQ    R9, AX
    JGE     tail

    MOVUPD  (SI)(R9*8), X2      // X1 := x[i : i+1]

    MOVQ    (R8)(R9*8), R10     // R10 := indx[i]
    IMULQ   DI, R10             // R10 *= incy
    INCQ    R9                  // i++
    MOVLPD  (CX)(R10*8), X1     // X2l = y[R10]

    MOVQ    (R8)(R9*8), R11     // R11 := indx[i]
    IMULQ   DI, R11             // R11 *= incy
    INCQ    R9                  // i++
    MOVHPD  (CX)(R11*8), X1     // X2h = y[R11]

    MULPD   X0, X2              // X2 := alpha * x[i : i+1]
    ADDPD   X2, X1              // X1 := y + X2

    MOVLPD  X1, (CX)(R10*8)
    MOVHPD  X1, (CX)(R11*8)

    JMP     loop

tail:
    INCQ    AX
    CMPQ    R9, AX
    JGE     end

    // one more y[indx[i]*incy] += alpha * x[i]
    MOVSD   (SI)(R9*8), X2      // X1 := x[i : i+1]

    MOVQ    (R8)(R9*8), R10     // R10 := indx[i]
    IMULQ   DI, R10             // R10 *= incy
    MOVSD   (CX)(R10*8), X1

    MULSD   X0, X2
    ADDSD   X2, X1
    MOVSD   X1, (CX)(R10*8)

end:
    RET
