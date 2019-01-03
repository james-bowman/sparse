//+build !noasm,!appengine,!safe

#include "textflag.h"

// func Dusdot(x []float64, indx []int, y []float64, incy int) (dot float64)
TEXT ·Dusdot(SB), NOSPLIT, $0
    MOVQ    x+0(FP), R8
    MOVQ    indx+24(FP), SI    
    MOVQ    y+48(FP), DX       
    MOVQ    indx+32(FP), AX             // len(indx)
    MOVQ    incy+72(FP), CX 
    
    XORPS   X0, X0                      // 2 accumulators for reduced dependencies and
    XORPS   X9, X9                      // better software pipelining

    SUBQ    $4, AX
    JL      tailstart

loop:
    MOVQ    (SI), R10           
    MOVQ    8(SI), R11           
    MOVQ    16(SI), R12           
    MOVQ    24(SI), R13           

    MOVUPD  (R8), X1                    // load packed pairs of elements from x into SSE registers
    MOVUPD  16(R8), X3       

    IMULQ   CX, R10                     // multiply indx (indices of x) by incy (stride for y) to
    IMULQ   CX, R11                     // calculate corresponding indexes for y
    IMULQ   CX, R12      
    IMULQ   CX, R13      

    MOVLPD  (DX)(R10*8), X2             // pack scattered elements of y into SSE registers (gather)
    MOVHPD  (DX)(R11*8), X2
    MOVLPD  (DX)(R12*8), X4
    MOVHPD  (DX)(R13*8), X4

    MULPD   X2, X1                      
    MULPD   X4, X3

    ADDPD   X1, X0
    ADDPD   X3, X9

    ADDQ    $32, SI                     // increment pointers by 4*sizeOf(float64)
    ADDQ    $32, R8

    SUBQ    $4, AX
    JGE     loop

tailstart:
    ADDQ    $4, AX
    JE      end

tail:
    // process remaining elements individually where length is not divisible by 4
    MOVQ    (SI), R10           
    MOVSD   (R8), X1      
    IMULQ   CX, R10            

    MULSD   (DX)(R10*8), X1
    ADDSD   X1, X0

    ADDQ    $8, SI
    ADDQ    $8, R8

    SUBQ    $1, AX
    JNE     tail

end:
    ADDPD   X9, X0                      // add accumulators together
    MOVSD   X0, X7                      // unpack SSE register and add 2 values together
    UNPCKHPD X0, X0
    ADDSD   X7, X0

    MOVSD   X0, dot+80(FP)      
    RET
