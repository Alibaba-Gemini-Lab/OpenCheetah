/*
Authors: Nishant Kumar, Deevashwer Rathee
Modified by Wen-jie Lu
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GLOBALS_H___
#define GLOBALS_H___

#include "NonLinear/argmax.h"
#include "NonLinear/maxpool.h"
#include "NonLinear/relu-interface.h"
#include "defines.h"
#include "defines_uniform.h"
#include <chrono>
#include <cstdint>
#include <thread>
#include "OT/kkot.h"
#ifdef SCI_OT
#include "BuildingBlocks/aux-protocols.h"
#include "BuildingBlocks/truncation.h"
#include "LinearOT/linear-ot.h"
#include "LinearOT/linear-uniform.h"
#include "Math/math-functions.h"
#endif
// Additional Headers for Athos
#ifdef SCI_HE
#include "LinearHE/elemwise-prod-field.h"
#include "LinearHE/fc-field.h"
#include "LinearHE/conv-field.h"
#endif

#if USE_CHEETAH
#include "cheetah/cheetah-api.h"
#endif

// #define MULTI_THREADING

#define MAX_THREADS 4

extern sci::NetIO *io;
extern sci::OTPack<sci::NetIO> *otpack;

#ifdef SCI_OT
extern LinearOT *mult;
extern AuxProtocols *aux;
extern Truncation *truncation;
extern XTProtocol *xt;
extern MathFunctions *math;
#endif
extern ArgMaxProtocol<sci::NetIO, intType> *argmax;
extern ReLUProtocol<sci::NetIO, intType> *relu;
extern MaxPoolProtocol<sci::NetIO, intType> *maxpool;
// Additional classes for Athos

#ifdef SCI_OT
extern MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>> *multUniform;
#elif defined(SCI_HE)
extern FCField *he_fc;
extern ElemWiseProdField *he_prod;
#endif

#if USE_CHEETAH
extern gemini::CheetahLinear *cheetah_linear;
extern bool kIsSharedInput;
#elif defined(SCI_HE)
extern ConvField *he_conv;
#endif

extern sci::IKNP<sci::NetIO> *iknpOT;
extern sci::IKNP<sci::NetIO> *iknpOTRoleReversed;
extern sci::KKOT<sci::NetIO> *kkot;
extern sci::PRG128 *prg128Instance;

extern sci::NetIO *ioArr[MAX_THREADS];
extern sci::OTPack<sci::NetIO> *otpackArr[MAX_THREADS];
#ifdef SCI_OT
extern LinearOT *multArr[MAX_THREADS];
extern AuxProtocols *auxArr[MAX_THREADS];
extern Truncation *truncationArr[MAX_THREADS];
extern XTProtocol *xtArr[MAX_THREADS];
extern MathFunctions *mathArr[MAX_THREADS];
#endif
extern ReLUProtocol<sci::NetIO, intType> *reluArr[MAX_THREADS];
extern MaxPoolProtocol<sci::NetIO, intType> *maxpoolArr[MAX_THREADS];
// Additional classes for Athos
#ifdef SCI_OT
extern MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>>
    *multUniformArr[MAX_THREADS];
#endif
extern sci::IKNP<sci::NetIO> *otInstanceArr[MAX_THREADS];
extern sci::KKOT<sci::NetIO> *kkotInstanceArr[MAX_THREADS];
extern sci::PRG128 *prgInstanceArr[MAX_THREADS];

extern std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
extern uint64_t comm_threads[MAX_THREADS];
extern uint64_t num_rounds;

#ifdef LOG_LAYERWISE
extern uint64_t ConvTimeInMilliSec;
extern uint64_t MatAddTimeInMilliSec;
extern uint64_t BatchNormInMilliSec;
extern uint64_t TruncationTimeInMilliSec;
extern uint64_t ReluTimeInMilliSec;
extern uint64_t MaxpoolTimeInMilliSec;
extern uint64_t AvgpoolTimeInMilliSec;
extern uint64_t MatMulTimeInMilliSec;
extern uint64_t MatAddBroadCastTimeInMilliSec;
extern uint64_t MulCirTimeInMilliSec;
extern uint64_t ScalarMulTimeInMilliSec;
extern uint64_t SigmoidTimeInMilliSec;
extern uint64_t TanhTimeInMilliSec;
extern uint64_t SqrtTimeInMilliSec;
extern uint64_t NormaliseL2TimeInMilliSec;
extern uint64_t ArgMaxTimeInMilliSec;

extern uint64_t ConvCommSent;
extern uint64_t MatAddCommSent;
extern uint64_t BatchNormCommSent;
extern uint64_t TruncationCommSent;
extern uint64_t ReluCommSent;
extern uint64_t MaxpoolCommSent;
extern uint64_t AvgpoolCommSent;
extern uint64_t MatMulCommSent;
extern uint64_t MatAddBroadCastCommSent;
extern uint64_t MulCirCommSent;
extern uint64_t ScalarMulCommSent;
extern uint64_t SigmoidCommSent;
extern uint64_t TanhCommSent;
extern uint64_t SqrtCommSent;
extern uint64_t NormaliseL2CommSent;
extern uint64_t ArgMaxCommSent;
#endif

#endif // GLOBALS_H__
