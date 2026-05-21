// include/tax/ode/detail/verner_tableaus.hpp
//
// Butcher tableaux for J.H. Verner's "efficient" RK pairs:
//   Verner78Tab — 13-stage, propagates at order 8 with order-7 embedded
//                 error estimator. Ported from pre-Stage-1 branch
//                 (claude/add-verner-integrators-vEgRF) which itself
//                 reproduced the SciML/OrdinaryDiffEq.jl `Vern8` tableau,
//                 derived from Verner's published rationals
//                 (https://www.sfu.ca/~jverner/).
//   Verner89Tab — 16-stage, propagates at order 9 with order-8 embedded.
//                 (Added in a follow-up task.)
//
// Layout matches the tax::ode::detail::adaptive_rk_step contract:
//   c     : nodes  (size n_stages)
//   a     : lower-triangular row-major  (size n_stages*(n_stages-1)/2)
//   b     : main weights (size n_stages)
//   b_emb : embedded weights (size n_stages)

#pragma once

#include <array>

namespace tax::ode::detail
{

struct Verner78Tab
{
    static constexpr int n_stages  = 13;
    static constexpr int order     = 8;
    static constexpr int order_emb = 7;
    static constexpr bool fsal     = false;

    // c[0] = 0.0 (first stage at t), then c2..c13 from the pre-Stage-1 source.
    static constexpr std::array< double, 13 > c{
        0.0,
        0.05,
        0.1065625,
        0.15984375,
        0.39,
        0.465,
        0.155,
        0.943,
        0.901802041735857,
        0.909,
        0.94,
        1.0,
        1.0
    };

    // 78 values, row-major lower-triangular (without diagonal):
    //   index 0          = a21             (stage 2, depends on stage 1 only)
    //   indices 1..2     = a31, a32        (stage 3)
    //   indices 3..5     = a41, a42, a43   (stage 4)
    //   ...
    //   indices 66..77   = a13_1 .. a13_12 (stage 13)
    // Zero entries are explicit (pre-Stage-1 source omits stages 2,3 columns
    // for stages 5..13).
    static constexpr std::array< double, 78 > a{
        0.05,  // stage 2
        -0.0069931640625, 0.1135556640625,  // stage 3
        0.0399609375, 0.0, 0.1198828125,  // stage 4
        0.36139756280045754, 0.0, -1.3415240667004928, 1.3701265039000352,  // stage 5
        0.049047202797202795, 0.0, 0.0, 0.23509720422144048, 0.18085559298135673,  // stage 6
        0.06169289044289044, 0.0, 0.0, 0.11236568314640277, -0.03885046071451367, 0.01979188712522046,  // stage 7
        -1.767630240222327, 0.0, 0.0, -62.5, -6.061889377376669, 5.6508231982227635, 65.62169641937624,  // stage 8
        -1.1809450665549708, 0.0, 0.0, -41.50473441114321, -4.434438319103725, 4.260408188586133, 43.75364022446172, 0.00787142548991231,  // stage 9
        -1.2814059994414884, 0.0, 0.0, -45.047139960139866, -4.731362069449576, 4.514967016593808, 47.44909557172985, 0.01059228297111661, -0.0057468422638446166,  // stage 10
        -1.7244701342624853, 0.0, 0.0, -60.92349008483054, -5.951518376222392, 5.556523730698456, 63.98301198033305, 0.014642028250414961, 0.06460408772358203, -0.0793032316900888,  // stage 11
        -3.301622667747079, 0.0, 0.0, -118.01127235975251, -10.141422388456112, 9.139311332232058, 123.37594282840426, 4.62324437887458, -3.3832777380682018, 4.527592100324618, -5.828495485811623,  // stage 12
        -3.039515033766309, 0.0, 0.0, -109.26086808941763, -9.290642497400293, 8.43050498176491, 114.20100103783314, -0.9637271342145479, -5.0348840888021895, 5.958130824002923, 0.0, 0.0  // stage 13
    };

    // Order-8 propagation weights (stages 2-5 and 13 do not contribute).
    static constexpr std::array< double, 13 > b{
        0.04427989419007951,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3541049391724449,
        0.24796921549564377,
        -15.694202038838085,
        25.084064965558564,
        -31.738367786260277,
        22.938283273988784,
        -0.2361324633071542,
        0.0
    };

    // Embedded order-7 weights: bhat_i = b_i - e_i
    // (pre-Stage-1 stored e_i = b_i - bhat_i; here e_i has been subtracted).
    static constexpr std::array< double, 13 > b_emb{
        0.04431261522908979,
        0.0,
        0.0,
        0.0,
        0.0,
        0.35460956423432266,
        0.24784804313666528,
        4.448134732475783,
        19.846886366118735,
        -23.58162337746562,
        0.0,
        0.0,
        -0.36016794372897754
    };
};

}  // namespace tax::ode::detail
