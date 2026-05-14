#pragma once

/**
 * @file
 * @brief Butcher tableaux for J.H. Verner's "efficient" adaptive RK pairs.
 *
 * Two pairs are provided:
 *
 *  - @ref Verner78Coeffs: Verner 8(7), the "efficient" 13-stage pair that
 *    propagates at order 8 and uses an embedded order-7 estimator
 *    (commonly called "Verner 7/8").
 *  - @ref Verner89Coeffs: Verner 9(8), the "efficient" 16-stage pair that
 *    propagates at order 9 and uses an embedded order-8 estimator
 *    (commonly called "Verner 8/9").
 *
 * Coefficient values reproduce the compiled-floats tables in
 * SciML/OrdinaryDiffEq.jl (`Vern8` and `Vern9` tableaux), themselves derived
 * from J.H. Verner's published rational coefficients
 * (https://www.sfu.ca/~jverner/).  The `e*` entries below are
 * `btilde = b - bhat`, i.e. the embedded error weights — multiplying them
 * against the stages and `h` yields a state-valued error estimate directly.
 */

namespace tax::ode::detail
{

// =============================================================================
// Verner 8(7) — "efficient" 13-stage pair (order 8 propagation, 7 error est.)
// =============================================================================

struct Verner78Coeffs
{
    static constexpr int num_stages = 13;
    static constexpr int order      = 8;  ///< Order of the propagated solution.
    static constexpr int err_order  = 7;  ///< Order of the embedded estimator.

    // c_i: stage time-shifts (c1 = 0 implicit).
    static constexpr double c2  = 0.05;
    static constexpr double c3  = 0.1065625;
    static constexpr double c4  = 0.15984375;
    static constexpr double c5  = 0.39;
    static constexpr double c6  = 0.465;
    static constexpr double c7  = 0.155;
    static constexpr double c8  = 0.943;
    static constexpr double c9  = 0.901802041735857;
    static constexpr double c10 = 0.909;
    static constexpr double c11 = 0.94;
    static constexpr double c12 = 1.0;
    static constexpr double c13 = 1.0;

    // a_ij: stage couplings (omitted entries are zero).
    static constexpr double a0201 = 0.05;

    static constexpr double a0301 = -0.0069931640625;
    static constexpr double a0302 = 0.1135556640625;

    static constexpr double a0401 = 0.0399609375;
    static constexpr double a0403 = 0.1198828125;

    static constexpr double a0501 = 0.36139756280045754;
    static constexpr double a0503 = -1.3415240667004928;
    static constexpr double a0504 = 1.3701265039000352;

    static constexpr double a0601 = 0.049047202797202795;
    static constexpr double a0604 = 0.23509720422144048;
    static constexpr double a0605 = 0.18085559298135673;

    static constexpr double a0701 = 0.06169289044289044;
    static constexpr double a0704 = 0.11236568314640277;
    static constexpr double a0705 = -0.03885046071451367;
    static constexpr double a0706 = 0.01979188712522046;

    static constexpr double a0801 = -1.767630240222327;
    static constexpr double a0804 = -62.5;
    static constexpr double a0805 = -6.061889377376669;
    static constexpr double a0806 = 5.6508231982227635;
    static constexpr double a0807 = 65.62169641937624;

    static constexpr double a0901 = -1.1809450665549708;
    static constexpr double a0904 = -41.50473441114321;
    static constexpr double a0905 = -4.434438319103725;
    static constexpr double a0906 = 4.260408188586133;
    static constexpr double a0907 = 43.75364022446172;
    static constexpr double a0908 = 0.00787142548991231;

    static constexpr double a1001 = -1.2814059994414884;
    static constexpr double a1004 = -45.047139960139866;
    static constexpr double a1005 = -4.731362069449576;
    static constexpr double a1006 = 4.514967016593808;
    static constexpr double a1007 = 47.44909557172985;
    static constexpr double a1008 = 0.01059228297111661;
    static constexpr double a1009 = -0.0057468422638446166;

    static constexpr double a1101 = -1.7244701342624853;
    static constexpr double a1104 = -60.92349008483054;
    static constexpr double a1105 = -5.951518376222392;
    static constexpr double a1106 = 5.556523730698456;
    static constexpr double a1107 = 63.98301198033305;
    static constexpr double a1108 = 0.014642028250414961;
    static constexpr double a1109 = 0.06460408772358203;
    static constexpr double a1110 = -0.0793032316900888;

    static constexpr double a1201 = -3.301622667747079;
    static constexpr double a1204 = -118.01127235975251;
    static constexpr double a1205 = -10.141422388456112;
    static constexpr double a1206 = 9.139311332232058;
    static constexpr double a1207 = 123.37594282840426;
    static constexpr double a1208 = 4.62324437887458;
    static constexpr double a1209 = -3.3832777380682018;
    static constexpr double a1210 = 4.527592100324618;
    static constexpr double a1211 = -5.828495485811623;

    static constexpr double a1301 = -3.039515033766309;
    static constexpr double a1304 = -109.26086808941763;
    static constexpr double a1305 = -9.290642497400293;
    static constexpr double a1306 = 8.43050498176491;
    static constexpr double a1307 = 114.20100103783314;
    static constexpr double a1308 = -0.9637271342145479;
    static constexpr double a1309 = -5.0348840888021895;
    static constexpr double a1310 = 5.958130824002923;

    // b_i: order-8 propagation weights (stages 2-5 do not contribute).
    static constexpr double b1  = 0.04427989419007951;
    static constexpr double b6  = 0.3541049391724449;
    static constexpr double b7  = 0.24796921549564377;
    static constexpr double b8  = -15.694202038838085;
    static constexpr double b9  = 25.084064965558564;
    static constexpr double b10 = -31.738367786260277;
    static constexpr double b11 = 22.938283273988784;
    static constexpr double b12 = -0.2361324633071542;

    // e_i = b_i - bhat_i: embedded order-7 error weights.
    static constexpr double e1  = -3.272103901028138e-5;
    static constexpr double e6  = -0.0005046250618777704;
    static constexpr double e7  = 0.0001211723589784759;
    static constexpr double e8  = -20.142336771313868;
    static constexpr double e9  = 5.2371785994398286;
    static constexpr double e10 = -8.156744408794658;
    static constexpr double e11 = 22.938283273988784;
    static constexpr double e12 = -0.2361324633071542;
    static constexpr double e13 = 0.36016794372897754;
};

// =============================================================================
// Verner 9(8) — "efficient" 16-stage pair (order 9 propagation, 8 error est.)
// =============================================================================

struct Verner89Coeffs
{
    static constexpr int num_stages = 16;
    static constexpr int order      = 9;  ///< Order of the propagated solution.
    static constexpr int err_order  = 8;  ///< Order of the embedded estimator.

    // c_i: stage time-shifts (c1 = 0 implicit; final two stages at c = 1).
    static constexpr double c2  = 0.03462;
    static constexpr double c3  = 0.09702435063878045;
    static constexpr double c4  = 0.14553652595817068;
    static constexpr double c5  = 0.561;
    static constexpr double c6  = 0.22900791159048503;
    static constexpr double c7  = 0.544992088409515;
    static constexpr double c8  = 0.645;
    static constexpr double c9  = 0.48375;
    static constexpr double c10 = 0.06757;
    static constexpr double c11 = 0.25;
    static constexpr double c12 = 0.6590650618730999;
    static constexpr double c13 = 0.8206;
    static constexpr double c14 = 0.9012;
    static constexpr double c15 = 1.0;
    static constexpr double c16 = 1.0;

    // a_ij: stage couplings (omitted entries are zero).
    static constexpr double a0201 = 0.03462;

    static constexpr double a0301 = -0.03893354388572875;
    static constexpr double a0302 = 0.13595789452450918;

    static constexpr double a0401 = 0.03638413148954267;
    static constexpr double a0403 = 0.10915239446862801;

    static constexpr double a0501 = 2.0257639143939694;
    static constexpr double a0503 = -7.638023836496291;
    static constexpr double a0504 = 6.173259922102322;

    static constexpr double a0601 = 0.05112275589406061;
    static constexpr double a0604 = 0.17708237945550218;
    static constexpr double a0605 = 0.0008027762409222536;

    static constexpr double a0701 = 0.13160063579752163;
    static constexpr double a0704 = -0.2957276252669636;
    static constexpr double a0705 = 0.08781378035642955;
    static constexpr double a0706 = 0.6213052975225274;

    static constexpr double a0801 = 0.07166666666666667;
    static constexpr double a0806 = 0.33055335789153195;
    static constexpr double a0807 = 0.2427799754418014;

    static constexpr double a0901 = 0.071806640625;
    static constexpr double a0906 = 0.3294380283228177;
    static constexpr double a0907 = 0.1165190029271823;
    static constexpr double a0908 = -0.034013671875;

    static constexpr double a1001 = 0.04836757646340646;
    static constexpr double a1006 = 0.03928989925676164;
    static constexpr double a1007 = 0.10547409458903446;
    static constexpr double a1008 = -0.021438652846483126;
    static constexpr double a1009 = -0.10412291746271944;

    static constexpr double a1101 = -0.026645614872014785;
    static constexpr double a1106 = 0.03333333333333333;
    static constexpr double a1107 = -0.1631072244872467;
    static constexpr double a1108 = 0.03396081684127761;
    static constexpr double a1109 = 0.1572319413814626;
    static constexpr double a1110 = 0.21522674780318796;

    static constexpr double a1201 = 0.03689009248708622;
    static constexpr double a1206 = -0.1465181576725543;
    static constexpr double a1207 = 0.2242577768172024;
    static constexpr double a1208 = 0.02294405717066073;
    static constexpr double a1209 = -0.0035850052905728597;
    static constexpr double a1210 = 0.08669223316444385;
    static constexpr double a1211 = 0.43838406519683376;

    static constexpr double a1301 = -0.4866012215113341;
    static constexpr double a1306 = -6.304602650282853;
    static constexpr double a1307 = -0.2812456182894729;
    static constexpr double a1308 = -2.679019236219849;
    static constexpr double a1309 = 0.5188156639241577;
    static constexpr double a1310 = 1.3653531876033418;
    static constexpr double a1311 = 5.8850910885039465;
    static constexpr double a1312 = 2.8028087862720628;

    static constexpr double a1401 = 0.4185367457753472;
    static constexpr double a1406 = 6.724547581906459;
    static constexpr double a1407 = -0.42544428016461133;
    static constexpr double a1408 = 3.3432791530012653;
    static constexpr double a1409 = 0.6170816631175374;
    static constexpr double a1410 = -0.9299661239399329;
    static constexpr double a1411 = -6.099948804751011;
    static constexpr double a1412 = -3.002206187889399;
    static constexpr double a1413 = 0.2553202529443446;

    static constexpr double a1501 = -0.7793740861228848;
    static constexpr double a1506 = -13.937342538107776;
    static constexpr double a1507 = 1.2520488533793563;
    static constexpr double a1508 = -14.691500408016868;
    static constexpr double a1509 = -0.494705058533141;
    static constexpr double a1510 = 2.2429749091462368;
    static constexpr double a1511 = 13.367893803828643;
    static constexpr double a1512 = 14.396650486650687;
    static constexpr double a1513 = -0.79758133317768;
    static constexpr double a1514 = 0.4409353709534278;

    static constexpr double a1601 = 2.0580513374668867;
    static constexpr double a1606 = 22.357937727968032;
    static constexpr double a1607 = 0.9094981099755646;
    static constexpr double a1608 = 35.89110098240264;
    static constexpr double a1609 = -3.442515027624454;
    static constexpr double a1610 = -4.865481358036369;
    static constexpr double a1611 = -18.909803813543427;
    static constexpr double a1612 = -34.26354448030452;
    static constexpr double a1613 = 1.2647565216956427;

    // b_i: order-9 propagation weights (stages 2-7 and stage 16 do not contribute).
    static constexpr double b1  = 0.014611976858423152;
    static constexpr double b8  = -0.3915211862331339;
    static constexpr double b9  = 0.23109325002895065;
    static constexpr double b10 = 0.12747667699928525;
    static constexpr double b11 = 0.2246434176204158;
    static constexpr double b12 = 0.5684352689748513;
    static constexpr double b13 = 0.058258715572158275;
    static constexpr double b14 = 0.13643174034822156;
    static constexpr double b15 = 0.030570139830827976;

    // e_i = b_i - bhat_i: embedded order-8 error weights.
    static constexpr double e1  = -0.005357988290444578;
    static constexpr double e8  = -2.583020491182464;
    static constexpr double e9  = 0.14252253154686625;
    static constexpr double e10 = 0.013420653512688676;
    static constexpr double e11 = -0.02867296291409493;
    static constexpr double e12 = 2.624999655215792;
    static constexpr double e13 = -0.2825509643291537;
    static constexpr double e14 = 0.13643174034822156;
    static constexpr double e15 = 0.030570139830827976;
    static constexpr double e16 = -0.04834231373823958;
};

}  // namespace tax::ode::detail
