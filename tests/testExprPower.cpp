#include "testUtils.hpp"

// =============================================================================
// SquareExpr — f^2 via Cauchy self-convolution
// =============================================================================

TEST(Square, ConstantSquare) {
    DAd<3> a{3.0};
    DAd<3> r = square(a);
    EXPECT_NEAR(r.value(), 9.0, kTol);
    for (std::size_t k = 1; k < DAd<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Square, LinearSquare) {
    // (1+x)^2 = 1 + 2x + x^2
    auto x = DAd<4>::variable<0>({1.0});
    DAd<4> r = square(x);
    EXPECT_NEAR(r[0], 1.0, kTol);
    EXPECT_NEAR(r[1], 2.0, kTol);
    EXPECT_NEAR(r[2], 1.0, kTol);
    for (std::size_t k = 3; k < DAd<4>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Square, MatchesSelfMultiply) {
    auto x = DAd<5>::variable<0>({2.0});
    DAd<5> r1 = square(x);
    DAd<5> r2 = x * x;
    ExpectCoeffsNear(r1, r2);
}

TEST(Square, Bivariate) {
    auto [x, y] = DAMd<3,2>::variables({1.0, 2.0});
    DAMd<3,2> r1 = square(x + y);
    DAMd<3,2> r2 = (x + y) * (x + y);
    ExpectCoeffsNear(r1, r2);
}

TEST(Square, OfExpression) {
    // square(a + b) should work when operand is a non-leaf
    auto x = DAd<4>::variable<0>({0.0});
    DAd<4> one{1.0};
    DAd<4> r = square(x + one);   // (1+x)^2
    EXPECT_NEAR(r[0], 1.0, kTol);
    EXPECT_NEAR(r[1], 2.0, kTol);
    EXPECT_NEAR(r[2], 1.0, kTol);
}

// =============================================================================
// CubeExpr — f^3 via direct triple convolution
// =============================================================================

TEST(Cube, ConstantCube) {
    DAd<3> a{2.0};
    DAd<3> r = cube(a);
    EXPECT_NEAR(r.value(), 8.0, kTol);
    for (std::size_t k = 1; k < DAd<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Cube, LinearCube) {
    // (1+x)^3 = 1 + 3x + 3x^2 + x^3
    auto x = DAd<4>::variable<0>({1.0});
    DAd<4> r = cube(x);
    EXPECT_NEAR(r[0], 1.0, kTol);
    EXPECT_NEAR(r[1], 3.0, kTol);
    EXPECT_NEAR(r[2], 3.0, kTol);
    EXPECT_NEAR(r[3], 1.0, kTol);
    EXPECT_NEAR(r[4], 0.0, kTol);
}

TEST(Cube, MatchesTripleMultiply) {
    auto x = DAd<5>::variable<0>({2.0});
    DAd<5> r1 = cube(x);
    DAd<5> r2 = x * x * x;
    ExpectCoeffsNear(r1, r2);
}

TEST(Cube, Bivariate) {
    auto [x, y] = DAMd<3,2>::variables({1.0, 1.0});
    DAMd<3,2> r1 = cube(x);
    DAMd<3,2> r2 = x * x * x;
    ExpectCoeffsNear(r1, r2);
}

TEST(Cube, OfExpression) {
    auto x = DAd<4>::variable<0>({0.0});
    DAd<4> one{1.0};
    DAd<4> r = cube(x + one);  // (1+x)^3
    EXPECT_NEAR(r[0], 1.0, kTol);
    EXPECT_NEAR(r[1], 3.0, kTol);
    EXPECT_NEAR(r[2], 3.0, kTol);
    EXPECT_NEAR(r[3], 1.0, kTol);
}

// =============================================================================
// SqrtExpr — Taylor series of sqrt(f)
// =============================================================================

TEST(Sqrt, ConstantSqrt) {
    DAd<3> a{9.0};
    DAd<3> r = sqrt(a);
    EXPECT_NEAR(r.value(), 3.0, kTol);
    for (std::size_t k = 1; k < DAd<3>::ncoef; ++k)
        EXPECT_NEAR(r[k], 0.0, kTol);
}

TEST(Sqrt, Sqrt1PlusX) {
    // sqrt(1+x) Taylor coefficients:
    //   c[0]=1, c[1]=1/2, c[2]=-1/8, c[3]=1/16, c[4]=-5/128
    auto x = DAd<4>::variable<0>({1.0});
    DAd<4> r = sqrt(x);
    EXPECT_NEAR(r[0],  1.0,        kTol);
    EXPECT_NEAR(r[1],  0.5,        kTol);
    EXPECT_NEAR(r[2], -0.125,      kTol);
    EXPECT_NEAR(r[3],  0.0625,     kTol);
    EXPECT_NEAR(r[4], -5.0/128.0,  kTol);
}

TEST(Sqrt, SqrtSquaredIsIdentity) {
    // sqrt(x)^2 should recover x
    auto x = DAd<5>::variable<0>({4.0});
    DAd<5> r = square(sqrt(x));
    ExpectCoeffsNear(r, x);
}

TEST(Sqrt, SquareThenSqrt) {
    // sqrt(x^2) when x>0 should give |x| = x
    auto x = DAd<4>::variable<0>({3.0});
    DAd<4> r = sqrt(square(x));
    ExpectCoeffsNear(r, x);
}

TEST(Sqrt, DerivativeCheck) {
    // d/dx sqrt(x) at x0=4: value = 2, deriv = 1/(2*sqrt(4)) = 0.25
    auto x = DAd<3>::variable<0>({4.0});
    DAd<3> r = sqrt(x);
    EXPECT_NEAR(r.value(), 2.0, kTol);
    EXPECT_NEAR(r.derivative({1}), 0.25, kTol);
}

TEST(Sqrt, Bivariate) {
    auto [x, y] = DAMd<3,2>::variables({4.0, 1.0});
    DAMd<3,2> r1 = sqrt(x);
    // sqrt(x) should not depend on y
    EXPECT_NEAR(r1.coeff({0,1}), 0.0, kTol);
    EXPECT_NEAR(r1.coeff({0,0}), 2.0, kTol);
}

TEST(Sqrt, OfExpression) {
    // sqrt(a+b) where a+b is a non-leaf expression
    DAd<4> a{3.0}, b{1.0};
    DAd<4> r = sqrt(a + b);   // sqrt(4) = 2
    EXPECT_NEAR(r.value(), 2.0, kTol);
}
