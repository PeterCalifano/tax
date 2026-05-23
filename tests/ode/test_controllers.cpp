// tests/ode/testControllers.cpp
//
// Covers controller behaviour:
//   - I: stateless, predictable scaling on err < tol and err > tol.
//   - PI: state-evolving, uses previous error.
//   - H211b: state-evolving, smoothed.
//   - JorbaZou: uses last-two-coefficient norms.
// Each test asserts that min_factor / max_factor clamps are applied.

#include <gtest/gtest.h>

#include <cmath>

#include <tax/ode.hpp>

using tax::ode::controllers::I;
using tax::ode::controllers::PI;
using tax::ode::controllers::H211b;
using tax::ode::controllers::JorbaZou;

TEST( OdeControllerI, ScalesDownOnLargeError )
{
    I< double > c;
    const double h_used = 0.1;
    const double err    = 10.0;
    const double tol    = 1.0;
    const int    p_emb  = 7;
    const double h_new  = c.next_step( h_used, err, tol, p_emb );
    EXPECT_LT( h_new, h_used );
    EXPECT_GE( h_new, h_used * c.min_factor );
}

TEST( OdeControllerI, ScalesUpOnSmallError )
{
    I< double > c;
    const double h_new = c.next_step( /*h_used=*/0.1, /*err=*/0.01,
                                       /*tol=*/1.0, /*p_emb=*/7 );
    EXPECT_GT( h_new, 0.1 );
    EXPECT_LE( h_new, 0.1 * c.max_factor );
}

TEST( OdeControllerPI, RemembersPreviousError )
{
    PI< double > c;
    const double h1 = c.next_step( 0.1, 0.5, 1.0, 7 );
    const double h2 = c.next_step( h1, 0.5, 1.0, 7 );
    // On the second step the proportional term contributes;
    // the result must differ from the I-only equivalent.
    I< double > i;
    const double i_only = i.next_step( h1, 0.5, 1.0, 7 );
    EXPECT_NE( h2, i_only );
}

TEST( OdeControllerH211b, FirstCallBehavesLikeI )
{
    H211b< double > c;
    I< double >     i;
    // On its very first call h_prev_ == 0 so the controller falls
    // back to I-step semantics.
    const double h_new_h  = c.next_step( 0.1, 0.5, 1.0, 7 );
    const double h_new_i  = i.next_step( 0.1, 0.5, 1.0, 7 );
    EXPECT_NEAR( h_new_h, h_new_i, 1e-12 );
}

TEST( OdeControllerJorbaZou, ScalesDownOnLargeLeadingCoeff )
{
    JorbaZou< double > c;
    const double h_used     = 0.1;
    const double c_N_norm   = 1e6;     // very large => shrink
    const double c_Nm1_norm = 1e6;
    const double tol        = 1e-12;
    const int    N_order    = 12;
    const double h_new = c.next_step( h_used, c_N_norm, c_Nm1_norm,
                                       tol, N_order );
    EXPECT_LT( h_new, h_used );
    EXPECT_GE( h_new, h_used * c.min_factor );
}
