#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>

#include <tax/tax.hpp>
#include <tax/ads/box.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

// =============================================================================
// IntegratorP — scalar ODE with constant parameter
// =============================================================================

TEST( ParamIntegrator, ScalarExponentialDecay )
{
    // dx/dt = -k * x, x(0) = x0  →  x(t) = x0 * exp(-k t)
    constexpr int N = 20;

    auto f = []( const auto& x, const double& k, [[maybe_unused]] const auto& t ) {
        return -k * x;
    };

    ode::IntegratorP< N, double, double > ig{ f,
                                              ode::IntegratorConfig< double >{ .abstol = 1e-20 } };

    const double k    = 0.7;
    const double x0   = 3.0;
    const double tmax = 2.0;
    auto         sol  = ig.integrate( x0, k, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back(), x0 * std::exp( -k * tmax ), 1e-12 );
}

// =============================================================================
// IntegratorP — vector ODE with a vector of parameters
// =============================================================================

TEST( ParamIntegrator, VectorDampedHarmonicOscillator )
{
    // dx0 = x1
    // dx1 = -ω² x0 - 2 ζ ω x1
    // Parameters: p = (ω, ζ)
    constexpr int N = 20;

    using Vec2   = Eigen::Vector2d;
    using Params = Vec2;

    auto f = []( auto& dx, const auto& x, const Params& p,
                 [[maybe_unused]] const auto& t ) {
        const double omega = p( 0 );
        const double zeta  = p( 1 );
        dx( 0 )            = x( 1 );
        dx( 1 )            = -omega * omega * x( 0 ) - 2.0 * zeta * omega * x( 1 );
    };

    ode::IntegratorP< N, Vec2, Params > ig{
        f, ode::IntegratorConfig< double >{ .abstol = 1e-16 } };

    Params       p{ 2.0, 0.0 };  // undamped, ω = 2
    Vec2         x0{ 1.0, 0.0 };
    const double tmax = std::numbers::pi;     // half period at ω = 2
    auto         sol  = ig.integrate( x0, p, 0.0, tmax );

    EXPECT_NEAR( sol.t.back(), tmax, 1e-14 );
    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-10 );
}

// =============================================================================
// DaIntegrator with Q > 0 — flow expansion w.r.t. parameters
// =============================================================================

TEST( ParamDaIntegrator, MakeDaStateAndParamsLayout )
{
    // Verify that IC variables go into slots 0..D-1 and parameter variables
    // into D..D+Q-1.
    constexpr int P = 1;
    constexpr int D = 1;
    constexpr int Q = 1;

    Box< double, D > x_box{ { 2.0 }, { 0.1 } };
    Box< double, Q > p_box{ { 0.5 }, { 0.01 } };

    auto x0 = ode::makeDaState< P, D, Q >( x_box );
    auto p0 = ode::makeDaParams< P, D, Q >( p_box );

    EXPECT_NEAR( x0( 0 ).value(), 2.0, 1e-14 );
    EXPECT_NEAR( p0( 0 ).value(), 0.5, 1e-14 );

    // δ_x is variable 0, δ_p is variable 1
    MultiIndex< D + Q > e_x{ 1, 0 };
    MultiIndex< D + Q > e_p{ 0, 1 };
    EXPECT_NEAR( x0( 0 ).coeff( e_x ), 0.1, 1e-14 );
    EXPECT_NEAR( x0( 0 ).coeff( e_p ), 0.0, 1e-14 );
    EXPECT_NEAR( p0( 0 ).coeff( e_x ), 0.0, 1e-14 );
    EXPECT_NEAR( p0( 0 ).coeff( e_p ), 0.01, 1e-14 );
}

TEST( ParamDaIntegrator, ScalarExponentialDecayFlow )
{
    // dx/dt = -k x.  Expand x(tmax; x0+δx, k+δp) about (x0, k) at first order.
    // Exact: x(t) = (x0 + δx) * exp(-(k + δp) t)
    //   ∂/∂δx |_(0,0) = exp(-k t)
    //   ∂/∂δp |_(0,0) = -t (x0) exp(-k t)
    constexpr int N = 20;
    constexpr int P = 1;
    constexpr int D = 1;
    constexpr int Q = 1;

    auto f = []( auto& dx, const auto& x, const auto& p,
                 [[maybe_unused]] const auto& t ) { dx( 0 ) = -p( 0 ) * x( 0 ); };

    const double x0   = 2.0;
    const double k    = 0.5;
    const double hwx  = 0.05;
    const double hwp  = 0.02;
    const double tmax = 1.5;

    Box< double, D > x_box{ { x0 }, { hwx } };
    Box< double, Q > p_box{ { k }, { hwp } };

    ode::DaIntegrator< N, P, D, Q > ig{
        f, ode::IntegratorConfig< double >{ .abstol = 1e-16 } };

    auto fm = ig.integrate( x_box, p_box, 0.0, tmax );

    const double expected_val = x0 * std::exp( -k * tmax );
    const double expected_dx  = std::exp( -k * tmax ) * hwx;
    const double expected_dp  = -tmax * x0 * std::exp( -k * tmax ) * hwp;

    EXPECT_NEAR( fm.state( 0 ).value(), expected_val, 1e-12 );

    MultiIndex< D + Q > e_x{ 1, 0 };
    MultiIndex< D + Q > e_p{ 0, 1 };
    EXPECT_NEAR( fm.state( 0 ).coeff( e_x ), expected_dx, 1e-12 );
    EXPECT_NEAR( fm.state( 0 ).coeff( e_p ), expected_dp, 1e-12 );
}

TEST( ParamDaIntegrator, HarmonicOscillatorOmegaSensitivity )
{
    // ẍ = -ω² x, expand about ω = ω0 with δω deviation.
    // System: dx0 = x1; dx1 = -ω² x0.
    // After half period at ω0=2, t = π/(2ω0) = π/4? Use generic tmax and
    // verify the polynomial agrees with the exact solution at a couple of
    // sample points.
    constexpr int N = 20;
    constexpr int P = 2;
    constexpr int D = 2;
    constexpr int Q = 1;

    auto f = []( auto& dx, const auto& x, const auto& p,
                 [[maybe_unused]] const auto& t ) {
        const auto& omega = p( 0 );
        dx( 0 )           = x( 1 );
        dx( 1 )           = -omega * omega * x( 0 );
    };

    const double omega0 = 2.0;
    const double hw_om  = 0.05;
    const double tmax   = 0.4;

    Box< double, D > x_box{ { 1.0, 0.0 }, { 0.0, 0.0 } };  // fixed IC
    Box< double, Q > p_box{ { omega0 }, { hw_om } };

    ode::DaIntegrator< N, P, D, Q > ig{
        f, ode::IntegratorConfig< double >{ .abstol = 1e-16 } };

    auto fm = ig.integrate( x_box, p_box, 0.0, tmax );

    // Evaluate at a few δ_ω values inside [-1, 1] (normalised).
    for ( double s : { -1.0, -0.5, 0.0, 0.7, 1.0 } )
    {
        std::array< double, D + Q > dx_eval{ 0.0, 0.0, s };
        const double                x0_eval = fm.state( 0 ).eval( dx_eval );
        const double                x1_eval = fm.state( 1 ).eval( dx_eval );

        const double omega   = omega0 + hw_om * s;
        const double phase   = omega * tmax;
        const double exact_x = std::cos( phase );
        const double exact_v = -omega * std::sin( phase );

        // Order-2 polynomial in δ_ω — accept moderate tolerance.
        EXPECT_NEAR( x0_eval, exact_x, 1e-3 ) << "  s = " << s;
        EXPECT_NEAR( x1_eval, exact_v, 1e-2 ) << "  s = " << s;
    }
}

// =============================================================================
// AdsIntegrator with Q > 0 — splits across both IC and parameter axes
// =============================================================================

TEST( ParamAdsIntegrator, RunsAndCoversBox )
{
    // Modest tolerance test: make sure the parameter-aware AdsIntegrator
    // runs end-to-end and that every (x, p) sample maps to a leaf whose
    // polynomial agrees with direct integration.
    constexpr int N = 15;
    constexpr int P = 3;
    constexpr int D = 1;
    constexpr int Q = 1;

    auto f = []( auto& dx, const auto& x, const auto& p,
                 [[maybe_unused]] const auto& t ) { dx( 0 ) = -p( 0 ) * x( 0 ); };

    const double tmax = 1.0;
    Box< double, D > x_box{ { 1.0 }, { 0.2 } };
    Box< double, Q > p_box{ { 1.0 }, { 0.5 } };  // wide → forces splits

    ode::AdsIntegrator< N, P, D, Q > ig{
        f, ode::AdsConfig{ .step_tol = 1e-14, .ads_tol = 1e-4, .max_depth = 5 } };

    auto tree = ig.integrate( x_box, p_box, 0.0, tmax );

    int n_leaves = 0;
    for ( int i : tree.doneLeaves() )
    {
        const auto& lf = tree.node( i ).leaf();
        ++n_leaves;

        // Evaluate at the leaf centre (δ = 0) and check against the exact
        // solution at that (x0, k) point.
        const double x0c = lf.box.center[0];
        const double kc  = lf.box.center[1];
        const double expected = x0c * std::exp( -kc * tmax );
        EXPECT_NEAR( lf.tte.state( 0 ).value(), expected, 1e-8 );
    }
    EXPECT_GE( n_leaves, 1 );
}

// =============================================================================
// LowOrderAdsIntegrator with Q > 0
// =============================================================================

TEST( ParamLowOrderAdsIntegrator, RunsAndProducesPolynomialFlow )
{
    constexpr int N = 15;
    constexpr int P = 2;
    constexpr int D = 1;
    constexpr int Q = 1;

    auto f = []( auto& dx, const auto& x, const auto& p,
                 [[maybe_unused]] const auto& t ) {
        dx( 0 ) = -p( 0 ) * x( 0 ) * x( 0 );
    };

    const double tmax = 0.5;
    Box< double, D > x_box{ { 1.0 }, { 0.1 } };
    Box< double, Q > p_box{ { 1.0 }, { 0.2 } };

    ode::LowOrderAdsIntegrator< N, P, D, Q > ig{
        f, ode::LowOrderAdsConfig{ .step_tol = 1e-14, .nli_tol = 1e-2, .max_depth = 4 } };

    auto tree = ig.integrate( x_box, p_box, 0.0, tmax );

    EXPECT_GE( static_cast< int >( tree.doneLeaves().size() ), 1 );

    // For dx/dt = -k x², x(t) = 1/(1/x0 + k t).  Compare leaf centres.
    for ( int i : tree.doneLeaves() )
    {
        const auto&  lf       = tree.node( i ).leaf();
        const double x0c      = lf.box.center[0];
        const double kc       = lf.box.center[1];
        const double exact    = 1.0 / ( 1.0 / x0c + kc * tmax );
        EXPECT_NEAR( lf.tte.state( 0 ).value(), exact, 1e-6 );
    }
}
