#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include <tax/tax.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

namespace
{

ode::VernerConfig cfg( double abstol, double reltol = 0.0 )
{
    return ode::VernerConfig{ .abstol = abstol, .reltol = reltol };
}

}  // namespace

// =============================================================================
// Scalar ODE — Verner 78
// =============================================================================

TEST( Verner78Scalar, ExponentialGrowth )
{
    auto f = []( const double& x, double /*t*/ ) -> double { return x; };
    ode::Verner78< double > ig{ f, cfg( 1e-12 ) };

    auto sol = ig.integrate( 1.0, 0.0, 1.0 );

    EXPECT_NEAR( sol.t.back(), 1.0, 1e-13 );
    EXPECT_NEAR( sol.x.back(), std::exp( 1.0 ), 1e-10 );
    EXPECT_GT( sol.n_accepted, 0 );
}

TEST( Verner78Scalar, ExponentialDecay )
{
    auto f = []( const double& x, double /*t*/ ) -> double { return -x; };
    ode::Verner78< double > ig{ f, cfg( 1e-12 ) };

    auto sol = ig.integrate( 3.0, 0.0, 2.0 );

    EXPECT_NEAR( sol.t.back(), 2.0, 1e-13 );
    EXPECT_NEAR( sol.x.back(), 3.0 * std::exp( -2.0 ), 1e-10 );
}

TEST( Verner78Scalar, BackwardIntegration )
{
    auto f = []( const double& x, double /*t*/ ) -> double { return x; };
    ode::Verner78< double > ig{ f, cfg( 1e-12 ) };

    auto sol = ig.integrate( 1.0, 0.0, -1.0 );

    EXPECT_NEAR( sol.t.back(), -1.0, 1e-13 );
    EXPECT_NEAR( sol.x.back(), std::exp( -1.0 ), 1e-10 );
}

// =============================================================================
// Scalar ODE — Verner 89 (same problems, tighter accuracy)
// =============================================================================

TEST( Verner89Scalar, ExponentialGrowth )
{
    auto f = []( const double& x, double /*t*/ ) -> double { return x; };
    ode::Verner89< double > ig{ f, cfg( 1e-13 ) };

    auto sol = ig.integrate( 1.0, 0.0, 1.0 );

    EXPECT_NEAR( sol.t.back(), 1.0, 1e-13 );
    EXPECT_NEAR( sol.x.back(), std::exp( 1.0 ), 1e-11 );
}

TEST( Verner89Scalar, CosineForcing )
{
    auto f = []( const double& /*x*/, double t ) -> double { return std::cos( t ); };
    ode::Verner89< double > ig{ f, cfg( 1e-13 ) };

    auto sol = ig.integrate( 0.0, 0.0, std::numbers::pi );

    EXPECT_NEAR( sol.t.back(), std::numbers::pi, 1e-13 );
    EXPECT_NEAR( sol.x.back(), std::sin( std::numbers::pi ), 1e-11 );
}

// =============================================================================
// Vector ODE — Eigen
// =============================================================================

TEST( Verner78Vector, HarmonicOscillator )
{
    auto f = []( const Eigen::Vector2d& x, double /*t*/ ) -> Eigen::Vector2d {
        return Eigen::Vector2d{ x( 1 ), -x( 0 ) };
    };
    ode::Verner78< Eigen::Vector2d > ig{ f, cfg( 1e-12 ) };

    const double tmax = 2.0 * std::numbers::pi;
    auto         sol  = ig.integrate( Eigen::Vector2d{ 1.0, 0.0 }, 0.0, tmax );

    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-9 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-9 );
}

TEST( Verner89Vector, KeplerCircularOrbit )
{
    using Vec = Eigen::Vector< double, 4 >;
    auto f    = []( const Vec& x, double /*t*/ ) -> Vec {
        const double r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
        const double r3 = r2 * std::sqrt( r2 );
        Vec          dx;
        dx << x( 2 ), x( 3 ), -x( 0 ) / r3, -x( 1 ) / r3;
        return dx;
    };
    ode::Verner89< Vec > ig{ f, cfg( 1e-13 ) };

    Vec x0;
    x0 << 1.0, 0.0, 0.0, 1.0;

    const double tmax = 2.0 * std::numbers::pi;
    auto         sol  = ig.integrate( x0, 0.0, tmax );

    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-9 );
    EXPECT_NEAR( sol.x.back()( 1 ), 0.0, 1e-9 );
    EXPECT_NEAR( sol.x.back()( 2 ), 0.0, 1e-9 );
    EXPECT_NEAR( sol.x.back()( 3 ), 1.0, 1e-9 );

    const auto&  xf = sol.x.back();
    const double r  = std::sqrt( xf( 0 ) * xf( 0 ) + xf( 1 ) * xf( 1 ) );
    const double v2 = xf( 2 ) * xf( 2 ) + xf( 3 ) * xf( 3 );
    EXPECT_NEAR( 0.5 * v2 - 1.0 / r, -0.5, 1e-9 );
}

// =============================================================================
// DA scalar (state-expanded series) — verify polynomial flow
// =============================================================================

TEST( Verner78DaScalar, ExponentialDecayPolynomial )
{
    // dx/dt = -x, x0 = TE<5>::variable(x0_value)
    // Solution: x(t) = x0 * exp(-t)
    // The polynomial flow w.r.t. dx is:  (x0_value + dx) * exp(-t).
    using DA = TE< 5 >;
    auto f   = []( const DA& x, double /*t*/ ) -> DA { return -x; };

    ode::Verner78< DA > ig{ f, cfg( 1e-12 ) };
    auto                sol = ig.integrate( DA::variable( 2.0 ), 0.0, 1.0 );

    const DA&    xf       = sol.x.back();
    const double expected = std::exp( -1.0 );
    EXPECT_NEAR( xf[0], 2.0 * expected, 1e-9 );  // constant term
    EXPECT_NEAR( xf[1], 1.0 * expected, 1e-9 );  // d/d(dx) coeff
    // Higher-order coefficients should be ~0 for a linear ODE.
    EXPECT_NEAR( xf[2], 0.0, 1e-9 );
    EXPECT_NEAR( xf[3], 0.0, 1e-9 );
}

// =============================================================================
// DA vector (DA flow map) — harmonic oscillator
// =============================================================================

TEST( Verner89DaVector, LinearHarmonicOscillator )
{
    constexpr int P = 2;
    constexpr int D = 2;
    using DA        = TEn< P, D >;
    using VecDa     = Eigen::Matrix< DA, D, 1 >;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( const VecDa& x, double /*t*/ ) -> VecDa {
        VecDa dx;
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
        return dx;
    };

    auto x0 = ode::makeDaState< P, D >( box );

    const double         tmax = std::numbers::pi / 2.0;
    ode::Verner89< VecDa > ig{ f, cfg( 1e-13 ) };
    auto                 sol = ig.integrate( x0, 0.0, tmax );

    const VecDa& xf = sol.x.back();

    // x(π/2) = v0, v(π/2) = -x0 for harmonic oscillator.
    EXPECT_NEAR( xf( 0 )[0], 0.0, 1e-9 );
    EXPECT_NEAR( xf( 1 )[0], -1.0, 1e-9 );

    MultiIndex< D > e_dx{ 1, 0 };
    MultiIndex< D > e_dv{ 0, 1 };
    EXPECT_NEAR( xf( 0 ).coeff( e_dx ), 0.0, 1e-9 );  // d x_f / d x0
    EXPECT_NEAR( xf( 0 ).coeff( e_dv ), 0.1, 1e-9 );  // d x_f / d v0 * halfwidth
    EXPECT_NEAR( xf( 1 ).coeff( e_dx ), -0.1, 1e-9 );
    EXPECT_NEAR( xf( 1 ).coeff( e_dv ), 0.0, 1e-9 );
}

// =============================================================================
// VernerAdsIntegrator — sanity check on a linear harmonic oscillator
// =============================================================================

TEST( Verner78Ads, LinearHarmonicOscillatorSingleLeaf )
{
    constexpr int P = 3;
    constexpr int D = 2;
    using DA        = TEn< P, D >;
    using VecDa     = Eigen::Matrix< DA, D, 1 >;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.05, 0.05 } };

    auto f = []( const VecDa& x, double /*t*/ ) -> VecDa {
        VecDa dx;
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
        return dx;
    };

    ode::VernerAdsConfig acfg{};
    acfg.step.abstol = 1e-13;
    acfg.ads_tol     = 1e-3;  // small linear ODE — no splits expected
    acfg.max_depth   = 10;

    ode::Verner78AdsIntegrator< P, D > ig{ f, acfg };
    auto                                tree = ig.integrate( box, 0.0, 1.0 );

    const auto& done = tree.doneLeaves();
    ASSERT_EQ( done.size(), 1u );
    const auto& leaf = tree.node( done.front() ).leaf();
    EXPECT_NEAR( leaf.tte.state( 0 )[0], std::cos( 1.0 ), 1e-9 );
    EXPECT_NEAR( leaf.tte.state( 1 )[0], -std::sin( 1.0 ), 1e-9 );
}

TEST( Verner89Ads, NonlinearDuffingDoesSplit )
{
    constexpr int P = 4;
    constexpr int D = 2;
    using DA        = TEn< P, D >;
    using VecDa     = Eigen::Matrix< DA, D, 1 >;

    // Cubic Duffing-like dynamics: dx = v, dv = -x - x^3
    // A reasonably wide IC box should force at least one ADS split at low ads_tol.
    Box< double, D > box{ { 1.0, 0.0 }, { 0.5, 0.5 } };

    auto f = []( const VecDa& x, double /*t*/ ) -> VecDa {
        VecDa dx;
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 ) - x( 0 ) * x( 0 ) * x( 0 );
        return dx;
    };

    ode::VernerAdsConfig acfg{};
    acfg.step.abstol = 1e-12;
    acfg.ads_tol     = 1e-5;
    acfg.max_depth   = 6;

    int n_splits = 0;
    ode::Verner89AdsIntegrator< P, D > ig{ f, acfg };
    ig.on_split = [&]( const ode::SplitEvent< P, D >& ) { ++n_splits; };

    auto tree = ig.integrate( box, 0.0, 1.5 );
    EXPECT_GT( tree.doneLeaves().size(), 1u );
    EXPECT_GT( n_splits, 0 );
}

// =============================================================================
// Consistency check — Verner78 and Verner89 agree with the Taylor integrator
// on a stiff-free linear problem within reasonable tolerance.
// =============================================================================

TEST( VernerConsistency, MatchesTaylorIntegratorScalar )
{
    auto f_taylor = []( const auto& x, [[maybe_unused]] const auto& t ) { return -x; };
    ode::Integrator< 25, double > taylor_ig{ f_taylor,
                                              ode::IntegratorConfig< double >{ .abstol = 1e-16 } };
    auto                          ref = taylor_ig.integrate( 1.0, 0.0, 2.0 );

    auto f                            = []( const double& x, double /*t*/ ) -> double { return -x; };

    ode::Verner78< double > v78{ f, cfg( 1e-12 ) };
    ode::Verner89< double > v89{ f, cfg( 1e-13 ) };

    EXPECT_NEAR( v78.integrate( 1.0, 0.0, 2.0 ).x.back(), ref.x.back(), 1e-9 );
    EXPECT_NEAR( v89.integrate( 1.0, 0.0, 2.0 ).x.back(), ref.x.back(), 1e-10 );
}
