#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>

#include <tax/tax.hpp>
#include <tax/ads/ads_tree.hpp>
#include <tax/ads/box.hpp>
#include <tax/ode/taylor_integrator.hpp>

using namespace tax;

// =============================================================================
// LowOrderAdsIntegrator — linear system never splits (NLI ≡ 0).
// =============================================================================

TEST( LowOrderAdsIntegrator, NoSplitLinearSystem )
{
    constexpr int N = 20;
    constexpr int P = 2;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    ode::LowOrderAdsIntegrator< N, P, D > ig{
        f, ode::LowOrderAdsConfig{ .step_tol = 1e-16, .nli_tol = 1e-6 } };
    auto tree = ig.integrate( box, 0.0, 1.0 );

    EXPECT_EQ( tree.numDone(), 1 );
    EXPECT_TRUE( tree.empty() );
}

// =============================================================================
// LowOrderAdsIntegrator — nonlinear ODE triggers splitting.
// =============================================================================

TEST( LowOrderAdsIntegrator, SplitsNonlinearODE )
{
    constexpr int N = 15;
    constexpr int P = 2;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.5, 0.5 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 ) - x( 0 ) * x( 0 ) * x( 0 );
    };

    ode::LowOrderAdsIntegrator< N, P, D > ig{
        f, ode::LowOrderAdsConfig{ .step_tol = 1e-12, .nli_tol = 1e-2, .max_depth = 6 } };

    int splits_observed = 0;
    ig.on_split = [&]( const ode::LowOrderSplitEvent< P, D >& ev ) {
        ++splits_observed;
        EXPECT_GE( ev.split_dim, 0 );
        EXPECT_LT( ev.split_dim, D );
        EXPECT_GE( ev.parent_depth, 0 );
        EXPECT_GT( ev.nonlinearity_index, 1e-2 );
    };

    auto tree = ig.integrate( box, 0.0, 3.0 );

    EXPECT_GT( tree.numDone(), 1 );
    EXPECT_TRUE( tree.empty() );
    EXPECT_GT( splits_observed, 0 );
}

// =============================================================================
// LowOrderAdsIntegrator — point accuracy on the planar two-body problem.
// =============================================================================

TEST( LowOrderAdsIntegrator, KeplerPointAccuracy )
{
    constexpr int N = 12;
    constexpr int P = 2;
    constexpr int D = 4;

    auto kepler = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        using std::sqrt;
        auto r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
        auto r  = sqrt( r2 );
        auto r3 = r2 * r;
        dx( 0 ) = x( 2 );
        dx( 1 ) = x( 3 );
        dx( 2 ) = -x( 0 ) / r3;
        dx( 3 ) = -x( 1 ) / r3;
    };

    // a = 1, e = 0.5 ellipse — periapsis IC.
    constexpr double a    = 1.0;
    constexpr double e    = 0.5;
    const double     rp   = a * ( 1.0 - e );
    const double     vp   = std::sqrt( ( 1.0 + e ) / ( 1.0 - e ) );
    const double     tmax = std::numbers::pi;   // half orbit

    Box< double, D > box{ { rp, 0.0, 0.0, vp }, { 0.0, 0.0, 0.0, 0.02 } };

    ode::LowOrderAdsIntegrator< N, P, D > lo_ig{
        kepler, ode::LowOrderAdsConfig{ .step_tol = 1e-14,
                                        .nli_tol  = 5e-3,
                                        .max_depth = 8 } };
    auto tree = lo_ig.integrate( box, 0.0, tmax );

    EXPECT_TRUE( tree.empty() );

    // Cross-validate every leaf against a plain scalar integration of its
    // box centre — the flow polynomial at δ = 0 must agree to high
    // accuracy regardless of where the splits happened.
    using Vec = Eigen::Vector< double, D >;
    ode::Integrator< N, Vec > scalar_ig{
        kepler, ode::IntegratorConfig< double >{ .abstol = 1e-16 } };

    for ( int idx : tree.doneLeaves() )
    {
        const auto& lf = tree.node( idx ).leaf();
        Vec         x0;
        for ( int k = 0; k < D; ++k ) x0( k ) = lf.box.center[k];

        auto sol = scalar_ig.integrate( x0, 0.0, tmax );

        for ( int k = 0; k < D; ++k )
            EXPECT_NEAR( lf.tte.state( k ).value(), sol.x.back()( k ), 1e-6 )
                << "  leaf " << idx << "  component " << k;
    }
}

// =============================================================================
// LowOrderAdsIntegrator — config validation
// =============================================================================

TEST( LowOrderAdsIntegrator, ConfigRejectsInvalidValues )
{
    auto f = []( auto&, const auto&, [[maybe_unused]] const auto& ) {};
    using IG = ode::LowOrderAdsIntegrator< 4, 2, 1 >;

    EXPECT_THROW( ( IG{ f, ode::LowOrderAdsConfig{ .step_tol = 0.0 } } ),
                  std::invalid_argument );
    EXPECT_THROW( ( IG{ f, ode::LowOrderAdsConfig{ .nli_tol = -1.0 } } ),
                  std::invalid_argument );
    EXPECT_THROW( ( IG{ f, ode::LowOrderAdsConfig{ .max_depth = -1 } } ),
                  std::invalid_argument );
    EXPECT_THROW( ( IG{ f, ode::LowOrderAdsConfig{ .max_steps = 0 } } ),
                  std::invalid_argument );
}
