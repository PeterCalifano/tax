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
// stepDa: verify Taylor coefficients for a single DA step
// =============================================================================

TEST( DaIntegrator, StepDaHarmonicOscillator )
{
    constexpr int N = 10;  // time Taylor order
    constexpr int P = 2;   // DA order
    constexpr int D = 2;   // state dimension

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };
    auto             x0 = ode::makeDaState< P, D >( box );

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    auto [p, h] = ode::stepDa< N, P, D >( f, x0, 0.0, 1e-16 );

    EXPECT_GT( h, 0.0 );

    EXPECT_NEAR( p( 0 )[0].value(), 1.0, 1e-14 );
    EXPECT_NEAR( p( 1 )[0].value(), 0.0, 1e-14 );

    EXPECT_NEAR( p( 0 )[1].value(), 0.0, 1e-14 );
    EXPECT_NEAR( p( 1 )[1].value(), -1.0, 1e-14 );

    EXPECT_NEAR( p( 0 )[2].value(), -0.5, 1e-14 );
}

// =============================================================================
// DaIntegrator::propagate — linear ODE, P=1 should give exact flow map
// =============================================================================

TEST( DaIntegrator, PropagateLinearHarmonicOscillator )
{
    constexpr int N = 20;
    constexpr int P = 1;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    const double          tmax = std::numbers::pi / 2.0;
    ode::DaIntegrator< N, P, D > ig{ ode::IntegratorConfig< double >{ .abstol = 1e-16 } };
    auto fm = ig.integrate( f, box, 0.0, tmax );

    EXPECT_NEAR( fm.state( 0 ).value(), 0.0, 1e-10 );
    EXPECT_NEAR( fm.state( 1 ).value(), -1.0, 1e-10 );

    MultiIndex< D > e_dx{ 1, 0 };
    MultiIndex< D > e_dv{ 0, 1 };
    EXPECT_NEAR( fm.state( 0 ).coeff( e_dx ), 0.0, 1e-10 );
    EXPECT_NEAR( fm.state( 0 ).coeff( e_dv ), 0.1, 1e-10 );

    EXPECT_NEAR( fm.state( 1 ).coeff( e_dx ), -0.1, 1e-10 );
    EXPECT_NEAR( fm.state( 1 ).coeff( e_dv ), 0.0, 1e-10 );
}

// =============================================================================
// DaIntegrator::propagate — point evaluation matches direct integration
// =============================================================================

TEST( DaIntegrator, PropagatePointEvaluation )
{
    constexpr int N = 20;
    constexpr int P = 3;
    constexpr int D = 2;

    using DA = TEn< P, D >;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    const double          tmax = 1.0;
    ode::DaIntegrator< N, P, D > ig{ ode::IntegratorConfig< double >{ .abstol = 1e-16 } };
    auto fm = ig.integrate( f, box, 0.0, tmax );

    DA::Input delta{ 0.5, -0.3 };
    double    x0_pt = 1.0 + 0.1 * 0.5;
    double    v0_pt = 0.0 + 0.1 * ( -0.3 );

    double x_exact = x0_pt * std::cos( tmax ) + v0_pt * std::sin( tmax );
    double v_exact = -x0_pt * std::sin( tmax ) + v0_pt * std::cos( tmax );

    EXPECT_NEAR( fm.state( 0 ).eval( delta ), x_exact, 1e-10 );
    EXPECT_NEAR( fm.state( 1 ).eval( delta ), v_exact, 1e-10 );
}

// =============================================================================
// AdsIntegrator — no splitting needed for a linear system
// =============================================================================

TEST( AdsIntegrator, NoSplitLinearSystem )
{
    constexpr int N = 20;
    constexpr int P = 2;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.1, 0.1 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 );
    };

    ode::AdsIntegrator< N, P, D > ig{ ode::AdsConfig{ .step_tol = 1e-16, .ads_tol = 1e-6 } };
    auto                          tree = ig.integrate( f, box, 0.0, 1.0 );

    EXPECT_EQ( tree.numDone(), 1 );
    EXPECT_TRUE( tree.empty() );
}

// =============================================================================
// AdsIntegrator — nonlinear ODE triggers splitting
// =============================================================================

TEST( AdsIntegrator, SplitsNonlinearODE )
{
    constexpr int N = 15;
    constexpr int P = 3;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.5, 0.5 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 ) - x( 0 ) * x( 0 ) * x( 0 );
    };

    ode::AdsIntegrator< N, P, D > ig{ ode::AdsConfig{
        .step_tol = 1e-12, .ads_tol = 1e-4, .max_depth = 6 } };

    int                                splits_observed = 0;
    ig.on_split = [&]( const ode::SplitEvent< P, D >& ev ) {
        ++splits_observed;
        EXPECT_GE( ev.split_dim, 0 );
        EXPECT_LT( ev.split_dim, D );
        EXPECT_GE( ev.parent_depth, 0 );
        EXPECT_GE( ev.truncation_error, 1e-4 );
    };

    auto tree = ig.integrate( f, box, 0.0, 3.0 );

    EXPECT_GT( tree.numDone(), 1 );
    EXPECT_TRUE( tree.empty() );
    EXPECT_GT( splits_observed, 0 );
}

// =============================================================================
// AdsIntegrator — point accuracy across subdomains
// =============================================================================

TEST( AdsIntegrator, PointAccuracyAcrossSubdomains )
{
    constexpr int N = 15;
    constexpr int P = 3;
    constexpr int D = 2;

    Box< double, D > box{ { 1.0, 0.0 }, { 0.5, 0.5 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        dx( 0 ) = x( 1 );
        dx( 1 ) = -x( 0 ) - x( 0 ) * x( 0 ) * x( 0 );
    };

    const double                       tmax = 2.0;
    ode::AdsIntegrator< N, P, D > ads_ig{ ode::AdsConfig{
        .step_tol = 1e-12, .ads_tol = 1e-4, .max_depth = 8 } };
    auto tree = ads_ig.integrate( f, box, 0.0, tmax );

    ode::Integrator< N > scalar_ig{ ode::IntegratorConfig< double >{ .abstol = 1e-16 } };

    const std::array< std::array< double, 2 >, 3 > test_points = {
        { { 1.2, 0.1 }, { 0.8, -0.3 }, { 1.0, 0.0 } } };

    for ( const auto& pt : test_points )
    {
        std::array< double, D > delta;
        for ( int k = 0; k < D; ++k )
            delta[k] = ( pt[k] - box.center[k] ) / box.halfWidth[k];

        bool in_domain = true;
        for ( int k = 0; k < D; ++k )
            if ( std::abs( delta[k] ) > 1.0 ) in_domain = false;
        ASSERT_TRUE( in_domain );

        const int idx = tree.findLeaf( { pt[0], pt[1] } );
        ASSERT_GE( idx, 0 );

        const auto&             leaf = tree.node( idx ).leaf();
        std::array< double, D > local_delta;
        for ( int k = 0; k < D; ++k )
            local_delta[k] = ( pt[k] - leaf.box.center[k] ) / leaf.box.halfWidth[k];

        double x_ads = leaf.tte.state( 0 ).eval( local_delta );
        double v_ads = leaf.tte.state( 1 ).eval( local_delta );

        Eigen::Vector2d x0_pt( pt[0], pt[1] );
        auto            sol_ref = scalar_ig.integrate( f, x0_pt, 0.0, tmax );

        EXPECT_NEAR( x_ads, sol_ref.x.back()( 0 ), 1e-2 );
        EXPECT_NEAR( v_ads, sol_ref.x.back()( 1 ), 1e-2 );
    }
}

// =============================================================================
// Kepler problem with ADS
// =============================================================================

TEST( AdsIntegrator, KeplerOrbitSplits )
{
    constexpr int N = 15;
    constexpr int P = 3;
    constexpr int D = 4;

    Box< double, D > box{ { 1.0, 0.0, 0.0, 1.0 }, { 0.01, 0.01, 0.01, 0.05 } };

    auto f = []( auto& dx, const auto& x, [[maybe_unused]] const auto& t ) {
        using std::sqrt;
        auto r2 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 );
        auto r  = sqrt( r2 );
        auto r3 = r2 * r;
        dx( 0 ) = x( 2 );
        dx( 1 ) = x( 3 );
        dx( 2 ) = -x( 0 ) / r3;
        dx( 3 ) = -x( 1 ) / r3;
    };

    const double                  tmax = std::numbers::pi;
    ode::AdsIntegrator< N, P, D > ads_ig{ ode::AdsConfig{
        .step_tol = 1e-14, .ads_tol = 1e-3, .max_depth = 4 } };
    auto                          tree = ads_ig.integrate( f, box, 0.0, tmax );

    EXPECT_TRUE( tree.empty() );
    EXPECT_GE( tree.numDone(), 1 );

    Eigen::Vector< double, D > x0c;
    x0c << 1.0, 0.0, 0.0, 1.0;

    ode::Integrator< N > vec_ig{ ode::IntegratorConfig< double >{ .abstol = 1e-16 } };
    auto                 sol_ref = vec_ig.integrate( f, x0c, 0.0, tmax );

    bool found = false;
    for ( int di : tree.doneLeaves() )
    {
        const auto& leaf = tree.node( di ).leaf();
        if ( !leaf.box.contains( box.center ) ) continue;

        std::array< double, D > local_delta{};
        for ( int k = 0; k < D; ++k )
            local_delta[k] = ( box.center[k] - leaf.box.center[k] ) / leaf.box.halfWidth[k];

        for ( int k = 0; k < D; ++k )
        {
            double val_ads = leaf.tte.state( k ).eval( local_delta );
            EXPECT_NEAR( val_ads, sol_ref.x.back()( k ), 1e-4 );
        }
        found = true;
        break;
    }
    EXPECT_TRUE( found );
}

// =============================================================================
// Configuration validation
// =============================================================================

TEST( AdsIntegrator, ConfigRejectsInvalidValues )
{
    using AI = ode::AdsIntegrator< 10, 2, 2 >;
    EXPECT_THROW( ( AI{ ode::AdsConfig{ .step_tol = 0.0 } } ), std::invalid_argument );
    EXPECT_THROW( ( AI{ ode::AdsConfig{ .ads_tol = 0.0 } } ), std::invalid_argument );
    EXPECT_THROW( ( AI{ ode::AdsConfig{ .max_depth = -1 } } ), std::invalid_argument );
    EXPECT_THROW( ( AI{ ode::AdsConfig{ .max_steps = 0 } } ), std::invalid_argument );
}
