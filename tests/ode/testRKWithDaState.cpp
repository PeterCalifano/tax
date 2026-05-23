// tests/ode/testRKWithDaState.cpp
//
// Propagate planar Kepler (e=0.5, GM=a=1) with State =
// Eigen::Matrix<TEn<P,M>, 4, 1> across the five RK families and verify:
//   (a) constant DA term  ≈ double-state propagation     (Task 7)
//   (b) linear DA term    ≈ finite-difference STM        (Task 8)

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cmath>

#include <tax/ode.hpp>
#include <tax/tax.hpp>

namespace {

constexpr int P = 2;
constexpr int M = 4;                          // 4 IC DA variables
constexpr double kEcc        = 0.5;
constexpr double kPeriapsis  = 1.0 - kEcc;
const     double kVPeriapsis = std::sqrt( ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
constexpr double kPeriod     = 2.0 * M_PI;
constexpr double kHalfWidth  = 1e-3;

using DA       = tax::TEn< P, M >;
using StateD   = Eigen::Matrix< double, 4, 1 >;
using StateDA  = Eigen::Matrix< DA,     4, 1 >;

StateD make_ic_double()
{
    StateD x0;
    x0( 0 ) = kPeriapsis;  x0( 1 ) = 0.0;
    x0( 2 ) = 0.0;         x0( 3 ) = kVPeriapsis;
    return x0;
}

StateDA make_ic_da()
{
    StateD c = make_ic_double();
    StateDA x0;
    for ( int i = 0; i < 4; ++i )
        x0( i ) = DA( c( i ) ) + DA( kHalfWidth ) * DA::variable( 0.0, i );
    return x0;
}

template < class S >
auto make_rhs()
{
    return []( const S& s, double /*t*/ )
    {
        using std::sqrt;
        using tax::sqrt;
        S out;
        const auto x  = s( 0 );
        const auto y  = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        out( 0 ) = s( 2 );
        out( 1 ) = s( 3 );
        out( 2 ) = -x / r3;
        out( 3 ) = -y / r3;
        return out;
    };
}

template < template < class, class, bool, class > class IntegratorAlias >
void check_constant_term_matches_double( const char* method_name,
                                         double tol_close )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    using IntegD  = IntegratorAlias<
        StateD,  tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateD >() ) > >;
    using IntegDA = IntegratorAlias<
        StateDA, tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateDA >() ) > >;

    IntegD  integ_d { make_rhs< StateD  >(), cfg };
    IntegDA integ_da{ make_rhs< StateDA >(), cfg };

    auto sol_d  = integ_d .integrate( make_ic_double(), 0.0, kPeriod );
    auto sol_da = integ_da.integrate( make_ic_da(),     0.0, kPeriod );

    const StateD&  xT_d  = sol_d .x.back();
    const StateDA& xT_da = sol_da.x.back();
    for ( int i = 0; i < 4; ++i )
    {
        EXPECT_NEAR( xT_da( i )[ 0 ], xT_d( i ), tol_close )
            << "method=" << method_name << " component=" << i;
    }
}

}  // namespace

TEST( OdeRKWithDaState, ConstantTermVerner78 )
{
    check_constant_term_matches_double< tax::ode::Verner78 >( "Verner78", 1e-8 );
}

TEST( OdeRKWithDaState, ConstantTermVerner89 )
{
    check_constant_term_matches_double< tax::ode::Verner89 >( "Verner89", 1e-9 );
}

TEST( OdeRKWithDaState, ConstantTermFehlberg78 )
{
    check_constant_term_matches_double< tax::ode::Fehlberg78 >( "Fehlberg78", 1e-8 );
}

TEST( OdeRKWithDaState, ConstantTermFeagin12 )
{
    check_constant_term_matches_double< tax::ode::Feagin12 >( "Feagin12", 1e-10 );
}

TEST( OdeRKWithDaState, ConstantTermFeagin14 )
{
    check_constant_term_matches_double< tax::ode::Feagin14 >( "Feagin14", 1e-11 );
}
