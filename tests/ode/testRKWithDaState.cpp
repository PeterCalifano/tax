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

namespace {

// Forward-difference STM: column i of STM ≈
//   ( x_T(x0 + eps * e_i) - x_T(x0 - eps * e_i) ) / ( 2 * eps )
template < template < class, class, bool, class > class IntegratorAlias >
Eigen::Matrix< double, 4, 4 > fd_stm( double eps )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    using Integ = IntegratorAlias<
        StateD,  tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateD >() ) > >;
    Integ integ{ make_rhs< StateD >(), cfg };

    Eigen::Matrix< double, 4, 4 > stm;
    for ( int i = 0; i < 4; ++i )
    {
        StateD ic_p = make_ic_double(); ic_p( i ) += eps;
        StateD ic_m = make_ic_double(); ic_m( i ) -= eps;

        auto sol_p = integ.integrate( ic_p, 0.0, kPeriod );
        auto sol_m = integ.integrate( ic_m, 0.0, kPeriod );

        const StateD& xp = sol_p.x.back();
        const StateD& xm = sol_m.x.back();
        for ( int j = 0; j < 4; ++j )
            stm( j, i ) = ( xp( j ) - xm( j ) ) / ( 2.0 * eps );
    }
    return stm;
}

template < template < class, class, bool, class > class IntegratorAlias >
void check_linear_term_matches_fd_stm( const char* method_name, double tol_abs )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    using IntegDA = IntegratorAlias<
        StateDA, tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_rhs< StateDA >() ) > >;
    IntegDA integ_da{ make_rhs< StateDA >(), cfg };
    auto sol_da = integ_da.integrate( make_ic_da(), 0.0, kPeriod );

    const StateDA& xT_da = sol_da.x.back();
    const auto stm_fd = fd_stm< IntegratorAlias >( 1e-6 );

    // DA layout: TEn<2,4>::flatIndex({e_i}) = 1 + i (degree-1 monomial i).
    // halfWidth chain rule: DA-coef at e_i = kHalfWidth * (∂x_T/∂x_0[i]).
    for ( int row = 0; row < 4; ++row )
    {
        for ( int col = 0; col < 4; ++col )
        {
            const std::size_t flat = static_cast< std::size_t >( 1 + col );
            const double      lhs  = xT_da( row )[ flat ] / kHalfWidth;
            const double      rhs  = stm_fd( row, col );
            EXPECT_NEAR( lhs, rhs, tol_abs )
                << "method=" << method_name
                << " row="   << row
                << " col="   << col;
        }
    }
}

}  // namespace

TEST( OdeRKWithDaState, LinearTermVerner78 )
{
    check_linear_term_matches_fd_stm< tax::ode::Verner78 >( "Verner78", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermVerner89 )
{
    check_linear_term_matches_fd_stm< tax::ode::Verner89 >( "Verner89", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermFehlberg78 )
{
    check_linear_term_matches_fd_stm< tax::ode::Fehlberg78 >( "Fehlberg78", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermFeagin12 )
{
    check_linear_term_matches_fd_stm< tax::ode::Feagin12 >( "Feagin12", 1e-5 );
}

TEST( OdeRKWithDaState, LinearTermFeagin14 )
{
    check_linear_term_matches_fd_stm< tax::ode::Feagin14 >( "Feagin14", 1e-5 );
}
