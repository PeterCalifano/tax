#include <gtest/gtest.h>
#include "../testUtils.hpp"
#include <Eigen/Core>

TEST( EigenNumTraits, TaylorAsScalar )
{
    using TE = tax::TE< 3, 2 >;
    Eigen::Matrix< TE, 2, 1 > v;
    typename TE::Input p{ 1.0, 2.0 };
    v( 0 ) = TE::variable< 0 >( p );
    v( 1 ) = TE::variable< 1 >( p );
    auto sum = v( 0 ) + v( 1 );
    EXPECT_NEAR( sum.value(), 3.0, 1e-12 );
}

TEST( EigenNumTraits, MatrixOfTE )
{
    using TE = tax::TE< 2, 2 >;
    Eigen::Matrix< TE, 2, 2 > A;
    typename TE::Input p{ 1.0, 0.0 };
    A( 0, 0 ) = TE::variable< 0 >( p );
    A( 0, 1 ) = TE::constant( 2.0 );
    A( 1, 0 ) = TE::constant( 3.0 );
    A( 1, 1 ) = TE::variable< 1 >( p );
    EXPECT_NEAR( A( 0, 0 ).value(), 1.0, 1e-12 );
    EXPECT_NEAR( A( 1, 1 ).value(), 0.0, 1e-12 );
}
