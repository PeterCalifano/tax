#include <gtest/gtest.h>
#include <tax/tax.hpp>

TEST(NumMonomials, KnownValues) {
    // numMonomials(N, M) = C(N+M, M)
    EXPECT_EQ(tax::numMonomials(0, 1), 1u);
    EXPECT_EQ(tax::numMonomials(3, 1), 4u);     // 1 + x + x^2 + x^3
    EXPECT_EQ(tax::numMonomials(2, 2), 6u);     // 1, x, y, x^2, xy, y^2
    EXPECT_EQ(tax::numMonomials(4, 3), 35u);
}

TEST(Binom, KnownValues) {
    EXPECT_EQ(tax::detail::binom(5, 2), 10u);
    EXPECT_EQ(tax::detail::binom(7, 0), 1u);
    EXPECT_EQ(tax::detail::binom(7, 7), 1u);
    EXPECT_EQ(tax::detail::binom(-1, 0), 0u);
    EXPECT_EQ(tax::detail::binom(3, 5), 0u);
}

TEST(FlatIndex, RoundTripUni) {
    EXPECT_EQ(tax::flatIndex<1>({0}), 0u);
    EXPECT_EQ(tax::flatIndex<1>({3}), 3u);
}

TEST(FlatIndex, RoundTripBiVar) {
    // Graded-lex over (a, b): (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
    using MI = tax::MultiIndex<2>;
    EXPECT_EQ(tax::flatIndex<2>(MI{0, 0}), 0u);
    EXPECT_EQ(tax::flatIndex<2>(MI{1, 0}), 1u);
    EXPECT_EQ(tax::flatIndex<2>(MI{0, 1}), 2u);
    EXPECT_EQ(tax::flatIndex<2>(MI{2, 0}), 3u);
    EXPECT_EQ(tax::flatIndex<2>(MI{1, 1}), 4u);
    EXPECT_EQ(tax::flatIndex<2>(MI{0, 2}), 5u);
}

TEST(TotalDegree, Sum) {
    using MI = tax::MultiIndex<3>;
    EXPECT_EQ(tax::totalDegree(MI{0, 0, 0}), 0);
    EXPECT_EQ(tax::totalDegree(MI{2, 1, 3}), 6);
}

TEST(UnflatIndex, RoundTripWithFlatIndex) {
    constexpr int N = 3;
    constexpr int M = 2;
    for (std::size_t k = 0; k < tax::numMonomials(N, M); ++k) {
        auto alpha = tax::unflatIndex<M>(k);
        EXPECT_EQ(tax::flatIndex<M>(alpha), k) << "round-trip failed at k=" << k;
    }
}

TEST(UnflatIndex, KnownValuesBiVar) {
    using MI = tax::MultiIndex<2>;
    EXPECT_EQ(tax::unflatIndex<2>(0), (MI{0, 0}));
    EXPECT_EQ(tax::unflatIndex<2>(1), (MI{1, 0}));
    EXPECT_EQ(tax::unflatIndex<2>(2), (MI{0, 1}));
    EXPECT_EQ(tax::unflatIndex<2>(3), (MI{2, 0}));
    EXPECT_EQ(tax::unflatIndex<2>(4), (MI{1, 1}));
    EXPECT_EQ(tax::unflatIndex<2>(5), (MI{0, 2}));
}

TEST(DegreeOf, ContainsTotalDegreeOfEachMonomial) {
    // For N=3, M=2: graded-lex order yields degrees [0, 1,1, 2,2,2, 3,3,3,3].
    constexpr tax::DegreeOf<3, 2> deg{};
    const std::array<int, 10> expected{0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
    ASSERT_EQ(deg.size, expected.size());
    for (std::size_t k = 0; k < deg.size; ++k) EXPECT_EQ(deg.value[k], expected[k]) << "k=" << k;
}
