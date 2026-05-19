#include <gtest/gtest.h>
#include <tax/tax.hpp>
#include <vector>

TEST(ForEachMonomial, VisitsAllInGradedLexOrder) {
    std::vector<tax::MultiIndex<2>> visited;
    tax::forEachMonomial<2, 3>([&](const tax::MultiIndex<2>& a) {
        visited.push_back(a);
    });
    ASSERT_EQ(visited.size(), tax::numMonomials(3, 2));
    EXPECT_EQ(visited[0], (tax::MultiIndex<2>{0, 0}));
    EXPECT_EQ(visited[1], (tax::MultiIndex<2>{1, 0}));
    EXPECT_EQ(visited[2], (tax::MultiIndex<2>{0, 1}));
}

TEST(ForEachSubIndex, SumsToOuter) {
    // For an outer multi-index (2, 1), sub-indices (k, alpha-k) span all
    // componentwise <= alpha pairs.
    using MI = tax::MultiIndex<2>;
    int count = 0;
    tax::forEachSubIndex<2>(MI{2, 1}, [&](const MI& k, const MI& sub) {
        EXPECT_EQ(k[0] + sub[0], 2);
        EXPECT_EQ(k[1] + sub[1], 1);
        ++count;
    });
    EXPECT_EQ(count, (2 + 1) * (1 + 1));  // (alpha_0+1)*(alpha_1+1)
}
