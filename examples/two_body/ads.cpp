// examples/two_body/ads.cpp
//
// ADS propagation of the planar Kepler problem with a small IC box.
// Truncation criterion (Wittig). Writes:
//   ads_traj.csv       — IC-centerpoint scalar trajectory (snapshots)
//   ads_tree.csv       — one row per done + retired leaf:
//                          idx, parent, sibling, depth, retired, done,
//                          t_entry, t_exit, cx0..cx3, hw0..hw3
//   ads_boxcount.csv   — t, n_alive — number of boxes alive at each snapshot
//   ads_timing.txt     — wall-clock + tree stats
//
// "Alive at time T" means the leaf existed (was propagating or done)
// at time T: tEntry <= T <= tExit, where tExit = either the children's
// tEntry (for retired leaves) or t_final (for done leaves).

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

#include <tax/ads.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

namespace
{

struct LeafInfo
{
    int                    idx;
    int                    parent;
    int                    sibling;
    int                    depth;
    bool                   retired;
    bool                   done;
    double                 t_entry;
    double                 t_exit;
    std::array< double, 4 > center;
    std::array< double, 4 > halfWidth;
};

}  // namespace

int main()
{
    using namespace example::two_body;

    constexpr int P = 6;
    constexpr int M = 4;
    constexpr int D = 4;

    using TE      = tax::TE< P, M >;
    using DAState = tax::la::VecNT< D, TE >;
    using ScState = tax::la::VecNT< D, double >;
    using Stepper = tax::ode::Verner89Stepper< DAState >;

    constexpr int    kNOrbits = 3;
    constexpr int    kNSnaps  = 200;
    const     double tFinal   = kNOrbits * kPeriod;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    // Small IC uncertainty box around the periapsis state.
    tax::ads::Box< double, M > ic_box{
        icCenterArray(),
        std::array< double, M >{ 1e-3, 1e-3, 1e-3, 1e-3 }
    };

    tax::ads::AdsDriver< Stepper, tax::ads::TruncationCriterion > driver{
        tax::ads::TruncationCriterion{ /*tol=*/1e-4, /*maxDepth=*/8 },
        cfg
    };

    const auto t0    = std::chrono::high_resolution_clock::now();
    auto       tree  = driver.run( rhs(), ic_box, icCenter(), 0.0, tFinal );
    const auto t1    = std::chrono::high_resolution_clock::now();
    const double ads_elapsed_ms =
        std::chrono::duration< double, std::milli >( t1 - t0 ).count();

    // ---- Walk the tree: collect done leaves and their retired ancestors.
    std::vector< LeafInfo > leaves;
    std::unordered_set< int > seen;
    auto pushLeaf = [&]( int idx, double t_entry, double t_exit )
    {
        const auto& l = tree.leaf( idx );
        LeafInfo info{};
        info.idx     = idx;
        info.parent  = l.parentIdx;
        info.sibling = l.siblingIdx;
        info.depth   = l.depth;
        info.retired = l.retired;
        info.done    = l.done;
        info.t_entry = t_entry;
        info.t_exit  = t_exit;
        for ( int j = 0; j < M; ++j )
        {
            info.center[ j ]    = l.box.center[ j ];
            info.halfWidth[ j ] = l.box.halfWidth[ j ];
        }
        leaves.push_back( info );
        seen.insert( idx );
    };

    for ( int idx : tree.done() ) pushLeaf( idx, tree.leaf( idx ).tEntry, tFinal );
    for ( int idx : tree.done() )
    {
        int    cur       = tree.leaf( idx ).parentIdx;
        double exit_time = tree.leaf( idx ).tEntry;
        while ( cur >= 0 && !seen.count( cur ) )
        {
            const auto& p = tree.leaf( cur );
            pushLeaf( cur, p.tEntry, exit_time );
            exit_time = p.tEntry;
            cur       = p.parentIdx;
        }
    }

    // ---- Reference: IC-centerpoint scalar trajectory (Dense=true).
    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-13;
    tax::ode::Taylor< 16, ScState, tax::ode::controllers::JorbaZou< double >,
                      /*Dense=*/true, decltype( rhs() ) >
        ref_integ{ rhs(), ref_cfg };
    auto ref_sol = ref_integ.integrate( icCenter(), 0.0, tFinal );

    // ---- Trajectory CSV.
    std::ofstream traj( "ads_traj.csv" );
    traj << "t,x,y,vx,vy\n";
    for ( int i = 0; i <= kNSnaps; ++i )
    {
        const double t = i * tFinal / kNSnaps;
        ScState      x = ref_sol( t );
        traj << t << ',' << x( 0 ) << ',' << x( 1 ) << ','
             << x( 2 ) << ',' << x( 3 ) << '\n';
    }

    // ---- Tree CSV.
    std::ofstream out( "ads_tree.csv" );
    out << "idx,parent,sibling,depth,retired,done,t_entry,t_exit";
    for ( int j = 0; j < M; ++j ) out << ",cx" << j;
    for ( int j = 0; j < M; ++j ) out << ",hw" << j;
    out << '\n';
    for ( const auto& info : leaves )
    {
        out << info.idx << ',' << info.parent << ',' << info.sibling << ','
            << info.depth << ',' << (info.retired ? 1 : 0) << ','
            << (info.done ? 1 : 0) << ','
            << info.t_entry << ',' << info.t_exit;
        for ( int j = 0; j < M; ++j ) out << ',' << info.center[ j ];
        for ( int j = 0; j < M; ++j ) out << ',' << info.halfWidth[ j ];
        out << '\n';
    }

    // ---- Box-count over time (at the same snapshot grid).
    std::ofstream bc( "ads_boxcount.csv" );
    bc << "t,n_alive\n";
    for ( int i = 0; i <= kNSnaps; ++i )
    {
        const double t = i * tFinal / kNSnaps;
        int          n = 0;
        for ( const auto& info : leaves )
            if ( info.t_entry <= t && t <= info.t_exit ) ++n;
        bc << t << ',' << n << '\n';
    }

    // ---- Timing summary.
    int max_depth = 0;
    for ( const auto& info : leaves ) max_depth = std::max( max_depth, info.depth );
    std::ofstream timing( "ads_timing.txt" );
    timing << "method=ads\n"
           << "criterion=truncation\n"
           << "P=" << P << "\nM=" << M << "\nD=" << D << '\n'
           << "elapsed_ms="   << ads_elapsed_ms << '\n'
           << "n_done="       << tree.done().size() << '\n'
           << "n_total="      << leaves.size()      << '\n'
           << "max_depth="    << max_depth          << '\n'
           << "t_final="      << tFinal             << '\n'
           << "n_orbits="     << kNOrbits           << '\n';

    std::cout << "[ads] elapsed: " << ads_elapsed_ms << " ms, "
              << tree.done().size() << " done leaves (max depth " << max_depth << ")\n";
    return 0;
}
