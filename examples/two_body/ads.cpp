// examples/two_body/ads.cpp
//
// ADS propagation of the planar Kepler problem with a small IC box.
// Truncation criterion (Wittig).

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <tax/ads.hpp>
#include <tax/ads/io.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

#include "common.hpp"

int main()
{
    using namespace example::two_body;
    using namespace tax::ode::methods;

    constexpr int P = 6;
    constexpr int M = 4;

    constexpr int    kNOrbits = 3;
    constexpr int    kNSnaps  = 200;
    const     double tFinal   = kNOrbits * kPeriod;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    tax::ads::Box< double, M > ic_box{ icCenter(),
                                       tax::la::VecNT< M, double >::Constant( 1e-3 ) };

    const auto t0   = std::chrono::high_resolution_clock::now();
    auto       tree = tax::ads::propagate< P >(
        Verner89{}, tax::ads::TruncationCriterion{ /*tol=*/1e-4, /*maxDepth=*/8 },
        rhs(), ic_box, icCenter(), 0.0, tFinal, cfg );
    const auto t1   = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration< double, std::milli >( t1 - t0 ).count();

    // IC-centerpoint scalar reference trajectory for plotting.
    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-13;
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Taylor< 16 >{}, rhs(), icCenter(), 0.0, tFinal, ref_cfg );

    const auto times = tax::ode::linspace( 0.0, tFinal, kNSnaps + 1 );
    tax::ode::writeCsv( ref_sol, times, "ads_traj.csv" );
    tax::ads::writeTreeCsv( tree, tFinal, "ads_tree.csv" );
    tax::ads::writeBoxCountCsv( tree, tFinal, times, "ads_boxcount.csv" );

    // Ground-truth distribution snapshots: scalar-propagate kNSamples
    // uniform-random ICs in the box and dump (sample, t, x0..x3) rows
    // at kNDistSnaps times. Used by plot.py to render a 2x3 panel of
    // scatter snapshots showing how the cloud spreads.
    constexpr int       kNSamples   = 200;
    constexpr int       kNDistSnaps = 6;
    const auto          dist_times  = tax::ode::linspace( 0.0, tFinal, kNDistSnaps );
    std::mt19937                          rng( 42 );
    std::uniform_real_distribution< double > uni( -1.0, 1.0 );
    std::ofstream                         dist( "two_body_distribution.csv" );
    dist << "sample,t,x0,x1,x2,x3\n";
    for ( int s = 0; s < kNSamples; ++s )
    {
        tax::la::VecNT< M, double > xi_local;
        for ( int j = 0; j < M; ++j ) xi_local( j ) = uni( rng );
        const auto ic_sample = ic_box.denormalize( xi_local );
        auto sample_sol = tax::ode::propagate< /*Dense=*/true >(
            Taylor< 16 >{}, rhs(), ic_sample, 0.0, tFinal, ref_cfg );
        for ( double t : dist_times )
        {
            const auto x = sample_sol( t );
            dist << s << ',' << t;
            for ( int j = 0; j < 4; ++j ) dist << ',' << x( j );
            dist << '\n';
        }
    }

    std::cout << "[ads] " << elapsed_ms << " ms, " << tree.done().size()
              << " done leaves\n";
    std::ofstream( "ads_timing.txt" )
        << "method=ads\ncriterion=truncation\n"
        << "P=" << P << "\nM=" << M << '\n'
        << "elapsed_ms=" << elapsed_ms << '\n'
        << "n_done="     << tree.done().size() << '\n'
        << "t_final="    << tFinal     << '\n'
        << "n_orbits="   << kNOrbits   << '\n';
    return 0;
}
