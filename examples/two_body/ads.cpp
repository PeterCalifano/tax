// examples/two_body/ads.cpp
//
// ADS propagation of the planar Kepler problem with a small IC box.
// Truncation criterion (Wittig).

#include <chrono>
#include <fstream>
#include <iostream>

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
