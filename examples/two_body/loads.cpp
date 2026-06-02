// examples/two_body/loads.cpp
//
// LOADS propagation of the planar Kepler problem. Identical to ads.cpp
// except the criterion is NliCriterion (Losacco/Fossà/Armellin 2024).
// Output names use the loads_ prefix.

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

    tax::ads::Box< double, M > ic_box{
        icCenterArray(),
        std::array< double, M >{ 1e-3, 1e-3, 1e-3, 1e-3 }
    };

    tax::ads::AdsDriver< Stepper, tax::ads::NliCriterion > driver{
        tax::ads::NliCriterion{ /*tol=*/0.1, /*maxDepth=*/8 },
        cfg
    };

    const auto t0   = std::chrono::high_resolution_clock::now();
    auto       tree = driver.run( rhs(), ic_box, icCenter(), 0.0, tFinal );
    const auto t1   = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration< double, std::milli >( t1 - t0 ).count();

    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-13;
    tax::ode::Taylor< 16, ScState, tax::ode::controllers::JorbaZou< double >,
                      /*Dense=*/true, decltype( rhs() ) >
        ref_integ{ rhs(), ref_cfg };
    auto ref_sol = ref_integ.integrate( icCenter(), 0.0, tFinal );

    const auto times = tax::ode::linspace( 0.0, tFinal, kNSnaps + 1 );
    tax::ode::writeCsv( ref_sol, times, "loads_traj.csv" );
    tax::ads::writeTreeCsv( tree, tFinal, "loads_tree.csv" );
    tax::ads::writeBoxCountCsv( tree, tFinal, times, "loads_boxcount.csv" );

    std::cout << "[loads] " << elapsed_ms << " ms, " << tree.done().size()
              << " done leaves\n";
    std::ofstream( "loads_timing.txt" )
        << "method=loads\ncriterion=nli\n"
        << "P=" << P << "\nM=" << M << "\nD=" << D << '\n'
        << "elapsed_ms=" << elapsed_ms << '\n'
        << "n_done="     << tree.done().size() << '\n'
        << "t_final="    << tFinal     << '\n'
        << "n_orbits="   << kNOrbits   << '\n';
    return 0;
}
