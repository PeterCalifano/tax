// examples/two_body/simple_taylor.cpp
//
// Scalar Taylor-method integration of the planar Kepler problem with
// dense output sampled at discrete times.

#include <chrono>
#include <fstream>
#include <iostream>

#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

#include "common.hpp"

int main()
{
    using namespace example::two_body;
    using namespace tax::ode::methods;
    using State = tax::la::VecNT< 4, double >;

    constexpr int    kNOrbits = 3;
    constexpr int    kNSnaps  = 200;
    const     double tFinal   = kNOrbits * kPeriod;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    const auto t0  = std::chrono::high_resolution_clock::now();
    auto sol = tax::ode::propagate< /*Dense=*/true >(
        Taylor< 16 >{}, rhs(), icCenter(), 0.0, tFinal, cfg );
    const auto t1  = std::chrono::high_resolution_clock::now();

    const double elapsed_ms = std::chrono::duration< double, std::milli >( t1 - t0 ).count();
    const std::size_t n_steps = sol.size() - 1;

    tax::ode::writeCsv( sol, tax::ode::linspace( 0.0, tFinal, kNSnaps + 1 ),
                        "simple_taylor_traj.csv" );

    std::cout << "[simple_taylor] " << elapsed_ms << " ms, " << n_steps << " steps\n";
    std::ofstream( "simple_taylor_timing.txt" )
        << "method=simple_taylor\n"
        << "elapsed_ms=" << elapsed_ms << '\n'
        << "n_steps="    << n_steps    << '\n'
        << "t_final="    << tFinal     << '\n'
        << "n_orbits="   << kNOrbits   << '\n';
    return 0;
}
