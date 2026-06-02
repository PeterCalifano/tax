// examples/two_body/simple_taylor.cpp
//
// Scalar Taylor-method integration of the planar Kepler problem with
// dense output sampled at discrete times.
//
// Output:
//   simple_taylor_traj.csv     — t, x0..x3 at snapshot times
//   simple_taylor_timing.txt   — wall-clock + step counts

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
    using State = tax::la::VecNT< 4, double >;

    constexpr int    kNOrbits = 3;
    constexpr int    kNSnaps  = 200;
    const     double tFinal   = kNOrbits * kPeriod;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Taylor< 16, State, tax::ode::controllers::JorbaZou< double >,
                      /*Dense=*/true, decltype( rhs() ) >
        integ{ rhs(), cfg };

    const auto t0  = std::chrono::high_resolution_clock::now();
    auto       sol = integ.integrate( icCenter(), 0.0, tFinal );
    const auto t1  = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration< double, std::milli >( t1 - t0 ).count();
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
