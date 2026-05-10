#pragma once

/**
 * @file
 * @brief Umbrella include for the Taylor ODE integration module.
 *
 * Public class API:
 *  - tax::ode::Integrator<N>      — adaptive scalar/vector Taylor integration
 *  - tax::ode::DaIntegrator<N,P,D> — DA flow expansion (no splitting)
 *  - tax::ode::AdsIntegrator<N,P,D> — DA flow with Automatic Domain Splitting
 */

#include <tax/ode/stepsize.hpp>
#include <tax/ode/step.hpp>
#include <tax/ode/solution.hpp>
#include <tax/ode/events.hpp>
#include <tax/ode/integrator.hpp>
#include <tax/ode/da_integrator.hpp>
#include <tax/ode/ads_integrator.hpp>
