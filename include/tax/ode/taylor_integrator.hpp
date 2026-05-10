#pragma once

/**
 * @file
 * @brief Umbrella include for the Taylor ODE integration module.
 *
 * Public class API:
 *  - tax::ode::Integrator<N>                  — adaptive scalar/vector Taylor integration
 *  - tax::ode::IntegratorP<N,State,Params>    — same, but RHS takes an extra parameter
 *                                                argument `p`  (f(x,p,t) / f(dx,x,p,t))
 *  - tax::ode::DaIntegrator<N,P,D,Q=0>        — DA flow expansion (no splitting);
 *                                                Q > 0 expands jointly w.r.t. Q parameters
 *  - tax::ode::AdsIntegrator<N,P,D,Q=0>       — DA flow with truncation-error ADS;
 *                                                Q > 0 splits across IC + parameter axes
 *  - tax::ode::LowOrderAdsIntegrator<N,P,D,Q=0> — DA flow with NLI-driven ADS
 *                                                  (Losacco et al., arXiv:2303.05791);
 *                                                  Q > 0 splits across IC + parameter axes
 */

#include <tax/ode/stepsize.hpp>
#include <tax/ode/step.hpp>
#include <tax/ode/solution.hpp>
#include <tax/ode/events.hpp>
#include <tax/ode/integrator.hpp>
#include <tax/ode/da_integrator.hpp>
#include <tax/ode/ads_integrator.hpp>
#include <tax/ode/low_order_ads_integrator.hpp>
