// include/tax/ads/driver.hpp
//
// AdsDriver<Stepper, Criterion> — BFS driver around tax::ode::Integrator.
// For each leaf in the work queue, the driver runs the integrator with
// a (SplitTrigger, SplitAction) pair appended to any user-supplied
// events. If the split event fires, the leaf is replaced by two
// children with the parent's DA state re-identified on each half via
// tax::ads::split. Otherwise the leaf is marked done with the
// propagated DA flow map stored as its payload.

#pragma once

#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/ads/tree.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/integrator.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace tax::ads
{

template < class Stepper, class Criterion >
class AdsDriver
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    using Cfg = typename Stepper::Config;
    using ExtraEvt = std::vector< tax::ode::Event< Stepper > >;

    using TE = typename State::Scalar;
    static constexpr int N = TE::order_v;
    static constexpr int M = TE::vars_v;
    static constexpr int D = State::RowsAtCompileTime;

    using Tree = AdsTree< State, M, T >;
    using BoxT = Box< T, M >;

    AdsDriver( Criterion crit, Cfg cfg, ExtraEvt extras = {} )
        : crit_( std::move( crit ) ), cfg_( std::move( cfg ) ), extras_( std::move( extras ) )
    {
    }

    template < class F >
    [[nodiscard]] Tree run( F&& rhs, const BoxT& ic_box, const Eigen::Matrix< T, D, 1 >& ic_center,
                            T t0, T t1 )
    {
        Tree tree;
        State root_state = tax::ads::create< N, M >( ic_box, ic_center );
        (void)tree.init( ic_box, std::move( root_state ), t0 );

        while ( !tree.empty() )
        {
            const int idx = tree.popFront();
            auto& l = tree.leaf( idx );

            SplitRequest< T > req;
            auto events = extras_;  // copy
            events.emplace_back( SplitTrigger( crit_, l.depth ), SplitAction( crit_, &req ) );

            tax::ode::Integrator< Stepper, std::decay_t< F > > integ{ rhs, cfg_,
                                                                      std::move( events ) };
            auto sol = integ.integrate( l.payload, l.tEntry, t1 );

            if ( req.fired )
            {
                const T splitValue = l.box.center[static_cast< std::size_t >( req.dim )];
                auto pr_state = tax::ads::split( sol.x.back(), l.box, req.dim );
                (void)tree.split( idx, req.dim, splitValue, std::move( pr_state.first ),
                                  std::move( pr_state.second ), req.t );
            } else
            {
                l.payload = std::move( sol.x.back() );
                tree.finalize( idx );
            }
        }
        return tree;
    }

   private:
    Criterion crit_;
    Cfg cfg_;
    ExtraEvt extras_;
};

}  // namespace tax::ads
