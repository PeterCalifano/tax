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

    AdsDriver( Criterion crit, Cfg cfg, ExtraEvt extras = {}, int num_threads = 1 )
        : crit_( std::move( crit ) ),
          cfg_( std::move( cfg ) ),
          extras_( std::move( extras ) ),
          num_threads_( num_threads < 1 ? 1 : num_threads )
    {
    }

    template < class F >
    [[nodiscard]] Tree run( F&& rhs, const BoxT& ic_box, const Eigen::Matrix< T, D, 1 >& ic_center,
                            T t0, T t1 )
    {
        Tree tree;
        State root_state = tax::ads::create< N, M >( ic_box, ic_center );
        (void)tree.init( ic_box, std::move( root_state ), t0 );

        driveSerial( rhs, tree, t1 );

        tree.canonicalizeDone();
        return tree;
    }

   protected:
    // Outcome of integrating one leaf: either split into two children or
    // finalize with a flow-map payload. Computed lock-free in stepLeaf.
    struct LeafVerdict
    {
        bool  split = false;
        int   dim = -1;
        T     splitTime{};
        T     splitValue{};
        State left{};
        State right{};
        State finalPayload{};
    };

    // Pure, lock-free: integrate one leaf from tEntry to t1 with the
    // split event appended, and decide split-vs-finalize. Reads only
    // crit_, cfg_, extras_ (all const) and the passed-in rhs / inputs.
    template < class F >
    [[nodiscard]] LeafVerdict stepLeaf( const F& rhs, const State& payload, T tEntry, int depth,
                                        const BoxT& box, T t1 ) const
    {
        SplitRequest< T > req;
        auto events = extras_;  // copy
        events.emplace_back( SplitTrigger( crit_, depth ), SplitAction( crit_, &req ) );

        tax::ode::Integrator< Stepper, std::decay_t< F > > integ{ rhs, cfg_,
                                                                  std::move( events ) };
        auto sol = integ.integrate( payload, tEntry, t1 );

        // Guard against a split fired at (or beyond) the final time —
        // splitting would queue two children with tEntry == t1, and the
        // integrator rejects empty intervals.
        const bool atFinal = req.fired && !( req.t < t1 );

        LeafVerdict v;
        if ( req.fired && !atFinal )
        {
            v.split      = true;
            v.dim        = req.dim;
            v.splitTime  = req.t;
            v.splitValue = box.center( req.dim );
            auto pr      = tax::ads::split( sol.x.back(), box, req.dim );
            v.left       = std::move( pr.first );
            v.right      = std::move( pr.second );
        }
        else
        {
            v.split        = false;
            v.finalPayload = std::move( sol.x.back() );
        }
        return v;
    }

    template < class F >
    void driveSerial( const F& rhs, Tree& tree, T t1 )
    {
        while ( !tree.empty() )
        {
            const int idx = tree.popFront();
            const auto& l = tree.leaf( idx );

            LeafVerdict v = stepLeaf( rhs, l.payload, l.tEntry, l.depth, l.box, t1 );

            if ( v.split )
            {
                (void)tree.split( idx, v.dim, v.splitValue, std::move( v.left ),
                                  std::move( v.right ), v.splitTime );
            }
            else
            {
                tree.leaf( idx ).payload = std::move( v.finalPayload );
                tree.finalize( idx );
            }
        }
    }

    Criterion crit_;
    Cfg cfg_;
    ExtraEvt extras_;
    int num_threads_ = 1;
};

}  // namespace tax::ads
