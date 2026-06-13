// include/tax/ads/refine.hpp
//
// RefineDriver / refine — a "propagate-then-assess" alternative to the
// classic in-flight ADS driver.
//
// The classic AdsDriver splits a box the moment a flow map stops
// converging, then resumes the integration on the two halves from the
// split time. Refinement here is structured the other way round: every box
// is *always* propagated all the way to the final time first, and only then
// is its quality assessed by bisecting it, propagating both halves to the
// final time as well, and comparing (see refine_criteria.hpp). If the split
// changes the answer the children are kept and the same test recurses on
// them; otherwise the parent is accepted.
//
// Because each box is carried to t1 independently of every other, the work
// is embarrassingly parallel: a box and its eventual descendants never need
// each other's partial state, so the whole recursion fans out across a
// thread pool. The classic driver, by contrast, serialises a box against
// its own children through the split time.
//
//   auto tree = tax::ads::refine<6>(
//       tax::ode::methods::Verner89{},
//       tax::ads::CoefficientMatchCriterion{ /*tol=*/1e-6, /*maxDepth=*/7 },
//       rhs, ic_box, ic_center, t0, t1, cfg, n_threads );
//
// The result is the same AdsTree<State, M, T> the classic driver returns:
// tree.done() lists the accepted leaves, each carrying the flow map valid
// on its sub-box.

#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/refine_criteria.hpp>
#include <tax/ads/tree.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/integrator.hpp>
#include <tax/ode/propagate.hpp>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace tax::ads
{

template < class Stepper, class Quality >
class RefineDriver
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    using Cfg = typename Stepper::Config;

    using TE = typename State::Scalar;
    static constexpr int N = TE::order_v;
    static constexpr int M = TE::vars_v;
    static constexpr int D = State::RowsAtCompileTime;

    using Tree = AdsTree< State, M, T >;
    using BoxT = Box< T, M >;

    RefineDriver( Quality quality, Cfg cfg, int num_threads = 1 )
        : quality_( std::move( quality ) ),
          cfg_( std::move( cfg ) ),
          num_threads_( num_threads < 1 ? 1 : num_threads )
    {
    }

    template < class F >
    [[nodiscard]] Tree run( F&& rhs, const BoxT& ic_box, const Eigen::Matrix< T, D, 1 >& ic_center,
                            T t0, T t1 )
    {
        Tree tree;
        State root_init = tax::ads::create< N, M >( ic_box, ic_center );
        const int rootIdx = tree.init( ic_box, State{}, t0 );

        // The root is the only box not propagated as someone's child, so
        // propagate it up front and seed the queue with its flow map.
        State root_flow = propagateLeaf( rhs, root_init, t0, t1 );

        std::deque< WorkItem > seed;
        seed.push_back(
            WorkItem{ std::move( root_init ), ic_box, 0, rootIdx, std::move( root_flow ) } );

        drive( rhs, tree, std::move( seed ), t0, t1 );

        tree.canonicalizeDone();
        return tree;
    }

   protected:
    // One box awaiting assessment: its identity (t0) DA state, its box, its
    // depth, the arena index of its (active) leaf, and its already-computed
    // flow map at t1.
    struct WorkItem
    {
        State init{};
        BoxT box{};
        int depth = 0;
        int treeIdx = -1;
        State flow{};
    };

    // Outcome of assessing one box, computed lock-free.
    struct Verdict
    {
        bool split = false;
        int dim = -1;
        T splitValue{};
        State initL{}, initR{};
        State flowL{}, flowR{};
    };

    template < class F >
    [[nodiscard]] State propagateLeaf( const F& rhs, const State& payload, T t0, T t1 ) const
    {
        tax::ode::Integrator< Stepper, std::decay_t< F > > integ{ rhs, cfg_, {} };
        auto sol = integ.integrate( payload, t0, t1 );
        return std::move( sol.x.back() );
    }

    // Pure, lock-free: bisect the box, propagate both halves to t1, and ask
    // the criterion whether the split is worth keeping.
    template < class F >
    [[nodiscard]] Verdict assess( const F& rhs, const WorkItem& it, T t0, T t1 ) const
    {
        Verdict v;
        const int dim = quality_.splitDim( it.flow );
        auto pr = tax::ads::split( it.init, it.box, dim );
        State initL = std::move( pr.first );
        State initR = std::move( pr.second );
        State flowL = propagateLeaf( rhs, initL, t0, t1 );
        State flowR = propagateLeaf( rhs, initR, t0, t1 );

        if ( quality_.acceptable( it.flow, flowL, flowR, it.depth ) ) return v;  // split = false

        v.split = true;
        v.dim = dim;
        v.splitValue = it.box.center( dim );
        v.initL = std::move( initL );
        v.initR = std::move( initR );
        v.flowL = std::move( flowL );
        v.flowR = std::move( flowR );
        return v;
    }

    // num_threads_ workers pull WorkItems from a shared queue. The expensive
    // assess() (two propagations) runs lock-free on copied-out inputs; the
    // mutex guards only the queue and the tree mutation plus the in-flight
    // counter. Termination: queue empty AND nothing in flight. The first
    // worker exception wins and is rethrown on the calling thread.
    template < class F >
    void drive( const F& rhs, Tree& tree, std::deque< WorkItem > queue, T t0, T t1 )
    {
        std::mutex mtx;
        std::condition_variable cv;
        int in_flight = 0;
        bool stopping = false;
        std::exception_ptr first_err = nullptr;

        auto worker = [&]() {
            for ( ;; )
            {
                std::unique_lock< std::mutex > lk( mtx );
                cv.wait( lk, [&] { return stopping || !queue.empty() || in_flight == 0; } );

                if ( stopping ) return;
                if ( queue.empty() )
                {
                    if ( in_flight == 0 )
                    {
                        cv.notify_all();
                        return;
                    }
                    continue;
                }

                WorkItem it = std::move( queue.front() );
                queue.pop_front();
                ++in_flight;
                lk.unlock();

                Verdict v;
                const bool capped = it.depth >= quality_.maxDepth;
                try
                {
                    if ( !capped ) v = assess( rhs, it, t0, t1 );
                } catch ( ... )
                {
                    lk.lock();
                    if ( !first_err ) first_err = std::current_exception();
                    stopping = true;
                    --in_flight;
                    cv.notify_all();
                    return;
                }

                lk.lock();
                if ( v.split )
                {
                    auto pr = tree.split( it.treeIdx, v.dim, v.splitValue, State{}, State{}, t0 );
                    auto boxes = it.box.split( v.dim );
                    queue.push_back( WorkItem{ std::move( v.initL ), boxes.first, it.depth + 1,
                                               pr.first, std::move( v.flowL ) } );
                    queue.push_back( WorkItem{ std::move( v.initR ), boxes.second, it.depth + 1,
                                               pr.second, std::move( v.flowR ) } );
                } else
                {
                    tree.leaf( it.treeIdx ).payload = std::move( it.flow );
                    tree.finalize( it.treeIdx );
                }
                --in_flight;
                cv.notify_all();
            }
        };

        std::vector< std::thread > pool;
        pool.reserve( static_cast< std::size_t >( num_threads_ ) );
        for ( int i = 0; i < num_threads_; ++i ) pool.emplace_back( worker );
        for ( auto& th : pool ) th.join();

        if ( first_err ) std::rethrow_exception( first_err );
    }

    Quality quality_;
    Cfg cfg_;
    int num_threads_ = 1;
};

// Function-form entry point, mirroring tax::ads::propagate. P is the DA
// truncation order; M (box dim) and D (state dim) are deduced. The method
// tag selects the stepper; the quality criterion drives refinement.
template < int P, class Method, class Quality, class F, class T, int M, int D >
[[nodiscard]] auto refine( Method, Quality quality, F&& rhs, const Box< T, M >& ic_box,
                           const Eigen::Matrix< T, D, 1 >& ic_center, const T& t0, const T& t1,
                           tax::ode::IntegratorConfig< T > cfg = {}, int num_threads = 1 )
{
    using TE = tax::TaylorExpansion< T, P, M, tax::storage::Dense >;
    using DAState = Eigen::Matrix< TE, D, 1 >;
    using Stepper = tax::ode::detail::StepperT< Method, DAState >;

    RefineDriver< Stepper, Quality > driver{ std::move( quality ), std::move( cfg ), num_threads };
    return driver.run( std::forward< F >( rhs ), ic_box, ic_center, t0, t1 );
}

}  // namespace tax::ads
