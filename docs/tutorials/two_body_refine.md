# Parallel ADS by refinement

The [two-body tutorial](two_body.md) showed *classic* Automatic Domain
Splitting: the integrator watches the flow polynomial as it advances and, the
moment the expansion stops converging, **splits the box in flight** and
resumes each half from the split time. It is accurate and frugal, but it is
also *sequential in time* — a box cannot be subdivided until the integration
has reached the point where it goes bad, and each child inherits its parent's
partial state.

This tutorial takes the opposite tack, implemented in
[`tax::ads::refine`](https://github.com/andreapasquale94/tax/tree/main/include/tax/ads/refine.hpp):

> **Always propagate the whole box to the final time first. Only then judge
> its quality — and if it is poor, split the *initial conditions* and try
> again.**

Because every box is carried to `t_final` on its own, with no dependence on
any other box's partial state, the entire refinement is **embarrassingly
parallel**: a box and all of its eventual descendants are independent
propagations that fan out across a thread pool.

Source: [`examples/two_body/refine.cpp`](https://github.com/andreapasquale94/tax/tree/main/examples/two_body/refine.cpp)
and [`plot_refine.py`](https://github.com/andreapasquale94/tax/tree/main/examples/two_body/plot_refine.py).

## The idea

Start from one box of initial conditions and propagate it to `t_final`,
yielding a single flow polynomial \(\Phi\). Now ask a sharp question:

> *Does splitting this box change the answer?*

To answer it, bisect the box along some direction, propagate **both halves**
all the way to `t_final` as well, and compare the two children
\(\Phi_L, \Phi_R\) against the parent \(\Phi\). If they agree, the parent was
already faithful — keep it. If they disagree, the parent had drifted past its
radius of convergence — discard it, keep the children, and recurse the same
test on each. Repeat until every surviving box passes, or a maximum depth is
reached.

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>
using namespace tax::ode::methods;

tax::ads::Box< double, 2 > ic_box{ center, half_width };  // box of initial (x, y)

auto tree = tax::ads::refine< /*P=*/6 >(
    Verner89{},
    tax::ads::CoefficientMatchCriterion{ /*tol=*/2e-3, /*maxDepth=*/8 },
    rhs, ic_box, ic_center, /*t0=*/0.0, /*t1=*/2 * M_PI, cfg, /*n_threads=*/8 );

for ( int li : tree.done() )
{
    const auto& leaf = tree.leaf( li );   // leaf.box, leaf.depth, leaf.payload
}
```

The return type is the same `AdsTree` the classic driver produces, so
everything downstream (point lookup, the merger, the I/O helpers) works
unchanged. The difference is purely *how* the tree was grown.

### Refinement vs. classic ADS

| | Classic ADS (`propagate`) | Refinement (`refine`) |
|---|---|---|
| When a box is split | mid-integration, at the failure time | after a full propagation to `t_final` |
| Child initial state | parent's partial map at the split time | fresh identity on the child sub-box |
| Quality probe | one flow map, inspected in flight | parent **vs. its two children** at `t_final` |
| Cost per box | one integration | three (self + two trial children) |
| Parallelism | independent boxes only | the **whole recursion** fans out |

Refinement deliberately spends more arithmetic — every box pays for two trial
children even if it is ultimately accepted — to buy a propagation pattern with
no time-ordering constraints at all.

## What quality index?

The verdict hinges on the comparison "parent vs. children". Two indices are
shipped in
[`refine_criteria.hpp`](https://github.com/andreapasquale94/tax/tree/main/include/tax/ads/refine_criteria.hpp);
both were tried on this problem.

### Area ratio (the motivating idea)

The most intuitive measure is *geometric*. Trace the image of the box boundary
under the flow map in two output components and measure its area by the
shoelace formula. When the parent polynomial is well shaped its two children
tile it exactly, so

$$
\rho \;=\; \frac{A(\Phi)}{A(\Phi_L) + A(\Phi_R)} \;\approx\; 1 .
$$

When the parent has diverged, its boundary balloons or folds and the shoelace
area inflates wildly — \(\rho\) departs from 1 (often \(\gg 1\)). That is a
very loud signal: for the single box on this orbit the parent "area" at
\(t = 2\pi\) comes out as \(\sim 10^{8}\) against a true set area of \(\approx
2\), so \(\rho \sim 10^{8}\). `AreaRatioCriterion` accepts a box when
\(|\rho - 1| \le \texttt{tol}\).

The catch is that area is a *global, shape-blind* measure. An
**area-preserving fold** — where the polynomial map turns the box inside-out
without changing the enclosed area — passes the test even though it is
pointwise wrong. In the animation below that shows up, with the area criterion,
as a transient spurious sliver mid-orbit on a box that nonetheless looks fine at
the endpoint. Great as a *divergence alarm*, weaker as a *stopping rule*.

### Coefficient match (the reliable one)

A dimension-free alternative compares the maps directly in coefficient space.
Re-identify the parent on a half-domain — the very same substitution ADS uses
to split, \(\xi_d \to \pm\tfrac12 + \tfrac12\,\xi'_d\) — and compare it term by
term to the independently propagated child:

$$
\delta \;=\; \max_i \;
  \frac{\big\| \Phi^{(i)}\!\restriction_{\text{half}} - \Phi^{(i)}_{\text{child}} \big\|_\infty}
       {\big\| \Phi^{(i)}_{\text{child}} \big\|_\infty} .
$$

While the parent is accurate the restriction reproduces the child and
\(\delta \approx 0\); once it drifts, \(\delta\) grows. Being a coefficient
norm it controls the polynomial *shape*, not just an integrated area, so it has
no fold blind spot. `CoefficientMatchCriterion` accepts when \(\delta \le
\texttt{tol}\), and it is what the example uses. The split *direction* for both
criteria is the coordinate carrying the most order-\(P\) coefficient mass — the
same heuristic as Wittig's truncation criterion.

## Watching it converge

The example sweeps the depth cap \(k = 0, 1, 2, \dots\): iteration 0 is the
single box, and each iteration adds a level of refinement until the partition
stops changing. At every iteration we push every sub-box to `t_final`, draw the
box images along the orbit, and score the piecewise-polynomial prediction
against a 350-point **Monte-Carlo** reference cloud.

![Iterative ADS refinement converging onto the Monte-Carlo set](img/two_body_refine.gif)

Iteration 0 — the lone box — detonates: by one full period the order-6
polynomial is extrapolating far past where it converges, its image shoots off
the frame, and the RMS error against Monte Carlo is \(\sim\! 2\times10^{4}\).
Two boxes already tame it to \(\sim\! 0.1\); each further level concentrates new
splits where the orbit is most nonlinear (the periapsis re-passage, drawn in the
deeper/yellow boxes) and the partition closes onto the true banana-shaped set.

The matching improves monotonically with the number of sub-boxes — exactly the
behaviour we want from a refinement scheme:

![RMS error against Monte Carlo vs. number of sub-boxes](img/two_body_refine_convergence.png)

| iteration | sub-boxes | RMS vs. Monte Carlo |
|----------:|----------:|--------------------:|
| 0 | 1  | \(2.0\times10^{4}\) |
| 1 | 2  | \(9.4\times10^{-2}\) |
| 2 | 4  | \(3.7\times10^{-2}\) |
| 3 | 7  | \(1.5\times10^{-2}\) |
| 4 | 11 | \(5.4\times10^{-3}\) |
| 5 | 17 | \(1.4\times10^{-3}\) |
| 6 | 23 | \(2.4\times10^{-4}\) |

## Parallelism

Each box is propagated independently of every other, so `refine` runs the
recursion across `num_threads` workers pulling from a shared queue: the
expensive three propagations happen lock-free on copied-out inputs, and the
mutex guards only the queue and the tree mutation. Because the accept/split
decision for a box depends solely on that box and its two trial children — never
on global ordering — the resulting partition is **identical** whether run on one
thread or many (leaves are canonicalised by box center), which the test suite
checks directly.

## Run it yourself

```bash
cmake -S . -B build -DTAX_BUILD_EXAMPLES=ON && cmake --build build -j
cd build/examples
./two_body_refine
python3 ../../examples/two_body/plot_refine.py --data . --out figs
```

Things to try:

- **Switch the index.** Drop `AreaRatioCriterion{ 0.01, k, 0, 1, 20 }` into the
  `crit` line and watch the area-blind fold appear mid-orbit while the endpoint
  stays accurate.
- **Grow the box** (`kHx`, `kHy`) and see the single polynomial fail even harder
  and the converged leaf count climb.
- **Set `TAX_ADS_THREADS`** and confirm the partition (and every coefficient) is
  bit-for-bit independent of the worker count.
