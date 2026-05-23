# Examples

Worked examples that exercise the core `TaylorExpansion` API. All snippets
assume `#include <tax/tax.hpp>`.

---

## Creating variables

### Univariate

```cpp
// Order-5 expansion of x around x₀ = 1.0
auto x = tax::TE<5>::variable(1.0);
// Coefficients (Dense): [1.0, 1.0, 0, 0, 0, 0]
//                         ↑    ↑
//                       value  δx coefficient

// A pure constant (no δx dependence)
auto c = tax::TE<5>::constant(3.14);
// Coefficients: [3.14, 0, 0, 0, 0, 0]
```

### Multivariate — per-coordinate factories

```cpp
using TE2 = tax::TE<3, 2>;
const std::array<double, 2> p{1.0, 2.0};

auto x = TE2::variable<0>(p);   // δx coordinate around (1, 2)
auto y = TE2::variable<1>(p);   // δy coordinate around (1, 2)
```

### Multivariate — Eigen vector form

```cpp
auto v = tax::variables<tax::TE<3, 2>>(Eigen::Vector2d{1.0, 2.0});
auto& x = v(0);
auto& y = v(1);
```

---

## Accessing coefficients and derivatives

### Univariate

```cpp
auto x = tax::TE<5>::variable(0.0);
tax::TE<5> f = tax::sin(x);

double v   = f.value();              // sin(0) = 0
double c1  = f.coeff({1});           // 1                (1st Taylor coefficient)
double c3  = f.coeff({3});           // -1/6
double d1  = f.derivative({1});      // cos(0)  = 1
double d3  = f.derivative({3});      // -cos(0) = -1     (= 3! · c3)
```

### Multivariate

```cpp
using TE2 = tax::TE<3, 2>;
const std::array<double, 2> p{1.0, 2.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 g = x*x*y;

double c_200 = g.coeff({2, 0});       // coefficient of δx²
double c_110 = g.coeff({1, 1});       // coefficient of δx·δy
double c_110_ct = g.coeff<1, 1>();    // compile-time variant
double d_110 = g.derivative<1, 1>();  // ∂²g/∂x∂y at (1, 2)
```

---

## Arithmetic and composition

The library uses expression-template flattening internally; the user just
writes the math.

```cpp
auto x = tax::TE<5>::variable(1.0);

tax::TE<5> f = (x + 2.0) * (x - 3.0);   // x² - x - 6 at x₀ = 1
tax::TE<5> g = x + x*x + x*x*x;         // chained sums
tax::TE<5> h = 1.0 / (1.0 + x);         // reciprocal recurrence
```

### Multivariate products

```cpp
using TE2 = tax::TE<3, 2>;
const std::array<double, 2> p{1.0, 2.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 f = x*x + 2.0*x*y + y*y;   // (x + y)² at (1, 2)
```

---

## Differentiation and integration

### Compile-time variable index

```cpp
using TE2 = tax::TE<4, 2>;
const std::array<double, 2> p{1.0, 2.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 f = x*x*y + y*y;

auto df_dx = f.deriv<0>();    // ∂f/∂x  = 2xy
auto df_dy = f.deriv<1>();    // ∂f/∂y  = x² + 2y

auto F_x   = f.integ<0>();    // ∫f dx
auto F_y   = f.integ<1>();    // ∫f dy
```

### Runtime variable index

```cpp
auto x = tax::TE<5>::variable(1.0);
tax::TE<5> f = tax::exp(x);

auto df = f.deriv(0);          // d/dx exp(x)
auto F  = f.integ(0);          // ∫ exp(x) dx
```

### Verifying an identity

```cpp
auto x = tax::TE<6>::variable(0.5);
tax::TE<6> f = tax::sin(x);

tax::TE<6> df       = f.deriv<0>();
tax::TE<6> expected = tax::cos(tax::TE<6>::variable(0.5));
// The coefficient arrays of df and expected agree to machine precision.
```

---

## Transcendental functions

All standard mathematical functions are propagated via degree-by-degree
recurrences in a single forward pass.

```cpp
auto x = tax::TE<8>::variable(0.0);

tax::TE<8> s  = tax::sin(x);     // [0, 1, 0, -1/6, 0, 1/120, ...]
tax::TE<8> c  = tax::cos(x);     // [1, 0, -1/2, 0, 1/24, 0, ...]

auto y = tax::TE<8>::variable(1.0);
tax::TE<8> e  = tax::exp(y);     // exp(1+δx) = e · [1, 1, 1/2, 1/6, ...]
tax::TE<8> l  = tax::log(y);     // log(1+δx) = [0, 1, -1/2, 1/3, ...]

tax::TE<8> sq = tax::sqrt(tax::TE<8>::variable(4.0));  // √(4+δx)
tax::TE<8> cb = tax::cbrt(tax::TE<8>::variable(8.0));  // ∛(8+δx)

tax::TE<10> h = tax::atan(x) / (1.0 + x*x);
```

---

## Polynomial evaluation

`eval()` Horner-evaluates the truncated Taylor polynomial at a displacement
$\delta x$ from the expansion point.

```cpp
auto x = tax::TE<15>::variable(0.0);
tax::TE<15> f = tax::sin(x);

double approx = f.eval({0.3});   // sin(0.3) within machine precision
```

### Multivariate evaluation

```cpp
using TE2 = tax::TE<5, 2>;
const std::array<double, 2> p{0.0, 0.0};
auto x = TE2::variable<0>(p);
auto y = TE2::variable<1>(p);

TE2 f = tax::sin(x) * tax::cos(y);
double approx = f.eval({0.3, 0.5});   // ≈ sin(0.3) * cos(0.5)
```

---

## Sparse storage drop-in

The factories on `STE<N, M>` mirror those on `TE<N, M>`; the storage layout
differs but the API is identical.

```cpp
using STE2 = tax::STE<5, 2>;
const std::array<double, 2> p{0.0, 0.0};
auto x = STE2::variable<0>(p);
auto y = STE2::variable<1>(p);

STE2 f = x*x + y*y;                  // only 2 nonzeros stored
auto nnz = f.nnz();                  // = 2

auto fd = f.dense();                 // → TaylorExpansion<…, Dense> conversion
```

See [Dense vs Sparse Storage](storage.md) for guidance on when each pays off.
