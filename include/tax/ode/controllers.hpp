// include/tax/ode/controllers.hpp
//
// Step-size controllers for Stage 2a.
//   I       — classic integral (stateless, robust baseline)
//   PI      — Gustafsson PI (default for RK steppers)
//   H211b   — Söderlind digital filter (smoothed; robust on bumpy problems)
//   JorbaZou — Taylor-method predictor based on the last two polynomial
//              coefficient magnitudes (default for TaylorStepper)

#pragma once

#include <algorithm>
#include <cmath>

namespace tax::ode::controllers
{

namespace detail
{
template < class T >
constexpr T clamp_factor( T raw, T mn, T mx ) noexcept
{
    return std::min( std::max( raw, mn ), mx );
}
}  // namespace detail

// -------- I --------
template < class T = double >
struct I
{
    T safety     = T{ 0.9 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) const noexcept
    {
        using std::pow;
        const T ratio  = ( err_norm > T{ 0 } ) ? ( tol / err_norm ) : T{ 1 };
        const T exp    = T{ 1 } / T( p_emb + 1 );
        const T factor = detail::clamp_factor< T >( safety * pow( ratio, exp ),
                                                     min_factor, max_factor );
        return h_used * factor;
    }
};

// -------- PI (Gustafsson) --------
template < class T = double >
struct PI
{
    T safety     = T{ 0.9 };
    T alpha      = T{ 0.7 };
    T beta       = T{ 0.4 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) noexcept
    {
        using std::pow;
        const T denom = ( err_norm > T{ 0 } ) ? err_norm : T{ 1 };
        const T inv   = ( p_emb > 0 ) ? T{ 1 } / T( p_emb + 1 ) : T{ 1 };
        const T raw   = pow( tol / denom, beta * inv )
                        * pow( err_prev_ / denom, alpha * inv );
        const T factor = detail::clamp_factor< T >( safety * raw,
                                                     min_factor, max_factor );
        err_prev_ = denom;
        return h_used * factor;
    }

private:
    T err_prev_ = T{ 1 };
};

// -------- H211b (Söderlind) --------
// Reference: G. Söderlind, "Digital filters in adaptive time-stepping",
// ACM TOMS 29 (2003). The H211b filter uses b ≈ 4 by default.
template < class T = double >
struct H211b
{
    T safety     = T{ 0.9 };
    T b          = T{ 4 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) noexcept
    {
        using std::pow;
        const T denom = ( err_norm > T{ 0 } ) ? err_norm : T{ 1 };
        const T inv   = ( p_emb > 0 ) ? T{ 1 } / T( p_emb + 1 ) : T{ 1 };

        if ( h_prev_ <= T{ 0 } )
        {
            // First call — behave as I-controller.
            const T factor = detail::clamp_factor< T >(
                safety * pow( tol / denom, inv ), min_factor, max_factor );
            err_prev_ = denom;
            h_prev_   = h_used;
            return h_used * factor;
        }

        const T t1 = pow( tol / denom, T{ 1 } / ( b * inv > T{ 0 } ? b : T{ 1 } ) );
        const T t2 = pow( tol / err_prev_, T{ 1 } / ( b * inv > T{ 0 } ? b : T{ 1 } ) );
        const T t3 = pow( h_used / h_prev_, T{ -1 } / b );

        const T raw    = t1 * t2 * t3;
        const T factor = detail::clamp_factor< T >( safety * raw,
                                                     min_factor, max_factor );
        err_prev_ = denom;
        h_prev_   = h_used;
        return h_used * factor;
    }

private:
    T err_prev_ = T{ 1 };
    T h_prev_   = T{ 0 };
};

// -------- JorbaZou (Taylor-specific) --------
// Reference: À. Jorba & M. Zou, "A software package for the numerical
// integration of ODE by means of high-order Taylor methods",
// Experimental Mathematics 14 (2005). Variant that uses the magnitudes
// of the last two polynomial coefficients to predict h_new directly.
template < class T = double >
struct JorbaZou
{
    T safety     = T{ 0.9 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T c_N_norm, T c_Nm1_norm, T tol,
                                int N_order ) const noexcept
    {
        using std::pow;
        // rho1 = (tol / |c_N|)^(1/N)
        // rho2 = (tol / |c_{N-1}|)^(1/(N-1))
        // h_new = safety * min(rho1, rho2)
        const T denom1 = ( c_N_norm > T{ 0 } ) ? c_N_norm : T{ 1 };
        const T denom2 = ( c_Nm1_norm > T{ 0 } ) ? c_Nm1_norm : T{ 1 };
        const T rho1 = pow( tol / denom1, T{ 1 } / T( N_order ) );
        const T rho2 = pow( tol / denom2, T{ 1 } / T( N_order - 1 ) );
        const T raw  = safety * std::min( rho1, rho2 );
        // Clamp to a factor of h_used so we don't accept arbitrary jumps.
        const T factor = detail::clamp_factor< T >( raw / h_used,
                                                     min_factor, max_factor );
        return h_used * factor;
    }
};

}  // namespace tax::ode::controllers
