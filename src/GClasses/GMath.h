/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#ifndef __GMATH_H__
#define __GMATH_H__

#include <math.h>
#include <cstring>

#ifdef WINDOWS
#ifndef M_PI
# define M_E            2.7182818284590452354   /* e */
# define M_LOG2E        1.4426950408889634074   /* log_2 e = 1/(log_e 2)*/
# define M_LOG10E       0.43429448190325182765  /* log_10 e = 1/(log_e 10)*/
# define M_LN2          0.69314718055994530942  /* log_e 2 */
# define M_LN10         2.30258509299404568402  /* log_e 10 */
# define M_PI           3.14159265358979323846  /* pi */
# define M_PI_2         1.57079632679489661923  /* pi/2 */
# define M_PI_4         0.78539816339744830962  /* pi/4 */
# define M_1_PI         0.31830988618379067154  /* 1/pi */
# define M_2_PI         0.63661977236758134308  /* 2/pi */
# define M_2_SQRTPI     1.12837916709551257390  /* 2/sqrt(pi) */
# define M_SQRT2        1.41421356237309504880  /* sqrt(2) */
# define M_SQRT1_2      0.70710678118654752440  /* 1/sqrt(2) = sqrt(1/2) */
#endif
#endif


namespace GClasses {


typedef double (*MathFunc)(void* pThis, double x);

/// Provides some useful math functions
class GMath
{
public:
	/// Returns sign(x) * sqrt(ABS(x))
	inline static double signedRoot(double x)
	{

		if(x >= 0)
			return sqrt(x);
		else
			return -sqrt(-x);
	}

	/// The logistic sigmoid function.
	inline static double logistic(double x)
	{
		if(x >= 500.0) // Don't trigger a floating point exception
			return 1.0;
		if(x < -500.0) // Don't trigger a floating point exception
			return 0.0;
		return 1.0 / (exp(-x) + 1.0);
	}

	/// This evaluates the derivative of the sigmoid function
	inline static double logisticDerivative(double x)
	{
		double d = logistic(x);
		return d * (1.0 - d);
	}

	/// This is the inverse of the logistic sigmoid function
	inline static double logisticInverse(double y)
	{
		// return (log(y) - log(1.0 - y));
		return -log((1.0 / y) - 1.0);
	}

	/// Calculates a function that always passes through (0, 0),
	/// (1, 1), and (0.5, 0.5). The slope at (0.5, 0.5) will be
	/// "steepness". If steepness is > 1, then the slope at
	/// (0, 0) and (1, 1) will be 0. If steepness is < 1, the
	/// slope at (0, 0) and (1, 1) will be infinity. If steepness
	/// is exactly 1, the slope will be 1 at those points.
	/// softStep(1/x, 2) = PI*cauchy(x-1).
	inline static double softStep(double x, double steepness)
	{
		return 1.0 / (pow(1.0 / x - 1.0, steepness) + 1.0);
	}

	/// A soft step function with a very high degree of continuity at 0 and 1.
	inline static double interpolatingFunc(double x)
	{
		double a = pow(x, 1.0 / x);
		double b = pow(1.0 - x, 1.0 / (1.0 - x));
		return a / (a + b);
	}

	/// The bend function has a slope of 1 at very negative values of x, and 2 at
	/// very positive values of x, and a smooth transition in between.
	inline static double bend(double x)
	{
		if(x >= 500.0) // Don't trigger a floating point exception
			return x + x;
		if(x < -500.0) // Don't trigger a floating point exception
			return x;
		return x + log(exp(x) + 1.0);
	}

	/// The inverse of the bend function
	inline static double bendInverse(double y)
	{
		if(y >= 1000.0)
			return 0.5 * y;
		if(y < -500.0)
			return y;
		return log(0.5 * (sqrt(4.0 * exp(y) + 1.0) - 1.0));
	}

	/// The derivative of the bend function
	inline static double bendDerivative(double x)
	{
		return logistic(x) + 1.0;
	}

	/// The gamma function
	static double gamma(double x);

	/// returns log(Gamma(x))
	static double logGamma(double x);

	/// returns log(x!)
	static double logFactorial(int x);

	/// The gaussian function
	inline static double gaussian(double x)
	{
		return exp(-0.5 * (x * x));
	}

	/// Returns an approximation for the error function of x
	static double approximateErf(double x);

	/// Returns an approximation for the inverse of the error function
	static double approximateInverseErf(double x);

	/// Computes the y where x = y*exp(y). This is also known as the
	/// Omega function, or the Lambert W function. x must be > -1/e.
	static double productLog(double x);

	/// This implements Newton's method for determining a
	/// polynomial f(t) that goes through all the control points
	/// pFuncValues at pTValues.  (You could then convert to a
	/// Bezier curve to get a Bezier curve that goes through those
	/// points.)  The polynomial coefficients are put in pFuncValues
	/// in the form c0 + c1*t + c2*t*t + c3*t*t*t + ...
	static void newtonPolynomial(const double* pTValues, double* pFuncValues, int nPoints);

	/// Integrates the specified function from dStart to dEnd
	static double integrate(MathFunc pFunc, double dStart, double dEnd, int nSteps, void* pThis);

	/// Estimates the Incomplete Beta function by "integrating" with the specified number of steps
	static double incompleteBeta(double x, double a, double b, int steps);

	/// Returns the specified row from Pascal's triangle. pOutRow must be big enough to hold nRow + 1 elements
	/// Row 0           1
	/// Row 1          1 1
	/// Row 2         1 2 1
	/// Row 3        1 3 3 1
	/// Row 4       1 4 6 4 1
	/// etc. such that each value is the sum of its two parents
	static void pascalsTriangle(size_t* pOutRow, size_t nRow);

	/// Returns the value of n choose k
	static double nChooseK(unsigned int n, unsigned int k)
	{
		double d = n--;
		unsigned int i;
		for(i = 2; i <= k; i++)
		{
			d *= n;
			n--;
			d /= i;
		}
		return d;
	}

	/// Computes the p-value from the degrees of freedom, and the
	/// t-value obtained from a T-test.
	static double tTestAlphaValue(size_t v, double t);

	/// This computes the Wilcoxon P-value assuming n is large
	/// enough that the Normal approximation will suffice.
	static double wilcoxonPValue(int n, double t);

	/// Inverts a function using linear interpolation to iteratively find the target output.
	/// Assumes the function is monotonic in the region it will explore
	/// func gives the function to invert. params specifies parameters to the function. y is the target output.
	/// x1 and x2 are two initial estimates to seed the search. The search is complete when the output
	/// is within epsilon of the target. Throws an exception if progress stalls.
	static double functionInverse(double (*func)(const double* params, double x), const double* params, double y, double x1, double x2, double epsilon);

	/// If alpha=1, returns exp(x). If alpha = 0, returns x. If alpha = -1, returns log_e(x).
	/// Alpha can be any continuous value from -1 to 1 to continuously interpolate among these functions.
	static double softExponential(double alpha, double x);

	/// Given x(t), returns x(t + 1) in the Mackey-Glass series.
	/// x is x(t), xt is x(t - TAO), and beta, gamma, and n are other Mackey-Glass parameters.
	/// Call this in a loop to get a series.
	static double mackeyGlass(double x, double xt, double beta = 0.2, double gamma = 0.1, double n = 10);

	/// If false = 0.0 and true = 1.0, then
	/// not(x) = -x = fuzzy(x, 0.0, 0),
	/// and(x, y) = fuzzy(x, y, 1.0),
	/// or(x, y) = -fuzzy(x, y, 0.0),
	/// xor(x, y) = -fuzzy(x, y, 0.5),
	/// nand(x, y) = -fuzzy(x, y, 1.0),
	/// nor(x, y) = fuzzy(x, y, 0.0),
	/// nxor(x, y) = fuzzy(x, y, 0.5).
	static double fuzzy(double x, double y, double alpha);

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
};


} // namespace GClasses

#endif // __GMATH_H__
