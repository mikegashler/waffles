/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html


	The Mersene Twister Generator is derived from the file mt19937-64.c
	by Takuji Nishimura and Makoto Matsumoto which contains the
	following header comment:

   A C-program for MT19937-64 (2004/9/29 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)  
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and 
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and 
     Computer Simulation 8. (Jan. 1998) 3--30.

   Any feedback is very welcome.
   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
*/

#ifndef __GRAND_H__
#define __GRAND_H__

#include <stdint.h>
#include <vector>

namespace GClasses {


/// This is a 64-bit pseudo-random number generator.
///
/// When subclassing it, overriding the next and setSeed methods will
/// be sufficient.  However, all of the methods are virtual, so you
/// can give them more efficient or accurate versions if you wish.
class GRand
{
protected:
	uint64_t m_a;
	uint64_t m_b;

public:
	/// Create a new random number generator with the given seed
	///
	/// \param seed the seed to use for generating numbers from the
	///        random number generator
	GRand(uint64_t seed);

	/// Destructor
	virtual ~GRand();

	/// Sets the seed
	///
	/// \note If you subclass this, make sure you call the superclass
	///       version.  Because the earlier code was not designed for
	///       subclassing, the GRand class is not abstract.  Thus, if
	///       someone has passed a GRand object of your new type by
	///       value, they will get a copy of the sub-object with default
	///       behavior.  To make their program work, the seed for the
	///       default generator needs to be set rather than being left
	///       uninitialized.
	virtual void setSeed(uint64_t seed);

	/// Returns an unsigned pseudo-random 64-bit value
	virtual uint64_t next()
	{
		m_a = 0x141F2B69ull * (m_a & 0x3ffffffffull) + (m_a >> 32);
		m_b = 0xC2785A6Bull * (m_b & 0x3ffffffffull) + (m_b >> 32);
		return m_a ^ m_b;
	}

	/// Returns a pseudo-random uint from a discrete uniform
	/// distribution in the range 0 to range-1 (inclusive).
	/// (This method guarantees the result will be drawn from
	/// a uniform distribution, whereas doing "next() % range"
	/// does not guarantee a truly uniform distribution.)
	///
	/// \param range one greater than the largest number that will be
	///        returned
	virtual uint64_t next(uint64_t range);

	/// Returns a random value from a beta distribution with parameters \a alpha and \a beta
	///
	/// (The following was pasted from wikipedia with a little editing)
	///
	/// The probability density function of the beta distribution is:
	/// 
	/// \f{align}{
	/// f(x;\alpha,\beta) & = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{\int_0^1 u^{\alpha-1} (1-u)^{\beta-1}\, du} \\[6pt]
	/// & = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\, x^{\alpha-1}(1-x)^{\beta-1} \\[6pt]
	/// & = \frac{1}{\mathrm{B}(\alpha,\beta)}\, x
	/// ^{\alpha-1}(1-x)^{\beta-1}
	///  \f}
	/// 
	/// where \f$\Gamma(z)$f$ is the gamma function.  The beta function,
	/// \b B, appears as a normalization constant to ensure that the
	/// total probability integrates to unity.
	/// 
	/// A random variable \b X that is Beta-distributed with shape \b α
	/// and \b β is denoted
	/// 
	/// \f[ X \sim \textrm{Be}(\alpha, \beta)\f]
	/// 
	/// 
	///
	/// \param alpha The \b α parameter (see above)
	///
	/// \param beta The \b β parameter (see above)
	virtual double beta(double alpha, double beta);

	/// Returns a random value from a binomial distribution
	/// This method draws n samples from a uniform distribution,
	/// so it is very slow for large values of n. binomial_approx
	/// is generally much faster.
	virtual std::size_t binomial(std::size_t n, double p);

	/// Returns a random value approximately from a binomial distribution.
	/// This method uses a normal distribution to approximate the binomial
	/// distribution. It is O(1), and is generally quite accurate when
	/// n is large and p is not too close to 0 or 1.
	virtual std::size_t binomial_approx(std::size_t n, double p);

	/// Returns a random value from a categorical distribution
	/// with the specified vector of category probabilities. (Note: If you need
	/// to draw many values from a categorical distribution, the GCategoricalSampler
	/// and GCategoricalSamplerBatch classes are designed to do this more efficiently.)
	virtual std::size_t categorical(std::vector<double>& probabilities);

	/// Returns a random value from a standard Cauchy distribution
	virtual double cauchy();

	/// Returns a random value from a chi-squared distribution
	virtual double chiSquare(double t);

	/// Returns a random vector from a dirichlet distribution with the
	/// specified parameters and dimensionality. (Both pOutVec and
	/// pParams should be an array of dims values.)
	virtual void dirichlet(double* pOutVec, const double* pParams, int dims);

	/// Returns a random value from a standard exponential distribution.
	/// (To convert it to a random value from an arbitrary exponential
	/// distribution, just divide the value this returns by the
	/// rate (usually lambda), or if you use the scale parameterization,
	/// just multiply the value this returns by the scale (usually beta)).
	virtual double exponential();

	/// Returns a random value from an f-distribution
	virtual double f(double t, double u);

	/// Returns a random value from a gamma distribution with beta=theta=1.
	/// To convert to a value from an arbitrary gamma distribution,
	/// just divide the value this returns by beta (or use alpha=k, and
	/// multiply the value this returns by theta).
	virtual double gamma(double alpha);

	/// \brief Returns a random value from a geometric distribution 
	/// with support for {0, 1, 2, ...}.
	///
	/// \param p the probability of success in the underlying 
	///          bernoulli trials.  0 < p < 1
	virtual int geometric(double p);

	/// Returns a random value from a standard logistic distribution. To
	/// convert to a random value from an arbitrary logistic distribution,
	/// just multiply the value this returns by the scale (or divide by
	/// the steepness), and then add the mean.
	virtual double logistic();

	/// Returns a random value from a log-Normal distribution
	virtual double logNormal(double mean, double dev);

	/// Returns a random value from a standard normal distribution. (To
	/// convert it to a random value from an arbitrary normal distribution,
	/// just multiply the value this returns by the deviation (usually
	/// lowercase-sigma), then add the mean (usually mu).)
	virtual double normal();

	/// Returns a random value from a Poisson distribution
	virtual int poisson(double mu);

	/// Draws uniformly from a unit simplex. (This is a special case of
	/// drawing from a dirichlet distribution with uniform parameters.)
	virtual void simplex(double* pOutVec, std::size_t dims);

	/// Returns a random value from a soft-impulse distribution with support
	/// from 0 to 1. (The cdf of the soft-impulse distribution is the soft-step
	/// function: (1/(pow(1/x-1,s)+1)). The mean is always at 0.5, where
	/// the probability-density is s.
	virtual double softImpulse(double s);

	/// Returns a random point on the surface of a dims-dimensional unit sphere
	virtual void spherical(double* pOutVec, std::size_t dims);

	/// Returns a random point within the volume of a dims-dimensional unit sphere
	virtual void spherical_volume(double* pOutVec, std::size_t dims);

	/// Returns a random point uniformly distributed within a unit cube
	virtual void cubical(double* pOutVec, std::size_t dims);

	/// Returns a random value from Student's t-distribution
	virtual double student(double t);

	/// Returns a pseudo-random double from 0 (inclusive)
	/// to 1 (exclusive). This uses 52 random bits for the
	/// mantissa, and discards the extra 12 random bits.
	virtual double uniform();

	/// Returns a pseudo-random double from \a min (inclusive)
	/// to \a max (exclusive).
	///
	/// \param min the minimum of the interval on which the samples
	/// returned are distributed
	///
	/// \param max the supremum of the interval on which the samples
	/// returned are distributed.  max will not be generated unless max
	/// == min or unless the interval is otherwise pathological
	virtual double uniform(double min, double max){
		return uniform()*(max-min)+min;
	}

	/// Returns a random value from a Weibull distribution with lambda=1.
	virtual double weibull(double gamma);

	/// \brief Returns a reference to the global random number generator.  
	///
	/// Useful as a default parameter to constructors so that only users
	/// who have particular needs will need to construct a new GRand
	/// object.
	///
	/// Initializes the seed based on the time and some other system
	/// parameters the first time it is called.
	///
	/// \returns a reference to the global random number generator.  
	static GRand& global();

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !MIN_PREDICT
};

/// \brief A 64-bit pseudo-random number generator based on the
/// Mersenne Twister
///
/// Remember: do not pass objects of this class by value as GRand
/// objects - the resulting objects will still work, but they will not
/// use the Mersene Twister random number generator
///
/// This class is derived from code that contained the following
/// copyright notice:
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
///   1. Redistributions of source code must retain the above
///      copyright notice, this list of conditions and the following
///      disclaimer.
///
///   2. Redistributions in binary form must reproduce the above
///      copyright notice, this list of conditions and the following
///      disclaimer in the documentation and/or other materials
///      provided with the distribution.
///
///   3. The names of its contributors may not be used to endorse or
///      promote products derived from this software without specific
///      prior written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
/// CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
/// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
/// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
/// DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
/// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
/// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
/// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
/// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
/// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
/// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
/// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
/// SUCH DAMAGE.
///
/// References:
/// T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
///   ACM Transactions on Modeling and 
///   Computer Simulation 10. (2000) 348--357.
/// M. Matsumoto and T. Nishimura,
///   ``Mersenne Twister: a 623-dimensionally equidistributed
///     uniform pseudorandom number generator''
///   ACM Transactions on Modeling and 
///   Computer Simulation 8. (Jan. 1998) 3--30.
///
/// Any feedback is very welcome.
/// http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
/// email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
///
class GRandMersenneTwister:public GRand
{
private:
	///Number of elements in the state-vector array
	const static uint64_t NN=312;

	const static uint64_t MM=156;
	const static uint64_t MATRIX_A=0xB5026F5AA96619E9ULL;
	const static uint64_t UM=0xFFFFFFFF80000000ULL; /* Most significant 33 bits */
	const static uint64_t LM=0x7FFFFFFFULL; /* Least significant 31 bits */

	/// The array for the state vector
	uint64_t mt[NN]; 
	
	unsigned int mti; 

	/// A state variable used by the number generation
	uint64_t mag01[2];

	/// initializes mt[NN] with a seed
	void init_genrand64(uint64_t seed);


	/* initialize by an array with array-length */
	/* init_key is the array for initializing keys */
	/* key_length is its length */
	//
	// Only used in the test code right now
	void init_by_array64(uint64_t init_key[],
											 uint64_t key_length);

	/// generates a random number on [0, 2^64-1]-interval
	uint64_t genrand64_int64(void);

public:

	/// Create a new random number generator with the given seed
	///
	/// The sequence of numbers generated from the same seed is identical
	///
	/// \param seed the seed to use for generating numbers from the
	///        random number generator
	GRandMersenneTwister(uint64_t seed):GRand(seed){
		mag01[0]=0ULL;
		mag01[1]=MATRIX_A;
		init_genrand64(seed);
	}

	/// Destructor
	virtual ~GRandMersenneTwister(){}

	/// Sets the seed
	///
	/// \note If you subclass this, make sure you call GRand::setSeed
	///       (see the comment there)
	///
	/// \param seed the seed to use for generating numbers from the
	///        random number generator
	virtual void setSeed(uint64_t seed){
		GRand::setSeed(seed);
		init_genrand64(seed);
	}

	///\brief Returns an unsigned pseudo-random 64-bit value
	///
	///\return an unsigned pseudo-random 64-bit value
	virtual uint64_t next()	{
		return genrand64_int64();
	}

	///\brief Returns a random number uniformly distributed on the [0,1)
	///interval.  This interval includes 0, but does not include 1.
	///
	/// This overrides the superclass treatment of the underlying
	/// integer stream to use the behavior of genrand64_real2 in the
	/// Mersenne twister code
	///
	///\return a random number uniformly distributed on the [0,1)
	///interval.  This interval includes 0, but does not include 1.
	virtual double uniform(){
		return (next() >> 11) * (1.0/9007199254740992.0);
	}

	/// Returns a pseudo-random double from \a min (inclusive)
	/// to \a max (exclusive).
	///
	/// \param min the minimum of the interval on which the samples
	/// returned are distributed
	///
	/// \param max the supremum of the interval on which the samples
	/// returned are distributed.  max will not be generated unless max
	/// == min or unless the interval is otherwise pathological
	virtual double uniform(double min, double max){
		return uniform()*(max-min)+min;
	}

#ifndef MIN_PREDICT
	/// Performs unit tests for this class. Throws an exception if there
	/// is a failure.
	static void test();
#endif // !MIN_PREDICT
};



} // namespace GClasses

#endif // __GRAND_H__
