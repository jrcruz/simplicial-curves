#ifndef __SCJ_KERNELS_H__
#define __SCJ_KERNELS_H__

#include <functional>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

using kernel_type = std::function<double(double, double, double)>;

// Kernel to smooth out the local (in a section from 0 to 1 in the
// length-normalized) word histogram. <mu> (in [0, 1]) is the section of the
// curve where the vocabulary distribution should be extracted from, and
// <sigma> (in [0, +inf[) is the amount of smoothing to apply throughout the
// curve. The kernel's density (<x>) is also bounded between 0 and 1.
// <x> will only be used in conjunction with the document function to smooth
// the word histograms. <sigma> is supplied only once at the time of the curve
// construction. <mu> is the parameter to vary to query different histograms
// from different parts of the curve.
// Refer to Eq. 6 in the paper for more details.
inline double smoothingGaussianKernel(double x, double mu, double sigma) {
  // [ x > 1.0 or x < 0.0 -> return := 0.0
  // | else -> I ]
  if (x < 0.0 or x > 1.0) {
    return 0.0;
  }

  // [ normal_distribution := distribution object N(0, 1) ]
  static const boost::math::normal_distribution<double> gaussian_normal(0, 1);

  // [ kernel_num := N(x; μ, σ), where N is the Gaussian distribution
  // ; kernel_den := Φ((1-μ)/σ) - Φ(-μ/σ), where Φ is the CDF of the Gaussian normal ]
  const double kernel_num = boost::math::pdf(boost::math::normal_distribution(mu, sigma), x);
  const double kernel_den = boost::math::cdf(gaussian_normal, (1 - mu) / sigma) - cdf(gaussian_normal, -mu / sigma);
  // [ return := kernel_num/kernel_den ]
  return kernel_num / kernel_den;
}

// Beta kernel to smooth out the local word histograms. Refer to
// 'smoothingGaussianKernel' for a more in-depth explanation.
inline double smoothingBetaKernel(double x, double mu, double sigma) {
  constexpr double beta = 10;
  return boost::math::pdf(boost::math::beta_distribution(beta * mu / sigma, beta * (1 - mu) / sigma), x);
}

#endif
