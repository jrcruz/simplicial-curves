#ifndef __SCJ_UTILS_H__
#define __SCJ_UTILS_H__

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <sstream>
#include <fstream>

#include "args/args.hxx"

#include <Eigen/Eigen>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

// Print the rows of a matrix separated by ',' and the columns by '\n'.
inline void printMatrix(const std::string& outpath, const Eigen::MatrixXd& matrix) {
  // [ output_stream := open file to outpath ]
  std::ofstream output_stream(outpath);
  // [ output_stream := all the rows in the matrix, separated by '\n' ]
  for (int row = 0, last_row = matrix.rows(); row < last_row; ++row) {
    // [ output_stream := all the columns in the row, separated by ',' ]
    output_stream << matrix(row, 0);
    for (int col = 1, last_col = matrix.cols(); col < last_col; ++col) {
      output_stream << ',' << matrix(row, col);
    }
    output_stream << '\n';
  }
}

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
  using boost::math::normal_distribution;
  using boost::math::pdf;
  using boost::math::cdf;

  // [ x > 1.0 or x < 0.0 -> return := 0.0
  // | else -> I ]
  if (x < 0.0 or x > 1.0) {
    return 0.0;
  }

  // [ normal_distribution := distribution object N(0, 1) ]
  static const normal_distribution<double> gaussian_normal(0, 1);

  // [ kernel_num := N(x; μ, σ), where N is the Gaussian distribution
  // ; kernel_den := Φ((1-μ)/σ) - Φ(-μ/σ), where Φ is the CDF of the Gaussian normal ]
  const double kernel_num = pdf(normal_distribution(mu, sigma), x);
  const double kernel_den = cdf(gaussian_normal, (1 - mu) / sigma) - cdf(gaussian_normal, -mu / sigma);
  // [ return := kernel_num/kernel_den ]
  return kernel_num / kernel_den;
}

// Beta kernel to smooth out the local word histograms. Refer to
// 'smoothingGaussianKernel' for a more in-depth explanation.
inline double smoothingBetaKernel(double x, double mu, double sigma) {
  using boost::math::beta_distribution;
  using boost::math::pdf;

  constexpr double beta = 10;
  return pdf(beta_distribution(beta * mu / sigma, beta * (1 - mu) / sigma), x);
}

// Calculate the Fisher information of <dist1> and <dist2>.
inline double fisherInformationMetric(Eigen::VectorXd dist1, Eigen::VectorXd dist2) {
  return std::acos(dist1.cwiseProduct(dist2).cwiseSqrt().sum());
}

// Calculate the distribution's entropy.
inline double entropy(const Eigen::VectorXd& distribution) {
  // [ return := Σ_x x * log(x) ]
  return -distribution.unaryExpr([](double prob) {
    return prob * std::log(prob);}).sum();
}

// Split a string by a delimiter character.
inline std::vector<std::string> split(const std::string& to_split, char delim) {
  std::stringstream stream(to_split);
  std::string item;
  std::vector<std::string> result;
  while (std::getline(stream, item, delim))
    result.push_back(item);
  return result;
}

// Return the base name of a file given a path.
// E.g. "/a/b/c.txt" -> "c"; "a./b/c.dot.txt" -> "c.dot"; "a.txt" -> "a"
inline std::string getFileName(const std::string& file_path) {
  return split(split(file_path, '/').back(), '.').front();
}

#endif
