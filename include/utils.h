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

#include <Eigen/Eigen>

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

// Integrate <func> between <begin> and <end> using the trapezoidal method
// (https://en.wikipedia.org/wiki/Trapezoidal_rule) and number <points> of
// interval sections.
double trapezoidal_integral(std::function<double(double)> func, double begin, double end, int points) {
  const double step = (end - begin) / points;
  const double at_begin = func(begin) / 2.0;
  double middle = 0.0;
  for (int j = 1; j < points; ++j) {
    middle += func(begin + j * step);
  }
  const double at_end = func(end) / 2.0;
  return step * (at_begin + middle + at_end);
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
