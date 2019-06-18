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
#include <chrono>
#include <type_traits>

#include <Eigen/Eigen>

/**
 * Create a vector of pairs for each element in <word_sequence>, where the
 * first element of the pair is the index of the element in the <word_sequence>
 * and the second element of the pair is the element in the word sequence.
 */
template <typename T>
std::vector<std::pair<int, T>> enumerate(const std::vector<T>& word_sequence) {
  std::vector<std::pair<int, T>> result;
  for (size_t j = 0; j < word_sequence.size(); ++j) {
    result.emplace_back(j, word_sequence[j]);
  }
  return result;
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
  while (std::getline(stream, item, delim)) {
    result.push_back(item);
  }
  return result;
}

// Return the base name of a file given a path.
// E.g. "/a/b/c.txt" -> "c"; "a/b/c.dot.txt" -> "c"; "a" -> "a"
inline std::string getFileName(const std::string& file_path) {
  return split(split(file_path, '/').back(), '.').front();
}

// Splits <word> by any of the following characters:
// !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
std::vector<std::string> splitOnPunct(const std::string& word) {
  std::vector<std::string> split_vector;
  std::string tmp;
  for (char c : word) {
    if (std::ispunct(c) != 0) {
      split_vector.emplace_back(tmp);
      tmp.clear();
    } else {
      tmp.push_back(std::tolower(c));
    }
  }
  split_vector.emplace_back(tmp);
  return split_vector;
}

// Reads a file with one word per line and returns a mapping word -> int, where
// int is the line where the word was in the file.
std::unordered_map<std::string, int>
readVocab(const std::string& vocab_path)
{
  std::ifstream vocab_file(vocab_path);
  std::unordered_map<std::string, int> vocab;
  int voc_size = 0;
  std::string vocab_word;
  while (vocab_file >> vocab_word) {
    vocab[vocab_word] = voc_size;
    ++voc_size;
  }
  return vocab;
}

// Measure the time that the function <f> takes to execute (in milliseconds)
// and forward any value that <f> returns.
template <typename F>
auto
measureFunctionTime(F f) -> decltype(f())
{
    auto start = std::chrono::steady_clock::now();
    if constexpr (std::is_void_v<decltype(f())>) {
        f();
        std::chrono::duration<double, std::milli> dur = std::chrono::steady_clock::now() - start;
        std::cerr << "Function took " << dur.count() << "ms.\n";
    }
    else {
        auto ret = f();
        std::chrono::duration<double, std::milli> dur = std::chrono::steady_clock::now() - start;
        std::cerr << "Function took " << dur.count() << "ms.\n";
        return ret;
    }
}

#endif
