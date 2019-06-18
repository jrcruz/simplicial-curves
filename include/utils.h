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
// E.g. "/a/b/c.txt" -> "c"; "a./b/c.dot.txt" -> "c.dot"; "a.txt" -> "a"
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

// Given a file with text file paths, one in each line, return a mapping
// word → int for every punctuation-pruned word in all the files.
/*std::unordered_map<std::string, int> readAllVocab(const std::string& paths)
{
  std::ifstream path_file(paths);
  std::unordered_map<std::string, int> vocab;
  int vocab_size = 0; // To ensure index starts at 0.
  std::string raw_word;
  std::string path;

  // [ vocab_size := w/e
  // ; raw_word   := w/e
  // ; text_document := w_1,...,w_n, where w_i is a lower case word or digit ]
  while (path_file >> path) {
    std::ifstream text_stream(path);
    while (text_stream >> raw_word) {
      for (std::string word : splitOnPunct(raw_word)) {
        // Found punctuation at the end of a word.
        if (word.empty()) {
          continue;
        }
        // [ vocab[word] does not exists -> vocab_size := vocab_size + 1
        // | else -> I ]
        if (vocab.find(word) == vocab.end()) {
          vocab[word] = vocab_size;
          ++vocab_size;
        }
      }
    }
  }
  return vocab;
}*/

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
