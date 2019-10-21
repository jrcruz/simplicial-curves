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
#include <complex>
#include <type_traits>

#include <Eigen/Eigen>
#include <fftw/fftw3.h>

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


// Normalize the document length to be in the interval [0, 1]. This abstracts
// away the actual document length and focuses purely on its sequential
// progression, all=owing us to compare two different documents.
// Refer to Definition 4 in the paper for more details.
double lengthNormalization(Eigen::MatrixXd const* doc, double time, int word) {
  const int ceiled_time_index = std::ceil(time * (doc->rows() - 1));
  return (*doc)(ceiled_time_index, word);
}


// Calculates the Fast Fourier Transform of the vector <derivative_norm>.
// References:
//  (1) http://www.fftw.org/fftw3_doc/ComplexOne_002dDimensionalDFTs.html
//  (2) http://www.fftw.org/fftw3_doc/One_002dDimensionalDFTsofRealData.html
//  (3) http://www.fftw.org/fftw3_doc/PlannerFlags.html
// Steps:
//  1: Alloc arrays and plan
//  2: Create plan with _plan_dft
//  3: Execute plan. This populates the output array in step 2
//  4: Destroy plan and free all arrays
Eigen::VectorXcd
fourierTransform(const Eigen::VectorXd& derivative_norm)
{
    // [ input_size   := size of derivative_norm
    // ; output_size  := size of derivative_norm / 2 + 1, as per ref. (2)
    // ; input_array  := v in R^input_size, where v_j = w/e
    // ; output_array := v in C^output_size, where v_j = w/e
    // ; plan := plan to execute the FFT with in the positive direction with a
    //        very fast, suboptimal algorithm in correctness ]
    const int input_size  = derivative_norm.rows();
    const int output_size = static_cast<int>(std::floor(input_size / 2)) + 1;
    double* input_array   = fftw_alloc_real(input_size);
    fftw_complex* output_array = fftw_alloc_complex(output_size);
    // Must create the plan before initializing the input_array.
    fftw_plan plan = fftw_plan_dft_r2c_1d(input_size, input_array, output_array,
                                          FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

    // [ input_array := v in R^array_size where v[j] = derivative_norm[j] ]
    for (int j = 0; j < input_size; ++j) {
        input_array[j] = derivative_norm[j];
    }

    // [ output_array := FFT of input_array
    // ; input_array  := w/e ]
    fftw_execute(plan);

    // [ output_vector := v in C^output_size where v[j] = output_array[j] ]
    Eigen::VectorXcd output_vector(output_size);
    for (int j = 0; j < output_size; ++j) {
        output_vector[j] = std::complex<double>(output_array[j][0], output_array[j][1]);
    }

    // [ plan; input_array; output_array := empty; empty; empty ]
    fftw_destroy_plan(plan);
    fftw_free(input_array);
    fftw_free(output_array);

    return output_vector;
}

// Measure the time that the function <f> takes to execute (in milliseconds)
// and forward any value that <f> returns.
template <typename F>
auto
measureFunctionTime(F f) -> decltype(f())
{
    auto start = std::chrono::steady_clock::now();
    if constexpr (std::is_void<decltype(f())>::value) {
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
