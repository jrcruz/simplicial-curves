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

#include "pugixml/pugixml.cpp"

// #include "fmt/format.h"
// #include "fmt/format.cc"

#include "Eigen/Eigen"

#include "boost/math/distributions/normal.hpp"
#include "boost/math/distributions/beta.hpp"
#include "boost/math/quadrature/trapezoidal.hpp"

#include "fftw3/fftw3.h"



enum class SampleType : char { curve, deriv, both };


// Given a vector <fourier_results>, return the maximum element of that vector.
double
difMaximum(const std::vector<double>& v)
{
    constexpr double alpha = 100000;
    // [ softmax_normalizer := Σ_i=1^N (e^αx_i)
    //   where x_i is the value at position i of the vector v ]
    const double softmax_normalizer = std::accumulate(v.begin(), v.end(), 0.0,
        [alpha](double accum, double value) {
            return accum + std::exp(value * alpha);
        });

    // [ softmax_numerator := Σ_i=1^N x_i * (e^αx_i)
    //   where x_i is the value at position i of the vector v ]
    const double softmax_numerator = std::accumulate(v.begin(), v.end(), 0.0,
        [alpha](double accum, double value) {
            return accum + value * std::exp(value * alpha);
        });
    return softmax_numerator / softmax_normalizer;
}


// Print the rows of a matrix separated by ',' and the columns by '\n'.
void
printMatrix(const std::string& outpath, const Eigen::MatrixXd& matrix)
{
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


// Read the XML document given by <path> and return a tuple with a list of
// strings (words), all in in lowercase and without punctuation, and a
// dictionary of (word, position), indicating the position of 'word' in the
// lexicon.
std::pair<std::vector<std::string>,
          std::unordered_map<std::string, int>>
readXmlDocument(const std::string& path)
{
    // DUC provides their test documents in malformed XML, so we need to repair
    // it to feed it to the parser.
    // [ file := open stream to text document 'path'
    // ; document_stream := open stream to XML header followed by text document 'path' ]
    std::ifstream file(path);
    std::stringstream document_stream;
    document_stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                    << file.rdbuf();

    pugi::xml_document xml_document;
    // [ load_success -> I
    // | else -> print error and exit ]
    if (pugi::xml_parse_result load_success = xml_document.load(document_stream)
        ; not load_success) {
        std::cout << "Failed to load file '" << path << "'\n";
        std::cout << load_success.description() << '\n';
        std::exit(1);
    }

    // [ text_stream := stream with the normal text (without tags) in the document ]
    std::stringstream text_stream;
    text_stream << xml_document.child("DOC").child("TEXT").child_value();

    // [ vocab_size, vocab, text_document, word := -1, empty, empty, empty ]
    int vocab_size = -1; // To ensure index starts at 0.
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> text_document;
    std::string word;

    // [ vocab_size := w/e
    // ; word  := w/e
    // ; vocab := mapping f(w) = i where w is a lower case word or digit
    //            and i is its index in the vocabulary
    // ; text_document := w_1,...,w_n, where w_i is a lower case word or digit ]
    while (text_stream >> word) {
        // [ word := word with all upper case letters replaced by their lower case equivalents ]
        std::transform(word.begin(), word.end(), word.begin(),
                       [](char c) { return std::tolower(c); });

        // [ word := word with only numbers or lower case letters ]
        word.erase(std::remove_if(word.begin(),
                                  word.end(),
                                  [](char c) { return std::isalnum(c) == 0; }),
                   word.end());

        // [ vocab[word] does not exists -> vocab_size := vocab_size + 1
        // | else -> I ]
        if (vocab.find(word) == vocab.end()) {
            ++vocab_size;
        }

        // [ vocab := vocab U (word, vocab_size)
        // ; text_document := text_document ++ word, where ++ is vector append
        // ; word := empty ]
        vocab[word] = vocab_size;
        text_document.emplace_back(std::move(word));
    }

    return {text_document, vocab};
}


// Create a vector of pairs for each element in <word_sequence>, where the
// first element of the pair is the index of the element in the <word_sequence>
// and the second element of the pair is the element in the word sequence.
std::vector<std::pair<int, std::string>>
enumerate(const std::vector<std::string>& word_sequence)
{
    std::vector<std::pair<int, std::string>> result;
    for (int j = 0, lim = word_sequence.size(); j < lim; ++j) {
        result.emplace_back(j, word_sequence[j]);
    }
    return result;
}


// Transform a sequence of word tokens into a document matrix of
// time x vocabulary entry. Each row corresponds then to a distribution of
// words in a particular location ("time") in the document, hence it must sum
// to one. With <smoothing_amount> = 0 then the matrix is a simple count
// matrix.
Eigen::MatrixXd
makeMatrixRepresentation(const std::vector<std::string>& word_sequence,
                         const std::unordered_map<std::string, int>& vocab,
                         double smoothing_amount)
{
    // [ vocab_size := number of unique words (terms)
    // ; document_size := number of words
    // ; document := matrix M[i,j] = smoothing_amount for
    //               all 0 <= i <= document_size and 0 <= j <= vocab_size ]
    const int vocab_size    = vocab.size();
    const int document_size = word_sequence.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        document = Eigen::MatrixXd::Constant(document_size,
                                             vocab_size,
                                             smoothing_amount);

    // [ document := matrix M[i,j] = (smoothing_amount + P)/(1 + smoothing_amount * vocab_size)
    // , where P = 1 if word_sequence[i] = j , else P = 0 ]
    for (const auto& [time, word] : enumerate(word_sequence)) {
        document(time, vocab.at(word)) += 1;
        document.row(time) /= 1 + smoothing_amount * vocab_size;
    }
    return document;
}


// Normalize the document length to be in the interval [0, 1]. This abstracts
// away the actual document length and focuses purely on its sequential
// progression, allowing us to compare two different documents.
// Refer to Definition 4 in the paper for more details.
double
lengthNormalization(double time, int word, const Eigen::MatrixXd& document)
{
    const int ceiled_time_index = std::ceil(time * (document.rows() - 1));
    return document(ceiled_time_index, word);
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
double
smoothingGaussianKernel(double x, double mu, double sigma)
{
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
    const double kernel_den = cdf(gaussian_normal, (1 - mu) / sigma)
                              - cdf(gaussian_normal, -mu / sigma);
    // [ return := kernel_num/kernel_den ]
    return kernel_num / kernel_den;
}


// Beta kernel to smooth out the local word histograms. Refer to
// 'smoothingGaussianKernel' for a more in-depth explanation.
double
smoothingBetaKernel(double x, double mu, double sigma)
{
    using boost::math::beta_distribution;
    using boost::math::pdf;

    constexpr double beta = 10;
    return pdf(beta_distribution(beta * mu / sigma,
                                 beta * (1 - mu) / sigma),
               x);
}


// Integrate <func> between <begin> and <end> using the trapezoidal method
// (https://en.wikipedia.org/wiki/Trapezoidal_rule) and number <points> of
// interval sections.
template <typename Function>
double
trapIntegrate(Function func, double begin, double end, int points)
{
    const double step = (end - begin) / points;
    const double at_begin = func(begin) / 2.0;
    double middle = 0.0;
    for (int j = 1; j < points; ++j) {
        middle += func(begin + j * step);
    }
    const double at_end = func(end) / 2.0;
    return step * (at_begin + middle + at_end);
}


// Given a document representation and a scaling amount (<sigma> > 0), returns
// a function that accepts a <mu> between 0 and 1 (representing a timepoint
// in the document) and returns a distribution over words at that <mu>,
// properly smoothed with the provided <sigma> value and integral-sampled
// using number <integral_point> of integral approximation points.
template <typename Function>
auto
makeCurve(const Eigen::MatrixXd& document_function,
          double sigma,
          int integral_points,
          Function kernel_func)
{
    const int vocab_size = document_function.cols();

    // [ return := f :: Real -> Real^vocab_size, where f(μ) = distribution ]
    return [=, &document_function](double mu) -> Eigen::RowVectorXd {
        // [ μ < 0 or μ > 1 -> return := empty distribution
        // | else -> I ]
        if (mu < 0 or mu > 1.0) {
            return Eigen::RowVectorXd::Zero(vocab_size);
        }

        // [ distribution := vector 0,...,0 of length vocab_size ]
        Eigen::RowVectorXd distribution = Eigen::RowVectorXd::Zero(vocab_size);
        // [ distribution := distribution where all values sum to 1 and the
        //                   bigger values are concentrated around μ ]
        for (int word = 0; word < vocab_size; ++word) {
            auto integrand = [=, &document_function](double time) -> double {
                return lengthNormalization(time, word, document_function)
                       * kernel_func(time, mu, sigma);
            };
            // [ distribution[word] := integral_0^1 ϕ_t,w * K_μ,σ(t) dt ]
            distribution[word] = trapIntegrate(integrand, 0, 1, integral_points);
        }
        return distribution;
    };
}


// Approximate the derivative of the curve by computing the central
// differences between consecutive distributions (points on the curve). The
// number of sample points used is given by <curve_sample_points>. Naturally,
// the higher the sample points the more precise the approximation will be.
template <typename Function>
Eigen::MatrixXd
computeCurveDerivative(Function curve_function,
                       int sample_points,
                       int vocab_size)
{
    constexpr double h = 1e-7;
    Eigen::MatrixXd derivative = Eigen::MatrixXd::Zero(sample_points,
                                                       vocab_size);
    for (int j = 0; j < sample_points; ++j) {
        const double mu = static_cast<double>(j) / sample_points;
        std::cout << "\rAt " << j << " of " << sample_points << " points";
        derivative.row(j) = (curve_function(mu + h) - curve_function(mu)) / h;
    }
    return derivative;
}


// Calculate the Fisher information of <dist1> and <dist2>.
double
fisherInformationMetric(Eigen::VectorXd dist1, Eigen::VectorXd dist2)
{
    return std::acos(dist1.cwiseProduct(dist2).cwiseSqrt().sum());
}


// Given a position <norm-time> in the document (the kernel's mu) and a
// document size (length), return the closest (discrete) word to the given
// (continuous) time point.
// To revert from the curve to the original word, we can see where the current
// mu maps to the document. If N = 100 and mu = 0.1, then the current word
// should be the 10th word in the document. If this can't be done cleanly
// (which is what happens) then consider the word to be the floor of mu * N
// (since we consider the ceil for the normalization mapping, thus erring
// upward -- this way we err downward and _somehow_ compensate).
int
revertNormalization(double norm_time, int document_size)
{
    return std::floor(norm_time * (document_size - 1));
}


// Construct a discrete representation of the document curve by sampling the
// <curve_function> at uniform length <curve_sample_points>.
template <typename Function>
Eigen::MatrixXd
sampleCurveDistribution(Function curve_function,
                        int sample_points,
                        int vocab_size)
{
    Eigen::MatrixXd sampled_curve = Eigen::MatrixXd::Zero(sample_points + 1,
                                                          vocab_size);
    for (int mu = 0; mu < sample_points + 1; ++mu) {
        std::cout << "\rAt " << mu << " of " << sample_points << " points";
        sampled_curve.row(mu) = curve_function(static_cast<double>(mu) / sample_points);
    }
    const double abs_error = sampled_curve.rowwise()
                                          .sum()
                                          .unaryExpr([](double val) {
                                              return std::abs(1 - val); })
                                          .sum();
    std::cout << "\nTotal sample error was " << abs_error << '\n';
    return sampled_curve;
}


// Calculates the Fast Fourier Transform of the vector <derivative_norm>.
// References:
//  (1) http://www.fftw.org/fftw3_doc/Complex-One_002dDimensional-DFTs.html
//  (2) http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html
//  (3) http://www.fftw.org/fftw3_doc/Planner-Flags.html
// Steps:
//  1: Alloc arrays and plan
//  2: Create plan with _plan_dft
//  3: Execute plan. This populates the output array in step 2
//  4: Destroy plan and free all arrays
Eigen::VectorXd
fourierTransform(const Eigen::VectorXd& derivative_norm)
{

    // [ input_size   := size of derivative_norm
    // ; output_size  := size of derivative_norm / 2 + 1, as per ref. (2)
    // ; input_array  := v in R^input_size, where v_j = w/e
    // ; output_array := v in C^output_size, where v_j = w/e
    // ; plan := plan to execute the FFT with in the positive direction with a
    //        very fast, sub-optimal algorithm in correctness ]
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

    // [ output_vector := v in C^output_size where v[j] = Re(output_array[j]) ]
    Eigen::VectorXd output_vector(output_size);
    for (int j = 0; j < output_size; ++j) {
        output_vector[j] = output_array[j][0];
    }

    // [ plan; input_array; output_array := empty; empty; empty ]
    fftw_destroy_plan(plan);
    fftw_free(input_array);
    fftw_free(output_array);

    return output_vector;
}



// Calculate the distribution's entropy.
double
entropy(const Eigen::VectorXd& distribution)
{
    // [ return := Σ_x x * log(x) ]
    return -distribution.unaryExpr([](double prob) {
                                        return prob * std::log(prob); })
                        .sum();
}


// Calculate the total entropy of the curve. This is defined as the sum of the
// entropy of all the distributions in the curve (all the points). Since we have
// an infinite number of distributions, we integrate them.
template <typename Function>
double
curveEntropy(Function curve_function, int integral_points)
{
    // [ return := Int_0^1 H(γ(μ)) dμ ]
    return trapIntegrate([curve_function](double mu) {
            return entropy(curve_function(mu)); },
        0, 1, integral_points);
}


// Split a string by a delimiter character.
std::vector<std::string>
split(const std::string& to_split, char delim)
{
    std::vector<std::string> result;
    std::stringstream stream(to_split);
    std::string item;
    while (std::getline(stream, item, delim)) {
        result.push_back(item);
    }
    return result;
}


// Return the base name of a file given a path.
// E.g. "/a/b/c.txt" -> "c"; "a./b/c.dot.txt" -> "c.dot"; "a.txt" -> "a"
std::string
getFileName(const std::string& file_path)
{
    std::vector<std::string> f = split(file_path, '/');
    std::string last = f.back();
    std::vector<std::string> tmp = split(last, '.');
    return tmp[0];
}


int
main(int argc, char* argv[])
{
    const std::unordered_map<std::string, SampleType> map = {
        {"curve", SampleType::curve},
        {"deriv",  SampleType::deriv},
        {"both",  SampleType::both}
    };
    args::ArgumentParser parser("", "");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::Group all_group(parser, "Required options", args::Group::Validators::All);
    args::ValueFlag<std::string> filename_p(all_group, "xml file", "Path to an XML file with the document to analyse", {'f', "filepath"});
    args::MapFlag<std::string, SampleType> sample_type_p(all_group, "sample type", "Type of sampling to do. 'curve' outputs the document curve only. 'deriv' outputs the gradient only. 'both' outputs both", {"sp", "sample-type"}, map);

    args::ValueFlag<double> c_smoothing_p(parser, "categorical smoothing", "The amount of categorical smoothing to apply. Default is 0.1", {'c', "cat"});
    args::ValueFlag<double> k_smoothing_p(parser, "kernel smoothing", "The amount of kernel smoothing to apply. Default is 0.1", {'s', "sigma"});

    args::ValueFlag<int> sample_points_p(parser, "number of sample points", "The number of points to sample from the curve. Default is 100", {'n', "sample-number"});
    args::ValueFlag<int> integral_points_p(parser, "number of integral points", "The amount of points to use in integral calculations. Default is 50", {'i', "integral-points"});

    args::Flag use_beta(parser, "use beta", "Use beta kernel instead of gaussian kernel for curve smoothing", {"use-beta"});

    try {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help&) {
        std::cout << parser;
        std::exit(0);
    }
    catch (args::MapError&) {
        std::cout << "Sample type must be one of 'curve', 'deriv' or 'both'. Exiting\n";
        std::exit(1);
    }
    catch (args::ValidationError&) {
        std::cout << "Both '--filepath' and --sample-type are required arguments. Exiting\n";
        std::exit(1);
    }
    catch (args::ParseError& e) {
        std::cout << e.what() << '\n';
        std::exit(1);
    }

    const std::string filepath   = args::get(filename_p);
    const SampleType sample_type = args::get(sample_type_p);
    const double c_smoothing = c_smoothing_p ? args::get(c_smoothing_p)
                                             : 0.1;
    const double sigma       = k_smoothing_p ? args::get(k_smoothing_p)
                                             : 0.1;
    const int int_points     = integral_points_p ? args::get(integral_points_p)
                                                 : 50;
    const int sample_points  = sample_points_p   ? args::get(sample_points_p)
                                                 : 100;
    const auto kernel_func   = use_beta ? smoothingBetaKernel
                                        : smoothingGaussianKernel;

    if (sigma <= 0.0) {
        std::cerr << "Sigma should be larger than 0. Exiting.\n";
        std::exit(1);
    }
    if (c_smoothing < 0.0) {
        std::cerr << "Categorical smoothing amount can't be negative. Exiting\n";
        std::exit(1);
    }

    const auto [document, vocab] = readXmlDocument(filepath);
    std::cout << "Document size: "       << document.size()
              << " -- Vocabulary size: " << vocab.size() << '\n';
    Eigen::MatrixXd document_matrix = makeMatrixRepresentation(document, vocab, c_smoothing);
    auto curve_function = makeCurve(document_matrix, sigma, int_points, kernel_func);

    std::stringstream outfile_name;
    outfile_name << getFileName(filepath)
                 << "-c"  << c_smoothing << "-s"  << sigma
                 << "-ip" << int_points  << "-sp" << sample_points;

    if (sample_type == SampleType::curve or sample_type == SampleType::both) {
        std::cout << "Curve:\n";
        Eigen::MatrixXd curve = sampleCurveDistribution(curve_function,
                                                        sample_points,
                                                        vocab.size());
        printMatrix(outfile_name.str() + "_curve.txt", curve);
    }
    if (sample_type == SampleType::deriv or sample_type == SampleType::both) {
        std::cout << "Derivative:\n";
        Eigen::MatrixXd deriv = computeCurveDerivative(curve_function,
                                                       sample_points,
                                                       vocab.size());
        Eigen::VectorXd deriv_norm = deriv.rowwise().norm();
        Eigen::VectorXd fourier_trans = fourierTransform(deriv_norm);
        fourier_trans = fourier_trans.tail(fourier_trans.rows() - 1);
        printMatrix(outfile_name.str() + "_deriv.txt", deriv);
        printMatrix(outfile_name.str() + "_dnorm.txt", deriv_norm);
    }
}

