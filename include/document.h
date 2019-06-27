#ifndef __SCJ_DOCUMENT_H__
#define __SCJ_DOCUMENT_H__

#include <Eigen/Eigen>
#include <functional>
#include <string>
#include <memory>
#include <iostream>
#include <new>
#include <type_traits>
#include <vector>
#include "utils.h"
#include "kernels.h"
#include <lax/basic_functions_io.h>

class document;

struct curve {
  double _sigma;
  int _integral_points;
  kernel_type _kernel_func;
  Eigen::MatrixXd const * const _doc; // purely observational. don't delete

  curve(double s, int i, kernel_type k, Eigen::MatrixXd* d)
  : _sigma(s)
  , _integral_points(i)
  , _kernel_func(k)
  , _doc(d)
  {
    std::cerr << "New curve with matrix address at " << _doc << std::endl;
  }
  curve(const curve&) = default;
  curve(curve&&) = default;

  Eigen::RowVectorXd operator()(double mu) const {
    Eigen::RowVectorXd distribution = Eigen::RowVectorXd::Zero(_doc->cols());
    for (int word = 0; word < _doc->cols(); ++word) {
      auto integrand = [=](double time) -> double {
        return lengthNormalization(_doc, time, word) * _kernel_func(time, mu, _sigma);
      };
      distribution[word] = trapezoidal_integral(integrand, 0, 1, _integral_points);
    }
    return distribution;
  }


};

class document {

protected:
public:
  using ctype = std::function<Eigen::RowVectorXd(double)>;
  std::string _filename;
  std::shared_ptr<ctype> _curve;
  std::shared_ptr<Eigen::MatrixXd> _doc_matrix;
  int _vocab_size;

private: // FUNCTIONS USED BY LAPLACE TF CONSTRUCTOR

  /**
   * Read the document given by <path> and return a vector of words, all in
   * lowercase and without punctuation.
   */
  std::vector<std::string>
  readTextDocument(const std::string& path, const std::unordered_map<std::string, int>& vocab) const
  {
    // [ text_stream, text_document, raw_word := text file ready to be read,
    //                                           empty, empty ]
    std::ifstream text_stream(path);
    std::vector<std::string> text_document;
    std::string raw_word;

    // [ text_document := w_1,...,w_n, where w_i is a lower case word or digit ]
    while (text_stream >> raw_word) {
      for (std::string word : splitOnPunct(raw_word)) {
        // Only add word if it is in the vocabulary. This also handles empty
        // words and just punctuation.
        if (vocab.find(word) != vocab.cend()) {
          text_document.emplace_back(std::move(word));
        }
      }
    }
    return text_document;
  }

  document(const std::string& filename, std::shared_ptr<ctype>& curve, int vocab_size)
  : _filename{getFileName(filename)}
  , _curve(curve)
  , _doc_matrix{nullptr}
  , _vocab_size{vocab_size}
  {

  }

public:
  document(const document& other)
  : _filename{other._filename}
  , _curve{other._curve}
  , _doc_matrix{other._doc_matrix}
  , _vocab_size{other._vocab_size}
  {
    std::cerr << "Called the copy ctor." << std::endl;
    /*
    if (other._doc_matrix) {
      std::cerr << "and copied the matrix.\n";
      _doc_matrix = std::make_shared<Eigen::MatrixXd>(other._doc_matrix);
    }
    */
  }


  document(document&& other)
  : _filename(std::move(other._filename))
  , _curve{std::move(other._curve)}
  , _doc_matrix(std::move(other._doc_matrix))
  , _vocab_size(std::move(other._vocab_size))
  {
    std::cerr << "Called the move ctor." << std::endl;
  }


public:

  /**
   * Laplace TF document reading.
   *
   * Transform a sequence of word tokens into a document matrix of
   * time x vocabulary entry. Each row corresponds then to a distribution of
   * words in a particular location ("time") in the document, hence it must sum
   * to one. With <smoothing_amount> = 0 then the matrix is a simple frequency
   * matrix. The vocabulary must be already be constructed.
   */
  document(const std::string& pathname, const std::unordered_map<std::string, int>& vocab, double smoothing_amount)
  : _filename(getFileName(pathname))
  , _curve{nullptr}
  , _doc_matrix{nullptr}
  , _vocab_size{0}
  {
    // [ word_sequence := w_1,...,w_n, where w_i is a lower case word or digit ]
    const std::vector<std::string> word_sequence = readTextDocument(pathname, vocab);

    // [ document := matrix M[i,j] = smoothing_amount for
    //               all 0 <= i <= document_size and 0 <= j <= vocab_size ]
    _doc_matrix = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Constant(word_sequence.size(), vocab.size(), smoothing_amount));
    std::cerr << "Document created a matrix at " << _doc_matrix.get() << std::endl;
    _vocab_size = static_cast<int>(vocab.size());
    // [ document := matrix M[i,j] = (smoothing_amount + P)/(1 + smoothing_amount * vocab_size)
    // , where P = 1 if word_sequence[i] = j , else P = 0 ]
    for (const auto& [time, word] : enumerate(word_sequence)) {
      (*_doc_matrix)(time, vocab.at(word)) += 1;
      _doc_matrix->row(time) /= 1 + smoothing_amount * vocab.size();
    }
  }

  /**
   * LDA reader. The method of use is the same as in the vocabulary case, the
   * only difference is that we have to provide the simplex base beforehand
   * (this is a word x dimension of interest matrix). At this point base does
   * not need to sum to one (we force normalization ourselves).
   */
  document(const std::string& matrix_pathname, const std::string& document_pathname, const std::unordered_map<std::string, int>& vocab)
  : _filename(getFileName(document_pathname))
  , _curve{nullptr}
  , _doc_matrix{nullptr}
  , _vocab_size{static_cast<int>(vocab.size())}
  {
    // [ topic_embeddings := topic_embeddings (word x topic matrix), where
    //                       each row is the embedding in the topic space for
    //          time             the word ]
    Eigen::MatrixXd topic_embeddings = lax::read_matrix(matrix_pathname, ' ');
    topic_embeddings.transposeInPlace();
    const size_t n_topics = topic_embeddings.cols();

    // Read the original file and map each word to its corresponding embedding
    // (using the lexicon to index the embedding matrix)
    std::vector<Eigen::VectorXd> word_embedding_sequence;
    std::ifstream document_file(document_pathname);
    std::string raw_word;
    while (document_file >> raw_word) {
      for (const std::string& word : splitOnPunct(raw_word)) {
        if (vocab.find(word) != vocab.cend()) {
          word_embedding_sequence.push_back(topic_embeddings.row(vocab.at(word)));
        }
      }
    }

    // XXX(jcruz): Possibly wrong. We're losing basis embedding distance
    // information if we transform it into a simplex, no matter what the
    // embedding is (word2vec, LDA, etc.).
    // E.g. (in 2 dimensions)
    //
    // |     x                        |
    // |\  x      gets normalized to  |\      .
    // | \                            | X
    // |__\___                        |__\___
    _doc_matrix = std::make_unique<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(word_embedding_sequence.size(), n_topics));
    for (size_t row = 0; row < word_embedding_sequence.size(); ++row) {
      _doc_matrix->row(row) = word_embedding_sequence[row].array().exp();
      _doc_matrix->row(row) /= _doc_matrix->row(row).sum();
    }
  }


  Eigen::VectorXd operator()(double mu) const {
    if (_curve != nullptr and *_curve) { // Function object not empty.
      return (*_curve)(mu);
    }
    throw "Called operator() on a document with no initialized curve";
  }

  document operator+(const document& other) const {
    std::string filename = _filename + ":" + other.filename();
    std::cout << "operator+().filename = " << filename << "\n";
    auto c = std::make_shared<ctype>([own_f = _curve.get(),
                                      other_f = other._curve.get()]
      (double mu) -> Eigen::RowVectorXd {;
       return ((*own_f)(mu) + (*other_f)(mu)) / 2.0;
    });
    return {filename, c, _vocab_size};
  }

  document concat(const document& other) const {
    std::string filename = _filename + ":" + other.filename();
    std::cout << "operator+().filename = " << filename << "\n";
    auto c = std::make_shared<ctype>([own_f = _curve.get(),
                                      other_f = other._curve.get()]
      (double mu) -> Eigen::RowVectorXd {
       return mu < 0.5 ? (*own_f)(mu * 2.0) : (*other_f)(mu * 2.0);
    });
    return {filename, c, _vocab_size};
  }

  // Given a document representation and a scaling amount (<sigma> > 0), returns
  // a function that accepts a <mu> between 0 and 1 (representing a timepoint
  // in the document) and returns a distribution over words at that <mu>,
  // properly smoothed with the provided <sigma> value and integral-sampled
  // using number <integral_point> of integral approximation points.
  void makeCurveFunction(double sigma, int integral_points, kernel_type kernel_func) {

    _curve = std::make_shared<ctype>(curve(sigma, integral_points, kernel_func, _doc_matrix.get()));
    std::cerr << "Created curve at address " << _curve << " with matrix at " << _doc_matrix.get() << std::endl;
  }

  // Construct a discrete representation of the document curve by sampling the
  // <curve_function> at uniform length <curve_sample_points>.
  Eigen::MatrixXd compute_curve(int sample_points) const {
    Eigen::MatrixXd sampled_curve = Eigen::MatrixXd::Zero(sample_points + 1, vocab_size());
    std::cout << "s:" << (sample_points + 1) << " v:" << vocab_size() << "\n";
//#pragma omp parallel for
    for (int mu = 0; mu < sample_points + 1; ++mu) {
      sampled_curve.row(mu) = (*_curve)(static_cast<double>(mu) / sample_points);
    }
    const double abs_error = sampled_curve.rowwise().sum().unaryExpr([](double val) {
      return std::abs(1 - val);}).sum();
    std::cout << "\nTotal sample error was " << abs_error << '\n';
    return sampled_curve;
  }

  // Calculate the derivative of the curve in an uniform <curve_sample_points>
  // interval using a default limit approximation of 10^-8.
  Eigen::MatrixXd compute_derivative(int sample_points, double h=1e-8) const {
    Eigen::MatrixXd derivative = Eigen::MatrixXd::Zero(sample_points, vocab_size());
#pragma omp parallel for
    for (int j = 0; j < sample_points; ++j) {
      const double mu = static_cast<double>(j) / sample_points;
      derivative.row(j) = ((*_curve)(mu + h) - (*_curve)(mu)) / h;
    }
    return derivative;
  }


  int vocab_size() const {
    return _vocab_size;
  }


  int length() const {
    if (_doc_matrix != nullptr) {
      return _doc_matrix->rows();
    }
     return -1;
  }


  const std::string& filename() const {
    return _filename;
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
  int revertNormalization(double norm_time) const {
    return std::floor(norm_time * (length() - 1));
  }

  // Calculate the total entropy of the curve. This is defined as the sum of the
  // entropy of all the distributions in the curve (all the points). Since we have
  // an infinite number of distributions, we integrate them.
  double curveEntropy(int integral_points) const {
    // [ return := integral_0^1 entropy(curve(μ)) dμ ]
    return trapezoidal_integral([this](double mu) {return entropy((*_curve)(mu));}, 0, 1, integral_points);
  }

  friend std::ostream& operator<<(std::ostream& o, const document& doc) {
    for (int row = 0; row < doc.length(); ++row) {
      // [ output_stream := all the columns in the row, separated by ',' ]
      o << (*doc._doc_matrix)(row, 0);
      for (int col = 1; col < doc.vocab_size(); ++col) {
        o << ',' << (*doc._doc_matrix)(row, col);
      }
      o << '\n';
    }
    return o;
  }

};

#endif // __SCJ_DOCUMENT_H__
