#ifndef __SCJ_DOCUMENT_H__
#define __SCJ_DOCUMENT_H__

#include <Eigen/Eigen>
#include <functional>
#include <string>
#include <vector>
#include "utils.h"
#include "kernels.h"
#include <lax/basic_functions_io.h>

class document {

protected:

  Eigen::MatrixXd _document;
  std::string _filename;

  std::function<Eigen::RowVectorXd(double)> _curve;

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
  : _filename(pathname)
  {
    // [ word_sequence := w_1,...,w_n, where w_i is a lower case word or digit ]
    const std::vector<std::string> word_sequence = readTextDocument(pathname, vocab);

    // [ document := matrix M[i,j] = smoothing_amount for
    //               all 0 <= i <= document_size and 0 <= j <= vocab_size ]
    _document = Eigen::MatrixXd::Constant(word_sequence.size(), vocab.size(), smoothing_amount);

    // [ document := matrix M[i,j] = (smoothing_amount + P)/(1 + smoothing_amount * vocab_size)
    // , where P = 1 if word_sequence[i] = j , else P = 0 ]
    for (const auto& [time, word] : enumerate(word_sequence)) {
      _document(time, vocab.at(word)) += 1;
      _document.row(time) /= 1 + smoothing_amount * vocab.size();
    }
  }

  /**
   * LDA reader. The method of use is the same as in the vocabulary case, the
   * only difference is that we have to provide the simplex base beforehand
   * (this is a word x dimension of interest matrix). At this point base does
   * not need to sum to one (we force normalization ourselves).
   */
  document(const std::string& matrix_pathname, const std::string& document_pathname, const std::unordered_map<std::string, int>& vocab)
  : _filename(document_pathname)
  {
    // [ topic_embeddings := topic_embeddings (word x topic matrix), where
    //                       each row is the embedding in the topic space for
    //                       the word ]
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
    // |\  x      gets normalized to  |\
    // | \                            | X
    // |__\___                        |__\___
    _document = Eigen::MatrixXd::Zero(word_embedding_sequence.size(), n_topics);
    for (size_t row = 0; row < word_embedding_sequence.size(); ++row) {
      _document.row(row) = word_embedding_sequence[row].array().exp();
      _document.row(row) /= _document.row(row).sum();
    }
  }

  // Normalize the document length to be in the interval [0, 1]. This abstracts
  // away the actual document length and focuses purely on its sequential
  // progression, allowing us to compare two different documents.
  // Refer to Definition 4 in the paper for more details.
  double lengthNormalization(double time, int word) const {
    const int ceiled_time_index = std::ceil(time * (_document.rows() - 1));
    return _document(ceiled_time_index, word);
  }

  // Given a document representation and a scaling amount (<sigma> > 0), returns
  // a function that accepts a <mu> between 0 and 1 (representing a timepoint
  // in the document) and returns a distribution over words at that <mu>,
  // properly smoothed with the provided <sigma> value and integral-sampled
  // using number <integral_point> of integral approximation points.
  void makeCurveFunction(double sigma, int integral_points, kernel_type kernel_func) {
    // [ return := f :: Real -> Real^vocab_size, where f(μ) = distribution ]
    _curve = [=](double mu) -> Eigen::RowVectorXd {
      // [ μ < 0 or μ > 1 -> return := empty distribution
      // | else -> I ]
        if (mu < 0 or mu > 1.0) {
          return Eigen::RowVectorXd::Zero(vocab_size());
        }
        Eigen::RowVectorXd distribution = Eigen::RowVectorXd::Zero(vocab_size());
        // [ distribution := distribution where all values sum to 1 and the
        //                   bigger values are concentrated around μ ]
#pragma omp parallel for
        for (int word = 0; word < vocab_size(); ++word) {
          auto integrand = [=](double time) -> double {
            return lengthNormalization(time, word) * kernel_func(time, mu, sigma);
          };
          // [ distribution[word] := integral_0^1 ϕ_t,w * K_μ,σ(t) dt ]
          distribution[word] = trapezoidal_integral(integrand, 0, 1, integral_points);
        }

        return distribution;
      };
  }

  // Construct a discrete representation of the document curve by sampling the
  // <curve_function> at uniform length <curve_sample_points>.
  Eigen::MatrixXd compute_curve(int sample_points) const {
    Eigen::MatrixXd sampled_curve = Eigen::MatrixXd::Zero(sample_points + 1, vocab_size());
#pragma omp parallel for
    for (int mu = 0; mu < sample_points + 1; ++mu) {
      sampled_curve.row(mu) = _curve(static_cast<double>(mu) / sample_points);
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
      derivative.row(j) = (_curve(mu + h) - _curve(mu)) / h;
    }
    return derivative;
  }


  int vocab_size() const {
    return _document.cols();
  }


  int length() const {
    return _document.rows();
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
    return trapezoidal_integral([this](double mu) {return entropy(_curve(mu));}, 0, 1, integral_points);
  }

  friend std::ostream& operator<<(std::ostream& o, const document& doc) {
    for (int row = 0; row < doc.length(); ++row) {
      // [ output_stream := all the columns in the row, separated by ',' ]
      o << doc._document(row, 0);
      for (int col = 1; col < doc.vocab_size(); ++col) {
        o << ',' << doc._document(row, col);
      }
      o << '\n';
    }
    return o;
  }

};

#endif // __SCJ_DOCUMENT_H__
