#ifndef __SCJ_DOCUMENT_H__
#define __SCJ_DOCUMENT_H__

#include <Eigen/Eigen>
#include <functional>
#include <string>
#include <memory>
#include <iostream>
#include <numeric>
#include <new>
#include <type_traits>
#include <vector>
#include "utils.h"
#include "kernels.h"
#include <lax/basic_functions_io.h>


class document {
public:
	using CurveFunctionType = std::function<Eigen::RowVectorXd(double)>;

private:


  std::string _filename;
  std::shared_ptr<CurveFunctionType> _curve;
  std::shared_ptr<Eigen::MatrixXd> _doc_matrix;
  int _vocab_size;

  class curve {
    double _sigma;
    int _integral_points;
    kernel_type _kernel_func;
    Eigen::MatrixXd const * const _doc; // purely observational. don't delete

  public:
    curve(double s, int i, kernel_type k, Eigen::MatrixXd* d)
    : _sigma(s)
    , _integral_points(i)
    , _kernel_func(k)
    , _doc(d)
    {;}

    curve(const curve&) = default;

    curve(curve&&) = default;

    Eigen::RowVectorXd operator()(double mu) const {
      Eigen::RowVectorXd distribution = Eigen::RowVectorXd::Zero(_doc->cols());
#pragma omp parallel for
      for (int word = 0; word < _doc->cols(); ++word) {
        auto integrand = [=](double time) -> double {
          return lengthNormalization(_doc, time, word) * _kernel_func(time, mu, _sigma);
        };
        distribution[word] = trapezoidal_integral(integrand, 0, 1, _integral_points);
      }
      return distribution;
    }
  };

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

  document(const std::string& filename, std::shared_ptr<CurveFunctionType>& curve, int vocab_size)
  : _filename(getFileName(filename))
  , _curve(curve)
  , _doc_matrix(nullptr)
  , _vocab_size(vocab_size)
  {;}

  document(const document& other) = default;

  document(document&& other) = default;

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
  , _curve(nullptr)
  , _doc_matrix(nullptr)
  , _vocab_size(static_cast<int>(vocab.size()))
  {
    // [ word_sequence := w_1,...,w_n, where w_i is a lower case word or digit ]
    const std::vector<std::string> word_sequence = readTextDocument(pathname, vocab);
    if (word_sequence.size() == 0) {
      throw std::domain_error("Trying to create an empty document. Document '" + pathname + "' was empty after preprocessing");
    }
    // [ document := matrix M[i,j] = smoothing_amount for
    //               all 0 <= i <= document_size and 0 <= j <= vocab_size ]
    _doc_matrix = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Constant(word_sequence.size(), _vocab_size, smoothing_amount));
    // [ document := matrix M[i,j] = (smoothing_amount + P)/(1 + smoothing_amount * vocab_size)
    // , where P = 1 if word_sequence[i] = j , else P = 0 ]
    for (const auto& [time, word] : enumerate(word_sequence)) {
      (*_doc_matrix)(time, vocab.at(word)) += 1;
      _doc_matrix->row(time) /= 1 + smoothing_amount * _vocab_size;
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
  , _curve(nullptr)
  , _doc_matrix(nullptr)
  , _vocab_size(0)
  {
    // [ topic_embeddings := topic_embeddings (word x topic matrix), where
    //                       each row is the embedding in the topic space for
    //                       the word ]
    Eigen::MatrixXd topic_embeddings = lax::read_matrix(matrix_pathname, ' ');
    topic_embeddings.transposeInPlace();
    _vocab_size = topic_embeddings.cols();

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
    if (word_embedding_sequence.size() == 0) {
      throw std::domain_error("Trying to create an empty document. Document '" + document_pathname + "' was empty after preprocessing");
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
    _doc_matrix = std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(word_embedding_sequence.size(), _vocab_size));
    for (size_t row = 0; row < word_embedding_sequence.size(); ++row) {
      _doc_matrix->row(row) = word_embedding_sequence[row].array().exp();
      _doc_matrix->row(row) /= _doc_matrix->row(row).sum();
    }
  }

  Eigen::VectorXd operator()(double mu) const {
    if (_curve != nullptr and *_curve) { // Function object not empty.
      return (*_curve)(mu);
    }
    throw std::domain_error("Called operator() on a document with no initialized curve");
  }


  document operator+(const document& other) const {
    std::string filename = _filename + "+" + other.filename();
    auto avg_curve = std::make_shared<CurveFunctionType>(
      [own_f = _curve.get(), other_f = other._curve.get()]
      (double mu) -> Eigen::RowVectorXd {
        return ((*own_f)(mu) + (*other_f)(mu)) / 2.0;
    });
    return {filename, avg_curve, _vocab_size};
  }

  document operator*(double scalar) const {
    auto scaled_curve = std::make_shared<CurveFunctionType>(
      [scalar, own_f = _curve.get()](double mu) -> Eigen::RowVectorXd {
        return scalar * (*own_f)(mu);
    });
    return {_filename, scaled_curve, _vocab_size};
  }

  document concat(const document& other) const {
    std::string filename = _filename + "+" + other.filename();
    auto concat_curve = std::make_shared<CurveFunctionType>(
      [own_f = _curve.get(), other_f = other._curve.get()]
      (double mu) -> Eigen::RowVectorXd {
        return mu < 0.5 ? (*own_f)(mu * 2.0) : (*other_f)(mu * 2.0);
    });
    return {filename, concat_curve, _vocab_size};
  }

  // Given a document representation and a scaling amount (<sigma> > 0), returns
  // a function that accepts a <mu> between 0 and 1 (representing a timepoint
  // in the document) and returns a distribution over words at that <mu>,
  // properly smoothed with the provided <sigma> value and integral-sampled
  // using number <integral_point> of integral approximation points.
  void makeCurveFunction(double sigma, int integral_points, kernel_type kernel_func) {
    _curve = std::make_shared<CurveFunctionType>(curve(sigma, integral_points, kernel_func, _doc_matrix.get()));
  }

  // Construct a discrete representation of the document curve by sampling the
  // <curve_function> at uniform length <curve_sample_points>.
  Eigen::MatrixXd compute_curve(int sample_points) const {
    Eigen::MatrixXd sampled_curve = Eigen::MatrixXd::Zero(sample_points + 1, vocab_size());
#pragma omp parallel for
    for (int mu = 0; mu < sample_points + 1; ++mu) {
      sampled_curve.row(mu) = (*_curve)(static_cast<double>(mu) / sample_points);
    }
    const double abs_error = sampled_curve.rowwise().sum().unaryExpr([](double val) {
      return std::abs(1 - val);}).sum();
    std::cout << "Total sample error was " << abs_error << '\n';
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

  const std::string& filename() const {
    return _filename;
  }

  // Calculate the total entropy of the curve. This is defined as the sum of the
  // entropy of all the distributions in the curve (all the points). Since we have
  // an infinite number of distributions, we integrate them.
  double curveEntropy(int integral_points) const {
    // [ return := integral_0^1 entropy(curve(μ)) dμ ]
    return trapezoidal_integral([this](double mu) {return entropy((*_curve)(mu));}, 0, 1, integral_points);
  }
};


document
sumCurves(const std::vector<document>& curves)
{
	if (curves.size() <= 0) {
		throw std::invalid_argument("Curve vector must have at least one element.");
	}
	// Combine all filenames.
	std::string filename = std::accumulate(std::cbegin(curves),
										   std::cend(curves),
										   std::string{},
        [](const std::string& p1, const document& p2) {
		    return p1 + "+" + p2.filename();
	});

	// Average all the curves.
	auto avg_curve = std::make_shared<document::CurveFunctionType>(
		[&curves](double mu) -> Eigen::RowVectorXd {
			Eigen::RowVectorXd addition_id = Eigen::RowVectorXd::Zero(curves.front().vocab_size());
			return std::accumulate(std::cbegin(curves),
								   std::cend(curves),
								   addition_id,
				[mu](const Eigen::RowVectorXd& accum, const document& doc) -> Eigen::RowVectorXd {
					Eigen::RowVectorXd curve_at_mu = doc(mu);
					return accum + curve_at_mu;
				}) / curves.size();
	});

	return {filename, avg_curve, curves.front().vocab_size()};
}


document
concatenateCurves(const std::vector<document>& curves)
{
	if (curves.size() <= 0) {
		throw std::invalid_argument("Curve vector must have at least one element.");
	}
	// Combine all filenames.
	std::string filename = std::accumulate(std::cbegin(curves),
										   std::cend(curves),
										   std::string{},
        [](const std::string& p1, const document& p2) {
		    return p1 + "+" + p2.filename();
    });


	auto concat_curve = std::make_shared<document::CurveFunctionType>(
		[&curves](double mu) -> Eigen::RowVectorXd {
			const int disambiguate = std::floor(mu * curves.size());
			const double mu_in_selected_curve = mu / curves.size();
			return curves[disambiguate](mu_in_selected_curve);
	});

	return {filename, concat_curve, curves.front().vocab_size()};
}


document
conflateCurves(const std::vector<document>& curves)
{
	if (curves.size() <= 0) {
		throw std::invalid_argument("Curve vector must have at least one element.");
	}
	// Combine all filenames.
	std::string filename = std::accumulate(std::cbegin(curves),
										   std::cend(curves),
										   std::string{},
        [](const std::string& p1, const document& p2) {
		    return p1 + "+" + p2.filename();
	});

	// Multiply all the curves.
	auto mul_curve = std::make_shared<document::CurveFunctionType>(
		[&curves] (double mu) -> Eigen::RowVectorXd {
			Eigen::RowVectorXd multiplication_id = Eigen::RowVectorXd::Constant(curves.front().vocab_size(), 1);
			Eigen::RowVectorXd conflated_curves =
					std::accumulate(std::cbegin(curves),
									std::cend(curves),
									multiplication_id,
						[mu](const Eigen::RowVectorXd& accum, const document& doc) -> Eigen::RowVectorXd {
							Eigen::RowVectorXd curve_at_mu = doc(mu);
							return accum.cwiseProduct(curve_at_mu);
						});
			return conflated_curves / conflated_curves.sum();
	});

	return {filename, mul_curve, curves.front().vocab_size()};
}


#endif // __SCJ_DOCUMENT_H__
