#ifndef __SCJ_DOCUMENT_H__
#define __SCJ_DOCUMENT_H__

#include <Eigen/Eigen>
#include <functional>
#include <string>
#include <vector>
#include "utils.h"
#include "kernels.h"
#include "pugixml.hpp"
#include <lax/basic_functions_io.h>

class document {

protected:

  Eigen::MatrixXd _document;

  std::function<Eigen::RowVectorXd(double)> _curve;

private:
  // FUNCTIONS USED BY LAPLACE TF CONSTRUCTOR

  /**
   * Create a vector of pairs for each element in <word_sequence>, where the
   * first element of the pair is the index of the element in the <word_sequence>
   * and the second element of the pair is the element in the word sequence.
   */
  std::vector<std::pair<int, std::string>> enumerate(const std::vector<std::string>& word_sequence) {
    std::vector<std::pair<int, std::string>> result;
    for (size_t j = 0; j < word_sequence.size(); ++j)
      result.emplace_back(j, word_sequence[j]);
    return result;
  }

  /**
   * Read the XML document given by <path> and return a tuple with a list of
   * strings (words), all in in lowercase and without punctuation, and a
   * dictionary of (word, position), indicating the position of 'word' in the
   * lexicon.
   */
  std::pair<std::vector<std::string>, std::unordered_map<std::string, int>> readXmlDocument(const std::string& path) {

    // DUC provides their test documents in malformed XML, so we need to repair
    // it to feed it to the parser.
    // [ file := open stream to text document 'path'
    // ; document_stream := open stream to XML header followed by text document 'path' ]
    std::ifstream file(path);
    std::stringstream document_stream;
    document_stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" << file.rdbuf();

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
    int vocab_size = 0; // To ensure index starts at 0.
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> text_document;
    std::string raw_word;

    // [ vocab_size := w/e
    // ; raw_word   := w/e
    // ; vocab := mapping f(w) = i where w is a lower case word or digit
    //            and i is its index in the vocabulary
    // ; text_document := w_1,...,w_n, where w_i is a lower case word or digit ]
    while (text_stream >> raw_word) {
      for (std::string word : splitOnPunct(raw_word)) {
        // Found punctuation at the end of a word.
        if (word.empty()) {
          continue;
        }
        std::cout << "Read word '" << word << "'\n";
        // [ vocab[word] does not exists -> vocab_size := vocab_size + 1
        // | else -> I ]
        if (vocab.find(word) == vocab.end()) {
          std::cout << "Word wasn't in the vocabulary. New size is ";
          vocab[word] = vocab_size;
          ++vocab_size;
          std::cout << vocab_size << "\n";
        }

        // [ text_document := text_document ++ word, where ++ is vector append
        // ; word := empty ]
        text_document.emplace_back(std::move(word));
      }
    }

    return {text_document, vocab};

  }

private:

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

public:
  /**
   * Laplace TF document reading.
   *
   * Transform a sequence of word tokens into a document matrix of
   * time x vocabulary entry. Each row corresponds then to a distribution of
   * words in a particular location ("time") in the document, hence it must sum
   * to one. With <smoothing_amount> = 0 then the matrix is a simple count
   * matrix.
   */
  document(const std::string& pathname, double smoothing_amount) {
    const auto [word_sequence, vocab] = readXmlDocument(pathname);

    // [ vocab_size := number of unique words (terms)
    // ; document_size := number of words
    // ; document := matrix M[i,j] = smoothing_amount for
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
 * LDA reader.
 */
document(const std::string& matrix_pathname, const std::string& vocab_pathname, const std::string& document_pathname) {
  // [ topic_embeddings := topic_embeddings (word x topic matrix), where
  //                       each row is the embedding in the topic space for
  //                       the word ]
  Eigen::MatrixXd topic_embeddings = lax::read_matrix(matrix_pathname, ' ');
  topic_embeddings.transposeInPlace();
  size_t n_topics = topic_embeddings.cols();

  // Load entire lexicon (only the words that have a corresponding word embedding)
  // [ vocabulary_map := word -> N mapping, from the string to the words' index in the vocabulary ]
  std::ifstream vocab_file(vocab_pathname);
  std::unordered_map<std::string, int> vocabulary_map;
  int voc_size = 0;
  std::string vocab_word;
  while (vocab_file >> vocab_word) {
    vocabulary_map[vocab_word] = voc_size;
    ++voc_size;
  }

  // Read the original file and map each word to its corresponding embedding
  // (using the lexicon to index the embedding matrix)
  std::vector<Eigen::VectorXd> word_embedding_sequence;
  std::ifstream document_file(document_pathname);
  std::string doc_word;
  while (document_file >> doc_word) {
    for (const std::string& stripped : splitOnPunct(doc_word)) {
      // Only add pruned words
      // std::cout << "Found " << stripped << "\n";
      if (vocabulary_map.find(stripped) != vocabulary_map.cend()) {
        // std::cout << "'" << stripped << "' is in the vocabulary\n";
        word_embedding_sequence.push_back(topic_embeddings.row(vocabulary_map[stripped]));
      }
    }
  }

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

// Approximate the derivative of the curve by computing the central
// differences between consecutive distributions (points on the curve). The
// number of sample points used is given by <curve_sample_points>. Naturally,
// the higher the sample points the more precise the approximation will be.
Eigen::MatrixXd compute_derivative(int sample_points) {
  constexpr double h = 1e-7;
  Eigen::MatrixXd derivative = Eigen::MatrixXd::Zero(sample_points, vocab_size());
  for (int j = 0; j < sample_points; ++j) {
    const double mu = static_cast<double>(j) / sample_points;
    std::cout << "\rAt " << j << " of " << sample_points << " points";
    derivative.row(j) = (_curve(mu + h) - _curve(mu)) / h;
  }
  return derivative;
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

      // [ distribution := vector 0,...,0 of length vocab_size ]
      Eigen::RowVectorXd distribution = Eigen::RowVectorXd::Zero(vocab_size());
      // [ distribution := distribution where all values sum to 1 and the
      //                   bigger values are concentrated around μ ]
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
Eigen::MatrixXd compute_curve(int sample_points) {
  Eigen::MatrixXd sampled_curve = Eigen::MatrixXd::Zero(sample_points + 1, vocab_size());
  for (int mu = 0; mu < sample_points + 1; ++mu) {
    std::cout << "\rAt " << mu << " of " << sample_points << " points";
    sampled_curve.row(mu) = _curve(static_cast<double>(mu) / sample_points);
  }
  const double abs_error = sampled_curve.rowwise().sum().unaryExpr([](double val) {
    return std::abs(1 - val);}).sum();
  std::cout << "\nTotal sample error was " << abs_error << '\n';
  return sampled_curve;
}

// Calculate the total entropy of the curve. This is defined as the sum of the
// entropy of all the distributions in the curve (all the points). Since we have
// an infinite number of distributions, we integrate them.
double curveEntropy(int integral_points) {
  // [ return := Int_0^1 H(γ(μ)) dμ ]
  return trapezoidal_integral([this](double mu) {return entropy(_curve(mu));}, 0, 1, integral_points);
}

int vocab_size() const {
  return _document.cols();
}

int length() const {
  return _document.rows();
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
int revertNormalization(double norm_time) {
  return std::floor(norm_time * (length() - 1));
}

friend std::ostream &operator<<(std::ostream &o, const document &d) {
  for (int row = 0; row < d.length(); ++row) {
    // [ output_stream := all the columns in the row, separated by ',' ]
    o << d._document(row, 0);
    for (int col = 1; col < d.vocab_size(); ++col) {
      o << ',' << d._document(row, col);
    }
    o << '\n';
  }
  return o;
}

};

#endif
