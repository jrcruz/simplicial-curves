#include "utils.h"
#include "document.h"
#include "sample_type.h"

namespace {
  std::string filepath;
  SampleType sample_type;
  double c_smoothing;
  double sigma;
  int int_points;
  int sample_points;
  bool use_beta;
  std::string matrix_file, vocab_file;
}

auto parse_args(int argc, char* argv[]) {

  const std::unordered_map<std::string, SampleType> map = { //
      { "curve", SampleType::curve }, //
          { "deriv", SampleType::deriv }, //
          { "both", SampleType::both } //
      };
  //                                     Optional
  //                         Args +--------------------+
  //                          +                        |
  //                          |                        +-+ σ
  //                          |                        |
  //            +-------------+                        +-+ c
  //            |             |                        |
  //         All|or none      |    One of each         +-+ s
  //      +------------+      +-------+---------+      |
  //      |            |              |         |      +-+ n
  //      +            +              +         +      |
  // Vocabulary   LDA-matrix   Sample-type   Filepath  +-+ kernel
  args::ArgumentParser parser("", "");
  args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });

  args::Group all_group(parser, "Required options", args::Group::Validators::All);
  args::ValueFlag < std::string
      > filename_p(all_group, "document", "Path to a document to analyse. This can be a simple text file or an XML file", { 'f',
                       "filepath" });
  args::MapFlag < std::string, SampleType
      > sample_type_p(
          all_group, "sample type",
          "Type of sampling to do. 'curve' outputs the document curve only. 'deriv' outputs the gradient only. 'both' outputs both",
          { "sp", "sample-type" }, map);

  args::Group lda_file_group(parser, "LDA file", args::Group::Validators::AllOrNone);
  args::ValueFlag < std::string > matrix_file_p(lda_file_group, "matrix file", "Topic x Word matrix from LDA", { "matrix" });
  args::ValueFlag < std::string
      > vocab_file_p(lda_file_group, "vocabulary file", "Vocabulary file given as input to LDA (one word per line)", { "vocab" });

  args::ValueFlag<double> c_smoothing_p(parser, "categorical smoothing",
                                        "The amount of categorical smoothing to apply. Default is 0.1", { 'c', "cat" });
  args::ValueFlag<double> k_smoothing_p(parser, "kernel smoothing", "The amount of kernel smoothing to apply. Default is 0.1", {
                                            's', "sigma" });
  args::ValueFlag<int> sample_points_p(parser, "number of sample points",
                                       "The number of points to sample from the curve. Default is 100", { 'n', "sample-number" });
  args::ValueFlag<int> integral_points_p(parser, "number of integral points",
                                         "The amount of points to use in integral calculations. Default is 50", { 'i',
                                             "integral-points" });
  args::Flag use_beta(parser, "use beta", "Use beta kernel instead of gaussian kernel for curve smoothing", { "use-beta" });

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help&) {
    std::cout << parser;
    std::exit(0);
  } catch (args::MapError&) {
    std::cout << "Sample type must be one of 'curve', 'deriv' or 'both'. Exiting\n";
    std::exit(1);
  } catch (args::ValidationError&) {
    std::cout << "Both '--filepath' and --sample-type are required arguments.\n";
    std::cout << "If you are using LDA then you must provide both --matrix and --vocab. Exiting.\n";
    std::exit(1);
  } catch (args::ParseError& e) {
    std::cout << e.what() << '\n';
    std::exit(1);
  }

  filepath = args::get(filename_p);
  if (matrix_file_p) matrix_file = args::get(matrix_file_p);
  if (vocab_file_p) vocab_file = args::get(vocab_file_p);

  sample_type = args::get(sample_type_p);
  c_smoothing = c_smoothing_p ? args::get(c_smoothing_p) : 0.1;
  sigma = k_smoothing_p ? args::get(k_smoothing_p) : 0.1;
  int_points = integral_points_p ? args::get(integral_points_p) : 50;
  sample_points = sample_points_p ? args::get(sample_points_p) : 100;

  if (sigma <= 0.0) {
    std::cerr << "Sigma should be larger than 0. Exiting.\n";
    std::exit(1);
  }
  if (c_smoothing < 0.0) {
    std::cerr << "Categorical smoothing amount can't be negative. Exiting\n";
    std::exit(1);
  }

}

int main(int argc, char* argv[]) {

  parse_args(argc, argv);

  auto kernel_func = use_beta ? smoothingBetaKernel : smoothingGaussianKernel;

  document d(matrix_file, vocab_file, filepath);

  std::cout << "Word sequence size: " << d.length() << " -- Dimension size: " << d.vocab_size() << '\n';
  d.makeCurveFunction(sigma, int_points, kernel_func);

  std::stringstream outfile_name;
  outfile_name << getFileName(filepath) << "-c" << c_smoothing << "-s" << sigma << "-ip" << int_points << "-sp" << sample_points;

  if (sample_type == SampleType::curve or sample_type == SampleType::both) {
    std::cout << "Curve:\n";
    Eigen::MatrixXd curve = d.sampleCurveDistribution(sample_points);
    printMatrix(outfile_name.str() + "_curve.txt", curve);
  }
  if (sample_type == SampleType::deriv or sample_type == SampleType::both) {
    std::cout << "Derivative:\n";
    Eigen::MatrixXd deriv = d.computeCurveDerivative(sample_points);
    Eigen::VectorXd deriv_norm = deriv.rowwise().norm();
    printMatrix(outfile_name.str() + "_deriv.txt", deriv);
    printMatrix(outfile_name.str() + "_dnorm.txt", deriv_norm);
  }
}

