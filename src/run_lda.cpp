#include <boost/program_options.hpp>
#include "utils.h"
#include "document.h"
#include "sample_type.h"

namespace {
  std::string matrix_file, vocab_file, filepath;
  std::string sample_type_name = "both";
  SampleType sample_type;
  double c_smoothing = 0.1, sigma = 0.1;
  int int_points = 50, sample_points = 100;
  bool use_beta = false;

  void parse_options(int argc, const char *argv[]) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options() //
    ("help,h", "Display this help menu") //
    ("matrix,m", boost::program_options::value < std::string > (&matrix_file), "Topic x Word matrix from LDA [required]") //
    ("vocab,v", boost::program_options::value < std::string > (&vocab_file),
     "Vocabulary file given as input to LDA (one word per line) [required]") //
    ("filepath,f", boost::program_options::value < std::string > (&filepath),
     "Document to analyse (plain text or XML file) [required]") //
    ("sample-type,t",
     boost::program_options::value < std::string > (&sample_type_name),
     "Type of sampling to do: 'curve' outputs the document curve only; 'deriv' outputs the gradient only; 'both' outputs both [both]") //
    ("use-beta,b", "Use Beta kernel instead of Gaussian kernel for curve smoothing [use Gaussian]") //
    ("cat,c", boost::program_options::value<double>(&c_smoothing), "Amount of categorical smoothing to apply [0.1]") //
    ("sigma,s", boost::program_options::value<double>(&sigma), "Amount of kernel smoothing to apply [0.1]") //
    ("sample-number,n", boost::program_options::value<int>(&sample_points), "Number of points to sample from the curve [100]") //
    ("integral-points,i", boost::program_options::value<int>(&int_points), "Number of points to use in integral calculations [50]");

    boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cerr << desc << std::endl;
      exit(0);
    }

    if (vm.count("use-beta")) {
      use_beta = true;
    }

    if (!vm.count("matrix")) {
      std::cerr << "Matrix file was not specified" << std::endl;
      std::cerr << desc << std::endl;
      exit(1);
    }

    if (!vm.count("vocab")) {
      std::cerr << "Vocabulary file was not specified" << std::endl;
      std::cerr << desc << std::endl;
      exit(1);
    }

    if (!vm.count("filepath")) {
      std::cerr << "Document file was not specified" << std::endl;
      std::cerr << desc << std::endl;
      exit(1);
    }

    if (sample_type_name == "curve")
      sample_type = SampleType::curve;
    else if (sample_type_name == "deriv")
      sample_type = SampleType::deriv;
    else if (sample_type_name == "both")
      sample_type = SampleType::both;
    else {
      std::cerr << "Sample type must be one of 'curve', 'deriv' or 'both'. Exiting\n";
      std::exit(1);
    }

    if (sigma <= 0.0) {
      std::cerr << "Sigma should be larger than 0. Exiting.\n";
      std::exit(1);
    }
    if (c_smoothing < 0.0) {
      std::cerr << "Categorical smoothing amount can't be negative. Exiting\n";
      std::exit(1);
    }

  }

} // anonymous namespace

int main(int argc, const char* argv[]) {

  parse_options(argc, argv);

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

