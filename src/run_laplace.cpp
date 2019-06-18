#include <boost/program_options.hpp>
#include "utils.h"
#include "kernels.h"
#include "document.h"

namespace {
  std::string filepath, vocab_file;
  std::string sample_type = "both";
  double c_smoothing = 0.1, sigma = 0.1;
  int int_points = 50, sample_points = 100;
  bool use_beta = false;

  void parse_options(int argc, const char *argv[]) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options() //
    ("help,h", "Display this help menu") //
    ("filepath,f", boost::program_options::value < std::string > (&filepath),
     "Document to analyse (plain text or XML file) [required]") //
    ("vocab,v", boost::program_options::value < std::string > (&vocab_file),
     "Vocabulary file given as input to LDA (one word per line) [required]") //
    ("sample-type,t", boost::program_options::value < std::string > (&sample_type),
     "Sampling type: 'curve' outputs the document curve; 'gradient' outputs the gradient (derivative); 'both' outputs both [both]") //
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

    if (!vm.count("filepath")) {
      std::cerr << "Document file was not specified" << std::endl;
      std::cerr << desc << std::endl;
      exit(1);
    }

    if (!vm.count("vocab")) {
      std::cerr << "Vocabulary file was not specified" << std::endl;
      std::cerr << desc << std::endl;
      exit(1);
    }

    if (!(sample_type == "curve" || sample_type == "gradient" || sample_type == "both")) {
      std::cerr << "Sample type must be one of 'curve', 'gradient', or 'both'." << std::endl;
      std::exit(1);
    }

    if (sigma <= 0.0) {
      std::cerr << "Sigma must be positive." << std::endl;
      std::exit(1);
    }
    if (c_smoothing < 0.0) {
      std::cerr << "Categorical smoothing amount must not be negative." << std::endl;
      std::exit(1);
    }

  }

} // anonymous namespace

int main(int argc, const char* argv[]) {

  parse_options(argc, argv);
  auto kernel_func = use_beta ? smoothingBetaKernel : smoothingGaussianKernel;

  const std::unordered_map<std::string, int> vocab = readVocab(vocab_file);
  document doc(filepath, vocab, c_smoothing);
  std::cout << "Word sequence size: " << doc.length() << " -- Dimension size: " << doc.vocab_size() << '\n';

  doc.makeCurveFunction(sigma, int_points, kernel_func);

  std::stringstream outfile_name;
  outfile_name << getFileName(filepath) << "-c" << c_smoothing << "-s" << sigma << "-ip" << int_points << "-sp" << sample_points;

  if (sample_type == "curve" or sample_type == "both") {
    std::cout << "Curve:\n";
    lax::write_matrix(doc.compute_curve(sample_points), outfile_name.str() + "_curve.txt", ',');
  }
  if (sample_type == "gradient" or sample_type == "both") {
    std::cout << "Derivative:\n";
    lax::write_matrix(doc.compute_derivative(sample_points), outfile_name.str() + "_deriv.txt", ',');
  }
}

