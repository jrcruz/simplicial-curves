
# Library requirements
 * _pugixml_ installed at the system level.
 * _boost_ installed at the system level.
 * _lax_ located in the directory above the project's root directory.
 * _eigen3_ located in /usr/include/.

Everything should be compiled with a minimum of C++17.

# Folder descriptions
* _bin/_: Where all the binaries end up.
* _data/_: Small collection of test documents (both with and without xml) to test-drive the program. _data/duc2002-ready_ has the entire DUC 2002 corpus (**unlicensed, be careful!**) in plain text available for more comprehensive testing.
* _include/_: Main logic of the program. Any addition will end up here.
* _scripts/_: Helper scripts necessary to pre-process the data before actually creating the curves. Refer to the _Data preparation process.png_ flowchart to understand how to run everything.
* _src/_: Main programs (as in main.cpp) for different configurations of the curve creation process.

# Script descriptions
 - _convert_xml.py_: `file`
  Grab all the text under the `<TEXT>` XML tag in _file_ and put it in a _FILE_.PRUNED.txt. Sometimes `<TEXT>` may be sub-divided into `<P>`s, in which case we grab the text from those too.
  This is used mainly to process the DUC corpora since this is how they chose to distribute their texts.

- _dir_to_corpus.cpp_: `dir_listing {text | number} [minimum_frequency (exclusive)] [num_stopwords] [corpus_output] [vocabulary_output]`
  If _number_ is passed, create an LDA-C (the Blei version) readable input file from all the documents listed in the _dir_listing_ text file (that should have one file per line). If _text_ is passed then just output a vocabulary file with one word per line. Refer to the lda-c README for the structure of the input file.

- _transpose.py_: `matrix`
  Transpose _matrix_. The matrix can be of any arbitrary dimension (so not necessarily square), but the row values must be separated by a single whitespace: "` `". Outputs to stdout.

- _graph.py_: `matrix`
  Show the graph of the matrix using PyLab and save the plot to _matrix_.plot.png. Datapoints are in the rows and axes are in the columns (so a 2D plot would be an Nx2 matrix).

* _encode_document_as_vocab_indices.py_: `vocab_file text_file`
  Encode _text_file_ as a comma-separated list of indices of words in _vocab_file_. Outputs to stdout.

* _get-curve-info-from-name_.py: `curve_dir` _DEPRECATED_
  Get the _c_, _σ_ and entropy parameters used in creating the curve from the name of every final matrix sample file in _curve_dir_ and write the values in this order to a comma-separated allfile.csv.

* _svm-classifiers/_: Has all the svm python programs for classification of topics in documents using different tipes of features.


In order to run the main program, see the total available options. Note, however, that not all of these options are usable in any one executable since the LDA and the TF-based simplex curve construction options are located in separated executables.
Refer to the program's help to get more information.
```
                                    (Optional)
                        <Args> ---------------------+
                         |                          |
                         |                          +-- σ
                         |                          |
           +-------------+                          +-- c
           |             |                          |
       (All|or none)     |   (One of each)          +-- s
     +------------+      +-------+---------+        |
     |            |              |         |        +-- n
     |            |              |         |        |
Vocabulary   LDA-matrix   Sample-type   Filepath    +-- kernel
```

# SVM
All the SVM classifiers need the class to be written in the file name of each input file. The files can be differentiated by digits, but the class must be made up of letters only. The randomized train/test split is 0.8/0.2.
The svm programs go as follows:
 * _svm.py_: `dir_listing`
  Classic TF-IDF scheme for classification. The input files are regular text files, all listed in _dir_listing_.
* _svm-lda.py_: `dir_listing lda_gamma_matrix`
  Classification with features retrieved from LDA. The _lda_gamma_matrix_ is a document by topic matrix outputted by LDA. We only need the files in _dir_listing_ to extract the labels from the file names, so we better make sure that they are aligned with the rows of the gamma matrix.
* _svm-curve.py_: `dir_listing`
  Classification using curves. Since the curves are matrices and SVM expects a vector, we flatten the matrix horizontally (but we could also sum/average each column). We also get the labels from the file names of the curve sample files in _dir_listing_.
