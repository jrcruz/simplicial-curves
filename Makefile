INC := -Isrc -Iinclude/ -I/usr/include/eigen3 -I../lax
CXXFLAGS = -O3 -g -Wno-int-in-bool-context -Wall -Wextra -fopenmp

src/%.o: src/%.cpp
	$(CXX) -c -std=c++17 $(CXXFLAGS) $(INC) $< -o $@

all: bin/run_lda bin/run_laplace bin/run_laplace_document_directory bin/run_lda_document_directory

bin/run_lda: src/run_lda.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp


bin/run_laplace: src/run_laplace.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp


bin/run_laplace_document_directory: src/run_laplace_document_directory.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp


bin/run_lda_document_directory: src/run_lda_document_directory.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp

clean:
	$(RM) bin/* src/*.o
