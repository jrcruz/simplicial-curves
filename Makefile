INC := -Isrc -Iinclude/ -I/usr/include/eigen3 -I../lax
CXXFLAGS = -O3 -ggdb3 -Wno-int-in-bool-context -Wall -Wextra -fopenmp -fsanitize=address

src/%.o: src/%.cpp
	$(CXX) -c -std=c++17 $(CXXFLAGS) $(INC) $< -o $@

all: bin/run_lda bin/run_laplace bin/run_laplace_document_directory bin/run_lda_document_directory bin/average_documents_vocab

bin/run_lda: src/run_lda.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp -fsanitize=address


bin/run_laplace: src/run_laplace.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp -fsanitize=address


bin/run_laplace_document_directory: src/run_laplace_document_directory.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp -fsanitize=address


bin/run_lda_document_directory: src/run_lda_document_directory.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp -fsanitize=address


bin/average_documents_vocab: src/average_documents_vocab.o
	$(CXX) -g -o $@ $^ -lboost_program_options -fopenmp -fsanitize=address

clean:
	$(RM) bin/* src/*.o
