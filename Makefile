INC := -Isrc -Iinclude/ -I/usr/include/eigen3
CXXFLAGS = -O3 -g -Wno-int-in-bool-context -Wall -Wextra 

src/%.o: src/%.cpp
	$(CXX) -c -std=c++17 $(CXXFLAGS) $(INC) $< -o $@

all: bin/run_lda.exe bin/run_laplace.exe

bin/run_lda.exe: src/run_lda.o
	$(CXX) -o $@ $^ -lpugixml -lboost_program_options

bin/run_laplace.exe: src/run_laplace.o
	$(CXX) -o $@ $^ -lpugixml -lboost_program_options

clean:
	$(RM) bin/*.exe src/*.o
