INC := -Iinclude/


simplicial: src
	g++ $(INC) -g -Wno-int-in-bool-context -Wall -Wextra -std=c++17 -o ./bin/simplicial.exe ./src/simplicial.cpp


opt: src
	g++ $(INC) -O3 -Wno-int-in-bool-context -Wall -Wextra -march=native -std=c++17 -o ./bin/simplicial.exe ./src/simplicial.cpp
