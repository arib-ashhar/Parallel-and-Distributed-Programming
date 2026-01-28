all:
	g++ -std=c++17 -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib tests.cpp functions_sequential.cpp functions.cpp -o testgen

clean:
	rm -f testgen stats* snap*