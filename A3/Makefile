CXX = mpic++
CXXFLAGS = -O3 -DNDEBUG -march=native -flto -std=c++17
GENCXX = g++
GENFLAGS = -O3 -DNDEBUG -march=native -flto -std=c++17

all: main generate

main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main

generate: generate_testcase.cpp
	$(GENCXX) $(GENFLAGS) generate_testcase.cpp -o generate

clean:
	rm -f main generate *.o *.txt
