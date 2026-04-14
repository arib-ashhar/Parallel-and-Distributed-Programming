#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using namespace std;

class TestCaseGenerator {
public:
    int VertexCount;
    int Budget;
    double Sparsity;

    TestCaseGenerator(int vertexCount, int budget, double sparsity) {
        VertexCount = vertexCount;
        Budget = budget;
        Sparsity = sparsity;
    }

    void GenerateAndWrite(const string& outputFile) {
        random_device randomDevice;
        mt19937 generator(randomDevice());
        uniform_int_distribution<int> profitDistribution(1, 100);
        uniform_int_distribution<int> costDistribution(1, 100);
        uniform_real_distribution<double> edgeDistribution(0.0, 1.0);

        vector<int> profits(VertexCount);
        vector<int> costs(VertexCount);
        vector<pair<int, int> > edges;

        for (int i = 0; i < VertexCount; i++) {
            profits[i] = profitDistribution(generator);
            costs[i] = costDistribution(generator);
        }

        for (int i = 0; i < VertexCount; i++) {
            for (int j = i + 1; j < VertexCount; j++) {
                if (edgeDistribution(generator) < Sparsity) {
                    edges.push_back(make_pair(i, j));
                }
            }
        }

        ofstream outFile(outputFile.c_str());
        outFile << VertexCount << " " << edges.size() << " " << Budget << "\n";

        for (int i = 0; i < VertexCount; i++) {
            outFile << profits[i] << " " << costs[i] << "\n";
        }

        for (int i = 0; i < (int)edges.size(); i++) {
            outFile << edges[i].first << " " << edges[i].second << "\n";
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cout << "Usage: ./generate <num_vertices> <budget> <sparsity> <output_file>" << endl;
        return 1;
    }

    int vertexCount = stoi(argv[1]);
    int budget = stoi(argv[2]);
    double sparsity = stod(argv[3]);
    string outputFile = argv[4];

    TestCaseGenerator generator(vertexCount, budget, sparsity);
    generator.GenerateAndWrite(outputFile);
    return 0;
}
