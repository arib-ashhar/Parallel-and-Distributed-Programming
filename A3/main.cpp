#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <mpi.h>

using namespace std;

class Graph {
public:
    int VertexCount;
    int EdgeCount;
    int Budget;
    int WordCount;
    vector<int> Profit;
    vector<int> Cost;
    vector<vector<unsigned long long> > AdjacencyBits;

    Graph(int vertexCount, int edgeCount, int budget) {
        VertexCount = vertexCount;
        EdgeCount = edgeCount;
        Budget = budget;
        WordCount = (VertexCount + 63) >> 6;
        Profit.assign(VertexCount, 0);
        Cost.assign(VertexCount, 0);
        AdjacencyBits.assign(VertexCount, vector<unsigned long long>(WordCount, 0ULL));
    }

    void AddVertex(int vertex, int profit, int cost) {
        Profit[vertex] = profit;
        Cost[vertex] = cost;
    }

    void AddEdge(int left, int right) {
        AdjacencyBits[left][right >> 6] |= (1ULL << (right & 63));
        AdjacencyBits[right][left >> 6] |= (1ULL << (left & 63));
    }

    bool AreAdjacent(int left, int right) const {
        return (AdjacencyBits[left][right >> 6] & (1ULL << (right & 63))) != 0ULL;
    }
};

class ProfitComparator {
public:
    Graph* GraphPtr;

    ProfitComparator(Graph* graphPtr) {
        GraphPtr = graphPtr;
    }

    bool operator()(int left, int right) const {
        if (GraphPtr->Profit[left] != GraphPtr->Profit[right]) {
            return GraphPtr->Profit[left] > GraphPtr->Profit[right];
        }
        if (GraphPtr->Cost[left] != GraphPtr->Cost[right]) {
            return GraphPtr->Cost[left] < GraphPtr->Cost[right];
        }
        return left < right;
    }
};

class EfficiencyComparator {
public:
    Graph* GraphPtr;

    EfficiencyComparator(Graph* graphPtr) {
        GraphPtr = graphPtr;
    }

    bool operator()(int left, int right) const {
        long long leftValue = 1LL * GraphPtr->Profit[left] * GraphPtr->Cost[right];
        long long rightValue = 1LL * GraphPtr->Profit[right] * GraphPtr->Cost[left];

        if (leftValue != rightValue) {
            return leftValue > rightValue;
        }
        if (GraphPtr->Profit[left] != GraphPtr->Profit[right]) {
            return GraphPtr->Profit[left] > GraphPtr->Profit[right];
        }
        if (GraphPtr->Cost[left] != GraphPtr->Cost[right]) {
            return GraphPtr->Cost[left] < GraphPtr->Cost[right];
        }
        return left < right;
    }
};

class BitSetTools {
public:
    static bool Contains(const vector<unsigned long long>& bitSet, int vertex) {
        return (bitSet[vertex >> 6] & (1ULL << (vertex & 63))) != 0ULL;
    }

    static void Add(vector<unsigned long long>& bitSet, int vertex) {
        bitSet[vertex >> 6] |= (1ULL << (vertex & 63));
    }

    static bool IsEmpty(const vector<unsigned long long>& bitSet) {
        for (int i = 0; i < (int)bitSet.size(); i++) {
            if (bitSet[i] != 0ULL) {
                return false;
            }
        }
        return true;
    }

    static bool Intersects(const vector<unsigned long long>& firstBits, const vector<unsigned long long>& secondBits) {
        for (int i = 0; i < (int)firstBits.size(); i++) {
            if ((firstBits[i] & secondBits[i]) != 0ULL) {
                return true;
            }
        }
        return false;
    }
};

class ColorBound {
public:
    Graph* GraphPtr;
    vector<int>* ProfitOrderPtr;

    ColorBound(Graph* graphPtr, vector<int>* profitOrderPtr) {
        GraphPtr = graphPtr;
        ProfitOrderPtr = profitOrderPtr;
    }

    int ComputeColorBound(const vector<int>& candidateList, const vector<unsigned long long>& candidateBits) {
        if (candidateList.empty()) {
            return 0;
        }

        vector<unsigned long long> remainingBits = candidateBits;
        vector<unsigned long long> classBits(GraphPtr->WordCount, 0ULL);
        int boundValue = 0;

        while (!BitSetTools::IsEmpty(remainingBits)) {
            for (int i = 0; i < GraphPtr->WordCount; i++) {
                classBits[i] = 0ULL;
            }

            int maxProfitInClass = 0;

            for (int i = 0; i < (int)candidateList.size(); i++) {
                int vertex = candidateList[i];
                if (!BitSetTools::Contains(remainingBits, vertex)) {
                    continue;
                }

                if (!BitSetTools::Intersects(classBits, GraphPtr->AdjacencyBits[vertex])) {
                    BitSetTools::Add(classBits, vertex);
                    remainingBits[vertex >> 6] &= ~(1ULL << (vertex & 63));
                    if (GraphPtr->Profit[vertex] > maxProfitInClass) {
                        maxProfitInClass = GraphPtr->Profit[vertex];
                    }
                }
            }

            boundValue += maxProfitInClass;
        }

        return boundValue;
    }
};

class KnapsackBound {
public:
    Graph* GraphPtr;
    vector<int>* EfficiencyOrderPtr;

    KnapsackBound(Graph* graphPtr, vector<int>* efficiencyOrderPtr) {
        GraphPtr = graphPtr;
        EfficiencyOrderPtr = efficiencyOrderPtr;
    }

    int ComputeKnapsackBound(const vector<unsigned long long>& candidateBits, int remainingBudget) {
        if (remainingBudget <= 0) {
            return 0;
        }

        double boundValue = 0.0;
        int usedBudget = 0;

        for (int i = 0; i < (int)EfficiencyOrderPtr->size(); i++) {
            int vertex = (*EfficiencyOrderPtr)[i];
            if (!BitSetTools::Contains(candidateBits, vertex)) {
                continue;
            }

            int vertexCost = GraphPtr->Cost[vertex];
            int vertexProfit = GraphPtr->Profit[vertex];

            if (usedBudget + vertexCost <= remainingBudget) {
                usedBudget += vertexCost;
                boundValue += (double)vertexProfit;
            } else {
                int leftBudget = remainingBudget - usedBudget;
                if (leftBudget > 0) {
                    boundValue += ((double)leftBudget * (double)vertexProfit) / (double)vertexCost;
                }
                break;
            }
        }

        return (int)boundValue;
    }
};

class BranchAndBoundSolver {
public:
    Graph* GraphPtr;
    vector<int> ProfitOrder;
    vector<int> EfficiencyOrder;
    ColorBound ColorBoundComputer;
    KnapsackBound KnapsackBoundComputer;
    int BestProfit;
    vector<int> BestClique;
    int Rank;
    int Size;

    BranchAndBoundSolver(Graph* graphPtr, int rank, int size)
        : GraphPtr(graphPtr),
          ColorBoundComputer(graphPtr, &ProfitOrder),
          KnapsackBoundComputer(graphPtr, &EfficiencyOrder) {
        Rank = rank;
        Size = size;
        BestProfit = 0;
        BuildOrders();
    }

    void BuildOrders() {
        ProfitOrder.resize(GraphPtr->VertexCount);
        EfficiencyOrder.resize(GraphPtr->VertexCount);

        for (int i = 0; i < GraphPtr->VertexCount; i++) {
            ProfitOrder[i] = i;
            EfficiencyOrder[i] = i;
        }

        ProfitComparator profitComparator(GraphPtr);
        EfficiencyComparator efficiencyComparator(GraphPtr);

        sort(ProfitOrder.begin(), ProfitOrder.end(), profitComparator);
        sort(EfficiencyOrder.begin(), EfficiencyOrder.end(), efficiencyComparator);
    }

    void UpdateBestSolution(int currentProfit, const vector<int>& currentClique) {
        if (currentProfit > BestProfit) {
            BestProfit = currentProfit;
            BestClique = currentClique;
        }
    }

    void Search(const vector<int>& candidateList, const vector<unsigned long long>& candidateBits, int currentProfit, int currentCost, vector<int>& currentClique) {
        int colorBoundValue = ColorBoundComputer.ComputeColorBound(candidateList, candidateBits);
        if (currentProfit + colorBoundValue <= BestProfit) {
            return;
        }

        int remainingBudget = GraphPtr->Budget - currentCost;
        int knapsackBoundValue = KnapsackBoundComputer.ComputeKnapsackBound(candidateBits, remainingBudget);
        if (currentProfit + knapsackBoundValue <= BestProfit) {
            return;
        }

        vector<int> nextCandidateList;
        vector<unsigned long long> nextCandidateBits(GraphPtr->WordCount, 0ULL);

        for (int index = (int)candidateList.size() - 1; index >= 0; index--) {
            int vertex = candidateList[index];
            int newCost = currentCost + GraphPtr->Cost[vertex];

            if (newCost > GraphPtr->Budget) {
                continue;
            }

            int newProfit = currentProfit + GraphPtr->Profit[vertex];
            currentClique.push_back(vertex);
            UpdateBestSolution(newProfit, currentClique);

            nextCandidateList.clear();
            for (int i = 0; i < GraphPtr->WordCount; i++) {
                nextCandidateBits[i] = 0ULL;
            }

            for (int prefixIndex = 0; prefixIndex < index; prefixIndex++) {
                int nextVertex = candidateList[prefixIndex];
                if (GraphPtr->AreAdjacent(vertex, nextVertex)) {
                    nextCandidateList.push_back(nextVertex);
                    BitSetTools::Add(nextCandidateBits, nextVertex);
                }
            }

            if (!nextCandidateList.empty()) {
                Search(nextCandidateList, nextCandidateBits, newProfit, newCost, currentClique);
            }

            currentClique.pop_back();
        }
    }

    void Solve() {
        vector<int> currentClique;
        vector<int> rootCandidateList;
        vector<unsigned long long> rootCandidateBits(GraphPtr->WordCount, 0ULL);

        for (int index = (int)ProfitOrder.size() - 1; index >= 0; index--) {
            int branchNumber = (int)ProfitOrder.size() - 1 - index;
            if ((branchNumber % Size) != Rank) {
                continue;
            }

            int vertex = ProfitOrder[index];
            int vertexCost = GraphPtr->Cost[vertex];
            if (vertexCost > GraphPtr->Budget) {
                continue;
            }

            currentClique.clear();
            currentClique.push_back(vertex);
            UpdateBestSolution(GraphPtr->Profit[vertex], currentClique);

            rootCandidateList.clear();
            for (int i = 0; i < GraphPtr->WordCount; i++) {
                rootCandidateBits[i] = 0ULL;
            }

            for (int prefixIndex = 0; prefixIndex < index; prefixIndex++) {
                int nextVertex = ProfitOrder[prefixIndex];
                if (GraphPtr->AreAdjacent(vertex, nextVertex)) {
                    rootCandidateList.push_back(nextVertex);
                    BitSetTools::Add(rootCandidateBits, nextVertex);
                }
            }

            if (!rootCandidateList.empty()) {
                Search(rootCandidateList, rootCandidateBits, GraphPtr->Profit[vertex], vertexCost, currentClique);
            }
        }
    }
};

class ParallelSolver {
public:
    Graph* GraphPtr;
    BranchAndBoundSolver Solver;
    int Rank;
    int Size;
    int GlobalBestProfit;
    vector<int> GlobalBestClique;

    ParallelSolver(Graph* graphPtr, int rank, int size)
        : GraphPtr(graphPtr), Solver(graphPtr, rank, size) {
        Rank = rank;
        Size = size;
        GlobalBestProfit = 0;
    }

    void CollectBestResult() {
        int localCliqueSize = (int)Solver.BestClique.size();
        vector<int> localCliqueBuffer(GraphPtr->VertexCount, -1);

        for (int i = 0; i < localCliqueSize; i++) {
            localCliqueBuffer[i] = Solver.BestClique[i];
        }

        vector<int> gatheredProfits;
        vector<int> gatheredSizes;
        vector<int> gatheredCliques;

        if (Rank == 0) {
            gatheredProfits.resize(Size, 0);
            gatheredSizes.resize(Size, 0);
            gatheredCliques.resize(Size * GraphPtr->VertexCount, -1);
        }

        MPI_Gather(&Solver.BestProfit, 1, MPI_INT, Rank == 0 ? &gatheredProfits[0] : NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&localCliqueSize, 1, MPI_INT, Rank == 0 ? &gatheredSizes[0] : NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&localCliqueBuffer[0], GraphPtr->VertexCount, MPI_INT, Rank == 0 ? &gatheredCliques[0] : NULL, GraphPtr->VertexCount, MPI_INT, 0, MPI_COMM_WORLD);

        if (Rank == 0) {
            int bestRank = 0;
            int bestProfit = gatheredProfits[0];

            for (int i = 1; i < Size; i++) {
                if (gatheredProfits[i] > bestProfit) {
                    bestProfit = gatheredProfits[i];
                    bestRank = i;
                }
            }

            GlobalBestProfit = bestProfit;
            GlobalBestClique.clear();

            int cliqueSize = gatheredSizes[bestRank];
            int baseIndex = bestRank * GraphPtr->VertexCount;

            for (int i = 0; i < cliqueSize; i++) {
                int vertex = gatheredCliques[baseIndex + i];
                if (vertex >= 0) {
                    GlobalBestClique.push_back(vertex);
                }
            }
        }
    }

    void Solve() {
        MPI_Barrier(MPI_COMM_WORLD);
        auto startTime = chrono::high_resolution_clock::now();

        Solver.Solve();
        CollectBestResult();

        MPI_Barrier(MPI_COMM_WORLD);
        auto endTime = chrono::high_resolution_clock::now();

        if (Rank == 0) {
            long long elapsedMilliseconds = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
            cerr << "Execution time: " << elapsedMilliseconds << " ms" << endl;
        }
    }

    void WriteOutput(const string& outputFile) {
        if (Rank != 0) {
            return;
        }

        sort(GlobalBestClique.begin(), GlobalBestClique.end());

        ofstream outFile(outputFile.c_str());
        outFile << GlobalBestProfit << "\n";
        for (int i = 0; i < (int)GlobalBestClique.size(); i++) {
            if (i > 0) {
                outFile << " ";
            }
            outFile << GlobalBestClique[i];
        }
        outFile << "\n";
    }
};

class InputReader {
public:
    Graph* Read(const string& inputFile) {
        ifstream inFile(inputFile.c_str());

        int vertexCount = 0;
        int edgeCount = 0;
        int budget = 0;
        inFile >> vertexCount >> edgeCount >> budget;

        Graph* graph = new Graph(vertexCount, edgeCount, budget);

        for (int i = 0; i < vertexCount; i++) {
            int profit = 0;
            int cost = 0;
            inFile >> profit >> cost;
            graph->AddVertex(i, profit, cost);
        }

        for (int i = 0; i < edgeCount; i++) {
            int left = 0;
            int right = 0;
            inFile >> left >> right;
            graph->AddEdge(left, right);
        }

        return graph;
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <num_processes> ./main <input_file> <output_file>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];

    InputReader inputReader;
    Graph* graph = inputReader.Read(inputFile);

    ParallelSolver parallelSolver(graph, rank, size);
    parallelSolver.Solve();
    parallelSolver.WriteOutput(outputFile);

    delete graph;

    MPI_Finalize();
    return 0;
}
