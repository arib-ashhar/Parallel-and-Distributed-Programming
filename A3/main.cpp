#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

using namespace std;

class Graph {
public:
    int VertexCount;
    int EdgeCount;
    int Budget;
    int WordCount;
    vector<int> Profit;
    vector<int> Cost;
    vector<unsigned long long> AdjacencyBits;

    Graph(int vertexCount, int edgeCount, int budget) {
        VertexCount = vertexCount;
        EdgeCount = edgeCount;
        Budget = budget;
        WordCount = (VertexCount + 63) >> 6;
        Profit.assign(VertexCount, 0);
        Cost.assign(VertexCount, 0);
        AdjacencyBits.assign(VertexCount * WordCount, 0ULL);
    }

    // Stores vertex attributes.
    void AddVertex(int vertex, int profitValue, int costValue) {
        Profit[vertex] = profitValue;
        Cost[vertex] = costValue;
    }

    // Adds an undirected edge.
    void AddEdge(int leftVertex, int rightVertex) {
        AdjacencyBits[leftVertex * WordCount + (rightVertex >> 6)] |= (1ULL << (rightVertex & 63));
        AdjacencyBits[rightVertex * WordCount + (leftVertex >> 6)] |= (1ULL << (leftVertex & 63));
    }

    // Checks graph adjacency.
    bool IsAdjacent(int leftVertex, int rightVertex) const {
        return (AdjacencyBits[leftVertex * WordCount + (rightVertex >> 6)] & (1ULL << (rightVertex & 63))) != 0ULL;
    }

    // Returns the start of a vertex adjacency row.
    const unsigned long long* GetAdjacencyRow(int vertex) const {
        return &AdjacencyBits[vertex * WordCount];
    }
};

class EfficiencyComparator {
public:
    const Graph* GraphPtr;

    explicit EfficiencyComparator(const Graph* graphPtr) {
        GraphPtr = graphPtr;
    }

    bool operator()(int leftVertex, int rightVertex) const {
        long long leftValue = 1LL * GraphPtr->Profit[leftVertex] * GraphPtr->Cost[rightVertex];
        long long rightValue = 1LL * GraphPtr->Profit[rightVertex] * GraphPtr->Cost[leftVertex];

        if (leftValue != rightValue) {
            return leftValue > rightValue;
        }
        if (GraphPtr->Profit[leftVertex] != GraphPtr->Profit[rightVertex]) {
            return GraphPtr->Profit[leftVertex] > GraphPtr->Profit[rightVertex];
        }
        if (GraphPtr->Cost[leftVertex] != GraphPtr->Cost[rightVertex]) {
            return GraphPtr->Cost[leftVertex] < GraphPtr->Cost[rightVertex];
        }
        return leftVertex < rightVertex;
    }
};

class ProfitComparator {
public:
    const Graph* GraphPtr;

    explicit ProfitComparator(const Graph* graphPtr) {
        GraphPtr = graphPtr;
    }

    bool operator()(int leftVertex, int rightVertex) const {
        if (GraphPtr->Profit[leftVertex] != GraphPtr->Profit[rightVertex]) {
            return GraphPtr->Profit[leftVertex] > GraphPtr->Profit[rightVertex];
        }
        if (GraphPtr->Cost[leftVertex] != GraphPtr->Cost[rightVertex]) {
            return GraphPtr->Cost[leftVertex] < GraphPtr->Cost[rightVertex];
        }
        return leftVertex < rightVertex;
    }
};

class SearchFrame {
public:
    vector<int> Candidates;
    vector<int> OrderedCandidates;
    vector<unsigned long long> ColorClassBits;
    vector<int> ColorClassMaxProfit;
    vector<int> ProfitCounts;
    vector<int> ProfitOffsets;
    int WordCount;

    SearchFrame() {
        WordCount = 0;
    }

    SearchFrame(int vertexCount, int wordCount) {
        Candidates.reserve(vertexCount);
        OrderedCandidates.reserve(vertexCount);
        ColorClassBits.assign(vertexCount * wordCount, 0ULL);
        ColorClassMaxProfit.assign(vertexCount, 0);
        ProfitCounts.assign(101, 0);
        ProfitOffsets.assign(101, 0);
        WordCount = wordCount;
    }

    // Returns the bitset backing a color class.
    unsigned long long* GetColorClassBits(int classIndex) {
        return &ColorClassBits[classIndex * WordCount];
    }
};

class BranchAndBoundSolver {
public:
    const Graph* GraphPtr;
    int Rank;
    int Size;
    int BestProfit;
    vector<int> BestClique;
    vector<int> CurrentClique;
    vector<int> VertexOrder;
    vector<int> ProfitOrder;
    vector<SearchFrame> Frames;

    BranchAndBoundSolver(const Graph* graphPtr, int rankValue, int sizeValue) {
        GraphPtr = graphPtr;
        Rank = rankValue;
        Size = sizeValue;
        BestProfit = 0;

        VertexOrder.resize(GraphPtr->VertexCount);
        ProfitOrder.resize(GraphPtr->VertexCount);
        iota(VertexOrder.begin(), VertexOrder.end(), 0);
        iota(ProfitOrder.begin(), ProfitOrder.end(), 0);

        EfficiencyComparator efficiencyComparator(GraphPtr);
        ProfitComparator profitComparator(GraphPtr);
        sort(VertexOrder.begin(), VertexOrder.end(), efficiencyComparator);
        sort(ProfitOrder.begin(), ProfitOrder.end(), profitComparator);

        Frames.assign(GraphPtr->VertexCount + 1, SearchFrame(GraphPtr->VertexCount, GraphPtr->WordCount));
        CurrentClique.reserve(GraphPtr->VertexCount);
        BestClique.reserve(GraphPtr->VertexCount);

        InitializeGreedySolution();
    }

    // Updates the incumbent.
    void UpdateBestSolution(int profitValue) {
        if (profitValue > BestProfit) {
            BestProfit = profitValue;
            BestClique = CurrentClique;
        }
    }

    // Computes the structural color bound on the current candidate order.
    int ComputeColorBound(const vector<int>& candidates, SearchFrame& frame) {
        int colorCount = 0;
        int boundValue = 0;

        int candidateCount = static_cast<int>(candidates.size());
        frame.OrderedCandidates.resize(candidateCount);
        fill(frame.ProfitCounts.begin(), frame.ProfitCounts.end(), 0);

        for (int index = 0; index < candidateCount; index++) {
            int profitValue = GraphPtr->Profit[candidates[index]];
            frame.ProfitCounts[profitValue]++;
        }

        int currentOffset = 0;
        for (int profitValue = 100; profitValue >= 1; profitValue--) {
            frame.ProfitOffsets[profitValue] = currentOffset;
            currentOffset += frame.ProfitCounts[profitValue];
        }

        for (int index = 0; index < candidateCount; index++) {
            int vertex = candidates[index];
            int profitValue = GraphPtr->Profit[vertex];
            int writeIndex = frame.ProfitOffsets[profitValue];
            frame.OrderedCandidates[writeIndex] = vertex;
            frame.ProfitOffsets[profitValue]++;
        }

        for (int orderedIndex = 0; orderedIndex < candidateCount; orderedIndex++) {
            int vertex = frame.OrderedCandidates[orderedIndex];
            int classIndex = 0;
            while (classIndex < colorCount && IntersectsClass(frame.GetColorClassBits(classIndex), vertex)) {
                classIndex++;
            }

            if (classIndex == colorCount) {
                ResetColorClass(frame.GetColorClassBits(colorCount), frame.WordCount);
                frame.ColorClassMaxProfit[colorCount] = 0;
                colorCount++;
            }

            AddVertexToClass(frame.GetColorClassBits(classIndex), frame.ColorClassMaxProfit[classIndex], vertex);
        }

        for (int index = 0; index < colorCount; index++) {
            boundValue += frame.ColorClassMaxProfit[index];
        }

        return boundValue;
    }

    // Computes the fractional knapsack bound on the same candidate order.
    int ComputeKnapsackBound(const vector<int>& candidates, int remainingBudget) const {
        if (remainingBudget <= 0) {
            return 0;
        }

        double boundValue = 0.0;
        int usedBudget = 0;

        for (int vertex : candidates) {
            int vertexCost = GraphPtr->Cost[vertex];
            int vertexProfit = GraphPtr->Profit[vertex];

            if (usedBudget + vertexCost <= remainingBudget) {
                usedBudget += vertexCost;
                boundValue += static_cast<double>(vertexProfit);
                continue;
            }

            int leftBudget = remainingBudget - usedBudget;
            if (leftBudget > 0) {
                boundValue += (static_cast<double>(leftBudget) * static_cast<double>(vertexProfit)) / static_cast<double>(vertexCost);
            }
            break;
        }

        return static_cast<int>(boundValue);
    }

    // Performs depth first branch and bound.
    void Search(int depth, int currentProfit, int currentCost) {
        SearchFrame& currentFrame = Frames[depth];
        if (currentFrame.Candidates.empty()) {
            return;
        }

        int colorBoundValue = ComputeColorBound(currentFrame.Candidates, currentFrame);
        if (currentProfit + colorBoundValue <= BestProfit) {
            return;
        }

        int remainingBudget = GraphPtr->Budget - currentCost;
        int knapsackBoundValue = ComputeKnapsackBound(currentFrame.Candidates, remainingBudget);
        if (currentProfit + knapsackBoundValue <= BestProfit) {
            return;
        }

        SearchFrame& nextFrame = Frames[depth + 1];

        for (int index = 0; index < static_cast<int>(currentFrame.Candidates.size()); index++) {
            int vertex = currentFrame.Candidates[index];
            int newCost = currentCost + GraphPtr->Cost[vertex];

            if (newCost > GraphPtr->Budget) {
                continue;
            }

            int newProfit = currentProfit + GraphPtr->Profit[vertex];
            CurrentClique.push_back(vertex);
            UpdateBestSolution(newProfit);

            nextFrame.Candidates.clear();
            int nextBudget = GraphPtr->Budget - newCost;

            for (int nextIndex = index + 1; nextIndex < static_cast<int>(currentFrame.Candidates.size()); nextIndex++) {
                int nextVertex = currentFrame.Candidates[nextIndex];
                if (GraphPtr->Cost[nextVertex] > nextBudget) {
                    continue;
                }
                if (GraphPtr->IsAdjacent(vertex, nextVertex)) {
                    nextFrame.Candidates.push_back(nextVertex);
                }
            }

            if (!nextFrame.Candidates.empty()) {
                Search(depth + 1, newProfit, newCost);
            }

            CurrentClique.pop_back();
        }
    }

    // Builds the root candidate list in the maintained order.
    void BuildRootCandidates(vector<int>& rootCandidates) const {
        rootCandidates.clear();
        rootCandidates.reserve(GraphPtr->VertexCount);

        for (int vertex : VertexOrder) {
            if (GraphPtr->Cost[vertex] <= GraphPtr->Budget) {
                rootCandidates.push_back(vertex);
            }
        }
    }

    // Runs the sequential search.
    void SolveSingleRank() {
        SearchFrame& rootFrame = Frames[0];
        BuildRootCandidates(rootFrame.Candidates);
        Search(0, 0, 0);
    }

    // Splits the first search level across ranks.
    void SolveMultiRank() {
        vector<int> rootCandidates;
        BuildRootCandidates(rootCandidates);

        long long taskId = 0;

        for (int index = 0; index < static_cast<int>(rootCandidates.size()); index++) {
            int rootVertex = rootCandidates[index];
            int rootCost = GraphPtr->Cost[rootVertex];
            int rootProfit = GraphPtr->Profit[rootVertex];
            if (rootCost > GraphPtr->Budget) {
                continue;
            }

            vector<int> levelOneCandidates;
            levelOneCandidates.reserve(rootCandidates.size());
            int remainingBudget = GraphPtr->Budget - rootCost;

            for (int nextIndex = index + 1; nextIndex < static_cast<int>(rootCandidates.size()); nextIndex++) {
                int nextVertex = rootCandidates[nextIndex];
                if (GraphPtr->Cost[nextVertex] > remainingBudget) {
                    continue;
                }
                if (GraphPtr->IsAdjacent(rootVertex, nextVertex)) {
                    levelOneCandidates.push_back(nextVertex);
                }
            }

            CurrentClique.clear();
            CurrentClique.push_back(rootVertex);
            UpdateBestSolution(rootProfit);

            for (int secondIndex = 0; secondIndex < static_cast<int>(levelOneCandidates.size()); secondIndex++) {
                long long currentTaskId = taskId;
                taskId++;

                if ((currentTaskId % Size) != Rank) {
                    continue;
                }

                int secondVertex = levelOneCandidates[secondIndex];
                int secondCost = rootCost + GraphPtr->Cost[secondVertex];
                if (secondCost > GraphPtr->Budget) {
                    continue;
                }

                int secondProfit = rootProfit + GraphPtr->Profit[secondVertex];
                CurrentClique.clear();
                CurrentClique.push_back(rootVertex);
                CurrentClique.push_back(secondVertex);
                UpdateBestSolution(secondProfit);

                SearchFrame& nextFrame = Frames[2];
                nextFrame.Candidates.clear();
                int nextBudget = GraphPtr->Budget - secondCost;

                for (int nextIndex = secondIndex + 1; nextIndex < static_cast<int>(levelOneCandidates.size()); nextIndex++) {
                    int nextVertex = levelOneCandidates[nextIndex];
                    if (GraphPtr->Cost[nextVertex] > nextBudget) {
                        continue;
                    }
                    if (GraphPtr->IsAdjacent(secondVertex, nextVertex)) {
                        nextFrame.Candidates.push_back(nextVertex);
                    }
                }

                if (!nextFrame.Candidates.empty()) {
                    Search(2, secondProfit, secondCost);
                }
            }
        }
    }

    // Starts the solver.
    void Solve() {
        if (Size == 1) {
            SolveSingleRank();
        } else {
            SolveMultiRank();
        }
    }

private:
    // Seeds a feasible incumbent before the search starts.
    void InitializeGreedySolution() {
        TryGreedyOrder(VertexOrder);
        TryGreedyOrder(ProfitOrder);
    }

    // Builds a greedy clique using a fixed vertex order.
    void TryGreedyOrder(const vector<int>& order) {
        vector<int> greedyClique;
        greedyClique.reserve(GraphPtr->VertexCount);

        int totalProfit = 0;
        int totalCost = 0;

        for (int vertex : order) {
            int newCost = totalCost + GraphPtr->Cost[vertex];
            if (newCost > GraphPtr->Budget) {
                continue;
            }

            if (!CanJoinClique(greedyClique, vertex)) {
                continue;
            }

            greedyClique.push_back(vertex);
            totalCost = newCost;
            totalProfit += GraphPtr->Profit[vertex];
        }

        if (totalProfit > BestProfit) {
            BestProfit = totalProfit;
            BestClique = greedyClique;
        }
    }

    // Checks whether a vertex can join the current clique.
    bool CanJoinClique(const vector<int>& clique, int vertex) const {
        for (int member : clique) {
            if (!GraphPtr->IsAdjacent(member, vertex)) {
                return false;
            }
        }
        return true;
    }

    // Checks whether a vertex conflicts with a color class.
    bool IntersectsClass(const unsigned long long* colorClassBits, int vertex) const {
        const unsigned long long* adjacencyBits = GraphPtr->GetAdjacencyRow(vertex);
        for (int wordIndex = 0; wordIndex < GraphPtr->WordCount; wordIndex++) {
            if ((colorClassBits[wordIndex] & adjacencyBits[wordIndex]) != 0ULL) {
                return true;
            }
        }
        return false;
    }

    // Clears a color class bitset.
    void ResetColorClass(unsigned long long* colorClassBits, int wordCount) const {
        for (int wordIndex = 0; wordIndex < wordCount; wordIndex++) {
            colorClassBits[wordIndex] = 0ULL;
        }
    }

    // Inserts a vertex into a color class.
    void AddVertexToClass(unsigned long long* colorClassBits, int& colorClassMaxProfit, int vertex) const {
        colorClassBits[vertex >> 6] |= (1ULL << (vertex & 63));
        if (GraphPtr->Profit[vertex] > colorClassMaxProfit) {
            colorClassMaxProfit = GraphPtr->Profit[vertex];
        }
    }
};

class ParallelSolver {
public:
    const Graph* GraphPtr;
    int Rank;
    int Size;
    BranchAndBoundSolver Solver;
    int GlobalBestProfit;
    vector<int> GlobalBestClique;

    ParallelSolver(const Graph* graphPtr, int rankValue, int sizeValue)
        : GraphPtr(graphPtr), Rank(rankValue), Size(sizeValue), Solver(graphPtr, rankValue, sizeValue) {
        GlobalBestProfit = 0;
    }

    // Collects the best solution from all ranks.
    void CollectBestResult() {
        int localCliqueSize = static_cast<int>(Solver.BestClique.size());
        vector<int> localCliqueBuffer(GraphPtr->VertexCount, -1);

        for (int index = 0; index < localCliqueSize; index++) {
            localCliqueBuffer[index] = Solver.BestClique[index];
        }

        vector<int> gatheredProfits;
        vector<int> gatheredSizes;
        vector<int> gatheredCliques;

        if (Rank == 0) {
            gatheredProfits.resize(Size, 0);
            gatheredSizes.resize(Size, 0);
            gatheredCliques.resize(Size * GraphPtr->VertexCount, -1);
        }

        MPI_Gather(&Solver.BestProfit, 1, MPI_INT, Rank == 0 ? gatheredProfits.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&localCliqueSize, 1, MPI_INT, Rank == 0 ? gatheredSizes.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(localCliqueBuffer.data(), GraphPtr->VertexCount, MPI_INT, Rank == 0 ? gatheredCliques.data() : nullptr, GraphPtr->VertexCount, MPI_INT, 0, MPI_COMM_WORLD);

        if (Rank != 0) {
            return;
        }

        int bestRank = 0;
        int bestProfit = gatheredProfits[0];

        for (int rankIndex = 1; rankIndex < Size; rankIndex++) {
            if (gatheredProfits[rankIndex] > bestProfit) {
                bestProfit = gatheredProfits[rankIndex];
                bestRank = rankIndex;
            }
        }

        GlobalBestProfit = bestProfit;
        GlobalBestClique.clear();

        int cliqueSize = gatheredSizes[bestRank];
        int baseIndex = bestRank * GraphPtr->VertexCount;

        for (int index = 0; index < cliqueSize; index++) {
            int vertex = gatheredCliques[baseIndex + index];
            if (vertex >= 0) {
                GlobalBestClique.push_back(vertex);
            }
        }
    }

    // Runs solve and times only the search.
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

    // Writes the answer in the required format.
    void WriteOutput(const string& outputFile) {
        if (Rank != 0) {
            return;
        }

        sort(GlobalBestClique.begin(), GlobalBestClique.end());

        ofstream outFile(outputFile.c_str());
        outFile << GlobalBestProfit << "\n";
        for (int index = 0; index < static_cast<int>(GlobalBestClique.size()); index++) {
            if (index > 0) {
                outFile << " ";
            }
            outFile << GlobalBestClique[index];
        }
        outFile << "\n";
    }
};

class InputReader {
public:
    // Reads the graph instance.
    Graph* Read(const string& inputFile) {
        ifstream inFile(inputFile.c_str());

        int vertexCount = 0;
        int edgeCount = 0;
        int budget = 0;
        inFile >> vertexCount >> edgeCount >> budget;

        Graph* graphPtr = new Graph(vertexCount, edgeCount, budget);

        for (int vertex = 0; vertex < vertexCount; vertex++) {
            int profitValue = 0;
            int costValue = 0;
            inFile >> profitValue >> costValue;
            graphPtr->AddVertex(vertex, profitValue, costValue);
        }

        for (int edgeIndex = 0; edgeIndex < edgeCount; edgeIndex++) {
            int leftVertex = 0;
            int rightVertex = 0;
            inFile >> leftVertex >> rightVertex;
            graphPtr->AddEdge(leftVertex, rightVertex);
        }

        return graphPtr;
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
    Graph* graphPtr = inputReader.Read(inputFile);

    ParallelSolver parallelSolver(graphPtr, rank, size);
    parallelSolver.Solve();
    parallelSolver.WriteOutput(outputFile);

    delete graphPtr;

    MPI_Finalize();
    return 0;
}
