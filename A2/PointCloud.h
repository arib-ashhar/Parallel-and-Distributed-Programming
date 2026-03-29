#pragma once

#include <vector>
#include <string>
#include <array>
using namespace std;

struct Point {
    float x, y, z;
    int   intensity;
    int   newIntensity;

    Point() : x(0), y(0), z(0), intensity(0), newIntensity(0) {}
    Point(float x, float y, float z, int intensity)
        : x(x), y(y), z(z), intensity(intensity), newIntensity(0) {}
};

class PointCloud {
public:
    PointCloud();
    void loadFromFile(string& filePath);

    void runExactKNNSequential();
    void runExactKNN();
    void runApproxKNN();
    void runKMeans();
    void runKMeansSequential();

    int  numPoints()   const { return static_cast<int>(points.size()); }
    int  getK()        const { return k; }
    int  getMaxIter()  const { return maxIter; }

private:
    vector<Point> points;
    int k;
    int maxIter;

    float squaredDist(int i, int j) const;
    vector<int> buildHistogram(const vector<int>& neighbors) const;
    int remapIntensity(const vector<int>& histogram, int origIntensity, int m) const;
    void writeOutput(const string& filePath) const;
};