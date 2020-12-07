#pragma once

#include <cmath>
#include <vector>

class Math {
public:
    static std::vector<double> scalar_product(std::vector<double> a, double scalar);

    static double dot_product(std::vector<double> &a, std::vector<double> &b);

    static std::vector<double> sum_vectors(std::vector<double> a, std::vector<double> b);

    static double sign(double number);

    static double same(double number);

    static double sigmoid(double number);
};

