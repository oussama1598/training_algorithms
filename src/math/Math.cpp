#include "Math.h"

std::vector<double> Math::scalar_product(std::vector<double> a, double scalar) {
    for (size_t i = 0; i < a.size(); i++)
        a[i] = a[i] * scalar;

    return a;
}


double Math::dot_product(std::vector<double> &a, std::vector<double> &b) {
    double product = 0;

    for (size_t i = 0; i < a.size(); i++)
        product += a.at(i) * b.at(i);

    return product;
}

std::vector<double> Math::sum_vectors(std::vector<double> a, std::vector<double> b) {
    std::vector<double> c;
    c.reserve(a.size());

    for (size_t i = 0; i < a.size(); i++)
        c.push_back(a.at(i) + b.at(i));

    return c;
}


double Math::sign(double number) {
    return number > 0 ? 1 : -1;
}

double Math::same(double number) {
    return number;
}

double Math::sigmoid(double number) {
    return 1 / (1 + std::exp(-number));
}
