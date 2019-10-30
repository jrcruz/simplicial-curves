#ifndef SCJ_DISTANCES_H
#define SCJ_DISTANCES_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <sstream>
#include <fstream>
#include <chrono>
#include <complex>
#include <type_traits>

#include <Eigen/Eigen>
#include <fftw/fftw3.h>


/* In statistics, the Bhattacharyya distance measures the similarity of two probability distributions. It is closely related to the Bhattacharyya coefficient which is a measure of the amount of overlap between two statistical samples or populations.
 *  It is used to measure the separability of classes in classification and it is considered to be more reliable than the Mahalanobis distance, as the Mahalanobis distance is a particular case of the Bhattacharyya distance when the standard deviations of the two classes are the same. Consequently, when two classes have similar means but different standard deviations, the Mahalanobis distance would tend to zero, whereas the Bhattacharyya distance grows depending on the difference between the standard deviations.
 */
double
bhattacharyya(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2)
{
	return -std::log(p1.cwiseProduct(p2).cwiseSqrt().sum());
}


/*
 * In probability and statistics, the Hellinger distance (closely related to, although different from, the Bhattacharyya distance) is used to quantify the similarity between two probability distributions. It is a type of f-divergence. T
 */
double
hellinger(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2)
{
	return std::sqrt(1 - p1.cwiseProduct(p2).cwiseSqrt().sum());
}


double
kullback(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2)
{
    const auto l = p1.cwiseQuotient(p2)
                     .unaryExpr([&](auto x){ return std::log(x); });
	return p1.cwiseProduct(l).sum();
}


double
jensenShannon(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2)
{
    const Eigen::VectorXd M = 0.5 * (p1 + p2); // @suppress("Invalid arguments")
    return 0.5 * (kullback(p1, M) + kullback(p2, M));
}


double
innerProduct(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2)
{
    return p1.dot(p2);
}

#endif SCJ_DISTANCES_H
