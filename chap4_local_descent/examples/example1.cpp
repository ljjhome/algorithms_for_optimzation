#include <string>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <functional>
#include <unistd.h>
#include <glog/logging.h>
#include "common/function_interface.h"
#include "chap4_local_descent/line_search.h"

double quadfunc(const Eigen::Matrix<double, 2, 1> &x)
{
    return pow(x(0, 0), 2) + x(0, 0) * x(1, 0) + pow(x(1, 0), 2);
}

Eigen::Matrix<double, 2, 1> dquadfunc(const Eigen::Matrix<double, 2, 1> &x)
{
    Eigen::Matrix<double, 2, 1> res;
    res(0, 0) = 2 * x(0, 0) + x(1, 0);
    res(1, 0) = 2 * x(1, 0) + x(0, 0);
    return -res;
}

int main(int argc, char **argv)
{
    std::cout << "hello " << std::endl;

    using Scalar = double;
    const int Rows = 2;
    const int Cols = 1;
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;

    // function
    auto func = [](const MatrixType& x) -> Scalar {
        return quadfunc(x);
    };

    auto grad = [](const MatrixType& x) -> MatrixType {
        return dquadfunc(x);
    };

    ConcreteFunction<Scalar, Rows, Cols> myFunction(func, grad);
    BacktrackingLineSearch<Scalar, Rows, Cols> lineSearch;

    std::string log_dir = "../data/log/";

    if (access(log_dir.c_str(), 0) == -1)
    {
        std::string command = "mkdir -p " + log_dir;
        system(command.c_str());

        // 文件夹创建成功，可以开始写入LOG；
        google::InitGoogleLogging(argv[0]);
        google::SetLogDestination(google::INFO, log_dir.c_str());
        google::SetLogDestination(google::ERROR, "");
        google::SetLogDestination(google::WARNING, "");
        google::SetLogDestination(google::FATAL, "");
        google::SetStderrLogging(google::INFO);
    }
    else
    {
        // 文件夹已经存在，也可以开始写log了；
        google::InitGoogleLogging(argv[0]);
        google::SetLogDestination(google::INFO, log_dir.c_str());
        google::SetLogDestination(google::ERROR, "");
        google::SetLogDestination(google::WARNING, "");
        google::SetLogDestination(google::FATAL, "");
        google::SetStderrLogging(google::INFO);
    }

    int max_iteration_num = 1e5;
    int current_iteration_num = 0;
    MatrixType d_f_x;
    double epsilon_a = 1e-5;
    double epsilon_r = 1e-3;
    double epsilon_g = 1e-5;
    double f_x_k = 0;
    double f_x_k_1 = 1e3;
    double alpha = 0.0;
    MatrixType x_k, x_k_1;
    x_k << 1, 2;
    x_k_1 << 1, 2;
    f_x_k = myFunction.evaluate(x_k);
    f_x_k_1 = f_x_k;
    while (true)
    {
        current_iteration_num++;

        // get descent direction
        d_f_x = myFunction.gradient(x_k);

        // get step size alpha
        alpha = lineSearch.search(x_k, d_f_x, myFunction, 50);

        x_k_1 = x_k + alpha * d_f_x;

        f_x_k_1 = myFunction.evaluate(x_k_1);

        // termination condition
        // c1. maximum iterations
        if (current_iteration_num > max_iteration_num)
        {
            LOG(INFO) << "break with max_iteration_num";
            break;
        }
        // c2. absolute improvements
        if (f_x_k - f_x_k_1 < epsilon_a)
        {
            LOG(INFO) << "break with absolute improvements: " << f_x_k - f_x_k_1;
            break;
        }
        // c3. relative improvements
        if (f_x_k - f_x_k_1 < epsilon_r * std::abs(f_x_k))
        {
            LOG(INFO) << "break with relative improvements: " << f_x_k - f_x_k_1;
            break;
        }
        // c4. Gradient magnitude
        if (d_f_x.norm() < epsilon_g)
        {
            LOG(INFO) << "break with Gradient magnitude: " << d_f_x.norm();
            break;
        }
        x_k = x_k_1;
        f_x_k = f_x_k_1;
        LOG(INFO) << x_k;
    }
}
