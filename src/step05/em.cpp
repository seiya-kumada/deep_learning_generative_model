#include "src/step05/em.h"
#include <cmath>

EMAlgorithm::EMAlgorithm(const EMAlgorithm::HyperParams& params, torch::Device device)
    : params_{params}
    , device_{device}
{
    
}

namespace
{
    // test passed
    /**
     * Calculates the probability density function of a multivariate normal distribution.
     *
     * @param x The input data matrix. Its shape is [d].
     * @param mu The mean vector of the distribution. Its shape is [d].
     * @param conv The covariance matrix of the distribution. Its shape is [d, d].
     * @return The probability density value. Its shape is [1].
     */
    auto multivatiate_normal(const torch::Tensor& x, const torch::Tensor& mu, const torch::Tensor& conv) 
        -> torch::Tensor 
    {
        auto det = conv.det();
        auto inv = conv.inverse();
        auto d = x.size(0);
        auto diff = (x - mu).view({d, 1});
        auto z = 1 / torch::sqrt(std::pow(2 * M_PI, d) * det);
        auto e0 = torch::mm(diff.t(), inv);
        auto e1 = -0.5 * torch::mm(e0, diff);
        auto y = z * torch::exp(e1);
        return y;
    }

    // test passed
    /**
     * Calculates the gaussian mixture model.
     *
     * @param x The input data matrix. Its shape is [d].
     * @param phis The mixture weights. Its shape is [k,1].
     * @param mus The mean vector of the distribution. Its shape is [k, d].
     * @param convs The covariance matrix of the distribution. Its shape is [k, d, d].
     * @return The probability density value.
     */
    auto gmm(const torch::Tensor& x, const torch::Tensor& phis, const torch::Tensor& mus, const torch::Tensor& convs, torch::Device device = torch::kCPU) 
        -> torch::Tensor 
    {
        auto k = phis.size(0);
        auto y = torch::zeros({1, 1}, torch::kFloat64).to(device); 
        for (auto i = 0; i < k; ++i) {
            auto h = multivatiate_normal(x, mus[i], convs[i]);
            y += phis[i] * h;
        }
        return y;
    }

    auto likelihood(const torch::Tensor& xs, const torch::Tensor& phis, const torch::Tensor& mus, const torch::Tensor& convs, torch::Device device = torch::kCPU) 
        -> torch::Tensor 
    {
        auto eps = 1e-08;
        auto l = torch::zeros({1, 1}, torch::kFloat64).to(device);
        auto n = xs.size(0);  
        for (auto i = 0; i < n; ++i) {
            auto y = gmm(xs[i], phis, mus, convs, device); 
            l += torch::log(y + eps);
        }
        return l / n; // [1,1]
    }
}

void EMAlgorithm::execute(
    const torch::Tensor& xs, 
    torch::Tensor& phis, 
    torch::Tensor& mus, 
    torch::Tensor& convs) const 
{
    auto current_likelihood = likelihood(xs, phis, mus, convs, device_);
    for (auto iter = 0; iter < params_.max_iters; ++iter) {
        // E-step
        auto qs = execute_e_step(xs, phis, mus, convs);
         
        // M-step
        execute_m_step(qs, xs, phis, mus, convs); 

        // Convergence check and update current_likelihood
        auto is_ok = judge_convergence(current_likelihood, xs, phis, mus, convs, params_.threshold);
        if (is_ok) {
            break;
        }
    }
}
auto EMAlgorithm::execute_e_step(
    const torch::Tensor& xs, 
    const torch::Tensor& phis, 
    const torch::Tensor& mus, 
    const torch::Tensor& convs) const
-> torch::Tensor
{
    auto qs = torch::zeros({params_.num_points, params_.num_gaussians}, torch::kFloat64).to(device_);
    for (auto n = 0; n < params_.num_points; ++n) {
        const auto& x = xs[n];
        
        for (auto k = 0; k < params_.num_gaussians; ++k) {
            auto h = phis[k] * multivatiate_normal(x, mus[k], convs[k]); 
            qs[n][k] =  h.item<double>();  
        }
        qs[n] /= gmm(x, phis, mus, convs, device_).item<double>();
    }
    return qs;

}

void EMAlgorithm::execute_m_step(
    const torch::Tensor& qs,  // [n,k]
    const torch::Tensor& xs,  // [n,d]
    torch::Tensor& phis, 
    torch::Tensor& mus, 
    torch::Tensor& convs)  const
{
    auto qs_sum = qs.sum(0);
    for (auto k = 0; k < params_.num_gaussians; ++k) {
        // update phis
        phis[k] = qs_sum[k] / params_.num_points;

        // update mus
        auto c =  torch::zeros(xs[0].sizes(), torch::kFloat64).to(device_);
        for (auto n = 0; n < params_.num_points; ++n) {
            c += qs[n][k] * xs[n];
        }
        mus[k] = c / qs_sum[k];
        
        // update convs
        auto h =  torch::zeros(convs[0].sizes(), torch::kFloat64).to(device_);
        for (auto n = 0; n < params_.num_points; ++n) {
            auto diff = xs[n] - mus[k];
            h += qs[n][k] * torch::mm(diff.view({-1, 1}), diff.view({1, -1}));
        }
        convs[k] = h / qs_sum[k];
    }

}

auto EMAlgorithm::judge_convergence(
    torch::Tensor& current_likelihood, 
    const torch::Tensor& xs, 
    const torch::Tensor& phis, 
    const torch::Tensor& mus, 
    const torch::Tensor& convs,
    double threshold) const ->  bool
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "current_likelihood: " << current_likelihood.item<double>() << std::endl;
    auto next_likelihood = likelihood(xs, phis, mus, convs, device_);
    auto diff = torch::abs(next_likelihood - current_likelihood);
    current_likelihood = next_likelihood;
    if (diff.item<double>() < threshold) {
        return true;
    } else {
        return false;
    }
}

auto load_data(const std::string& file_path, torch::Device device) -> torch::Tensor
{
    auto file = std::ifstream{file_path};
    if (!file) {
        throw std::runtime_error("Cannot open the file: " + file_path);
    }

    auto data = std::vector<double>{};
    auto line = std::string{};
    auto x = 0.0;
    auto y = 0.0;
    while (std::getline(file, line)) {
        std::istringstream stream{line};
        stream >> x >> y;
        data.push_back(x);
        data.push_back(y);
    } 
    auto n = static_cast<int>(data.size()) / 2;
    auto tensor = torch::from_blob(data.data(), {n, 2}, torch::kFloat64).clone().to(device);
    return tensor;
}

#ifdef UNIT_TEST
#include <boost/test/unit_test.hpp>
namespace
{
    void test_multivaridate_normal()
    {
        std::cout << " test_multivariate_normal" << std::endl;
        auto device = torch::Device{torch::kCUDA};
        auto d = 2;
        auto k = 2;
        auto n = 10;
        auto xs = torch::ones({n, d}, torch::kFloat64).to(device);
        auto phis = 0.5 * torch::ones({k, 1}, torch::kFloat64).to(device);
        auto mus = 0.5 * torch::ones({k, d}, torch::kFloat64).to(device);
        auto convs = torch::zeros({k, d, d}, torch::kFloat64).to(device);
        for (auto i = 0; i < k; ++i) {
            convs[i] = torch::eye(d, torch::kFloat64).to(device);
        }
 
        auto y = multivatiate_normal(xs[0], mus[0], convs[0]); 
        BOOST_CHECK_CLOSE(y.item<double>(), 1 / (2 * M_PI)  * std::exp(-1.0 / 4.0), 1e-5); 
    }

    void test_gmm()
    {
        std::cout << " test_gmm" << std::endl;
        auto device = torch::Device{torch::kCUDA};
        auto d = 2;
        auto k = 2;
        auto n = 10;
        auto xs = torch::ones({n, d}, torch::kFloat64).to(device);
        auto phis = 0.5 * torch::ones({k, 1}, torch::kFloat64).to(device);
        auto mus = 0.5 * torch::ones({k, d}, torch::kFloat64).to(device);
        auto convs = torch::zeros({k, d, d}, torch::kFloat64).to(device);
        for (auto i = 0; i < k; ++i) {
            convs[i] = torch::eye(d, torch::kFloat64).to(device);
        }
        auto y = gmm(xs[0], phis, mus, convs, device); 
        BOOST_CHECK_CLOSE(y.item<double>(), 1 / (2 * M_PI)  * std::exp(-1.0 / 4.0), 1e-5); 
    }

    void test_load_data()
    {
        std::cout << " test_load_data" << std::endl;
        auto device = torch::Device{torch::kCUDA};
        const std::string FILE_PATH = "/home/ubuntu/data/deep_learning_generative_model/old_faithful.txt";
        auto xs = load_data(FILE_PATH, device);  // [n,d]
        auto x = xs[0][0].item<double>();
        auto y = xs[0][1].item<double>();
        BOOST_CHECK_CLOSE(x, 3.6, 1e-5); 
        BOOST_CHECK_CLOSE(y, 79, 1e-5); 
    }

    void test_gmm_()
    {
        std::cout << " test_gmm_" << std::endl;
        auto device = torch::Device{torch::kCUDA};
        const std::string FILE_PATH = "/home/ubuntu/data/deep_learning_generative_model/old_faithful.txt";
        auto xs = load_data(FILE_PATH, device);  // [n,d]
        
        constexpr auto K = 2;
        constexpr auto THRESHOLD = 1.0e-4;
        constexpr auto MAX_ITERS = 100;
        auto hyper_params = EMAlgorithm::HyperParams{static_cast<int>(xs.size(0)), K, THRESHOLD, MAX_ITERS};

        // initialize the parameters
        auto phis = 0.5 * torch::ones({K, 1}, torch::kFloat64).to(device);
        auto mus = torch::tensor({{0.0, 50.0}, {0.0, 100.0}}, torch::kFloat64).to(device);
        auto d = xs.size(1);
        auto convs = torch::zeros({K, d, d}, torch::kFloat64).to(device);
        for (auto i = 0; i < K; ++i) {
            convs[i] = torch::eye(d, torch::kFloat64).to(device);
        }
        auto y = gmm(xs[0], phis, mus, convs, device); 
        BOOST_CHECK(y.item<double>() < 3.0e-100); 
    }

 
    void test_likelihood()
    {
        std::cout << " test_likelihood" << std::endl;
        auto device = torch::Device{torch::kCUDA};
        auto d = 2;
        auto k = 2;
        auto n = 10;
        auto xs = torch::ones({n, d}, torch::kFloat64).to(device);
        auto phis = 0.5 * torch::ones({k, 1}, torch::kFloat64).to(device);
        auto mus = 0.5 * torch::ones({k, d}, torch::kFloat64).to(device);
        auto convs = torch::zeros({k, d, d}, torch::kFloat64).to(device);
        for (auto i = 0; i < k; ++i) {
            convs[i] = torch::eye(d, torch::kFloat64).to(device);
        }
        auto y = likelihood(xs, phis, mus, convs, device);
        auto a = 1 / (2 * M_PI)  * std::exp(-1.0 / 4.0) + 1e-08;
        auto b = std::log(a);
        BOOST_CHECK_CLOSE(y.item<double>(), b, 1e-5); 
    }

    void test_execute_e_step()
    {
        std::cout << " test_e_step" << std::endl;
        const std::string FILE_PATH = "/home/ubuntu/data/deep_learning_generative_model/old_faithful.txt";
        auto device = torch::Device{torch::kCUDA};
        auto xs = load_data(FILE_PATH, device);  // [n,d]
        
        constexpr auto K = 2;
        constexpr auto THRESHOLD = 1.0e-4;
        constexpr auto MAX_ITERS = 100;
        auto hyper_params = EMAlgorithm::HyperParams{static_cast<int>(xs.size(0)), K, THRESHOLD, MAX_ITERS};

        // initialize the parameters
        auto phis = 0.5 * torch::ones({K, 1}, torch::kFloat64).to(device);
        auto mus = torch::tensor({{0.0, 50.0}, {0.0, 100.0}}, torch::kFloat64).to(device);
        auto d = xs.size(1);
        auto convs = torch::zeros({K, d, d}, torch::kFloat64).to(device);
        for (auto i = 0; i < K; ++i) {
            convs[i] = torch::eye(d, torch::kFloat64).to(device);
        }

        auto em = EMAlgorithm{hyper_params, device};
        auto qs = em.execute_e_step(xs, phis, mus, convs);
        //std::cout << qs[0] << std::endl;
        BOOST_CHECK(qs[0][0].item<double>() < 2.0e-87); 
        BOOST_CHECK_CLOSE(qs[0][1].item<double>(), 1, 1.0e-5); 
    }

    void test_execute_m_step()
    {
        std::cout << " test_m_step" << std::endl;
        const std::string FILE_PATH = "/home/ubuntu/data/deep_learning_generative_model/old_faithful.txt";
        auto device = torch::Device{torch::kCUDA};
        auto xs = load_data(FILE_PATH, device);  // [n,d]
        
        constexpr auto K = 2;
        constexpr auto THRESHOLD = 1.0e-4;
        constexpr auto MAX_ITERS = 100;
        auto hyper_params = EMAlgorithm::HyperParams{static_cast<int>(xs.size(0)), K, THRESHOLD, MAX_ITERS};

        // initialize the parameters
        auto phis = 0.5 * torch::ones({K, 1}, torch::kFloat64).to(device);
        auto mus = torch::tensor({{0.0, 50.0}, {0.0, 100.0}}, torch::kFloat64).to(device);
        auto d = xs.size(1);
        auto convs = torch::zeros({K, d, d}, torch::kFloat64).to(device);
        for (auto i = 0; i < K; ++i) {
            convs[i] = torch::eye(d, torch::kFloat64).to(device);
        }

        auto em = EMAlgorithm(hyper_params, device);
        auto qs = em.execute_e_step(xs, phis, mus, convs);
        em.execute_m_step(qs, xs, phis, mus, convs);
        BOOST_CHECK_CLOSE(phis[0].item<double>(), 0.47794117647058826, 1.0e-5);
        BOOST_CHECK_CLOSE(phis[1].item<double>(), 0.5220588235294118, 1.0e-5);
    }
}

BOOST_AUTO_TEST_CASE(test_em_algrithm)
{
    std::cout << "test_em_algorithm" << std::endl;
    test_multivaridate_normal();
    test_gmm();
    test_gmm_();
    test_load_data();
    test_likelihood();
    test_execute_e_step();
    test_execute_m_step();
}
#endif // UNIT_TEST