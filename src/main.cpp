
#ifdef UNIT_TEST
#define BOOST_TEST_MODULE MyTest
#include <boost/test/unit_test.hpp>
#else // UNIT_TEST

#include <iostream>
#include "src/step05/em.h"

void check_cuda() {
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available." << std::endl;
    } else {
        std::cout << "CUDA is available." << std::endl;
    }
}  

int main(int argc, char* argv[]) {
    try {
        auto device = torch::Device{torch::kCPU};
        //if (torch::cuda::is_available()) {
        //    device = torch::Device{torch::kCUDA};
        //    std::cout << "Using CUDA." << std::endl;
        //} else {
        //    std::cout << "Using CPU." << std::endl;
        //}

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
        for (auto k = 0; k < K; ++k) {
            convs[k] = torch::eye(d, torch::kFloat64).to(device);
        }
        
        auto em = EMAlgorithm{hyper_params, device};
        em.execute(xs, phis, mus, convs);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
#endif // UNIT_TEST