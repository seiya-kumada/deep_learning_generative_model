#include <torch/torch.h>

class EMAlgorithm {
public:
    struct HyperParams {
        int num_points;
        int num_gaussians;
        double threshold;
        int max_iters;
    };
    EMAlgorithm() = delete;
    EMAlgorithm(const EMAlgorithm&) = delete;
    EMAlgorithm& operator=(const EMAlgorithm&) = delete;
    EMAlgorithm(const HyperParams& params, torch::Device device = torch::kCPU);

    /**
     * Executes the EM algorithm to estimate the parameters of a generative model.
     * 
     * @param xs The input data matrix whose size is [n, d].
     * @param phis The input/output matrix to store the estimated probabilities, whose size is [k].
     * @param mus The input/output matrix to store the estimated means, whose size is [k,d].
     * @param convs The input/output matrix to store the estimated covariances, whose size is [k,d,d].
     */
    void execute(
        const torch::Tensor& xs, 
        torch::Tensor& phis, 
        torch::Tensor& mus, 
        torch::Tensor& convs) const;

    auto execute_e_step(
        const torch::Tensor& xs, 
        const torch::Tensor& phis, 
        const torch::Tensor& mus, 
        const torch::Tensor& convs) const
    -> torch::Tensor;
    
    void execute_m_step(
        const torch::Tensor& qs, 
        const torch::Tensor& xs,
        torch::Tensor& phis, 
        torch::Tensor& mus, 
        torch::Tensor& convs) const;

    auto judge_convergence(
        torch::Tensor& likelihood, 
        const torch::Tensor& xs, 
        const torch::Tensor& phis, 
        const torch::Tensor& mus, 
        const torch::Tensor& convs,
        double threshold) const -> bool;
private:
    HyperParams params_;
    torch::Device device_;
};

auto load_data(const std::string& file_path, torch::Device device = torch::kCPU) -> torch::Tensor;