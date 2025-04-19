// simulate_OU.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <sstream>

// -----------------------------------------------------------------------------
// Model class with the log‑normal sampler (rejection‑sampling style)
// -----------------------------------------------------------------------------
class Model {
public:
    Model();
    double random_lognormal(double mu, double sigma_n, double min_lin);
private:
    std::mt19937 gen;
};

Model::Model() {
    std::random_device rd;
    gen = std::mt19937(rd());
}

double Model::random_lognormal(double mu, double sigma_n, double min_lin) {
    std::lognormal_distribution<double> dist(mu, sigma_n);
    double value;
    do {
        value = dist(gen);
    } while (value < min_lin);
    return value;
}

// -----------------------------------------------------------------------------
// Single‑step OU update
// -----------------------------------------------------------------------------
double update_ou(double tcurrent,
                 double tmean,
                 double tcorr,
                 double sigma,
                 double dt,
                 std::mt19937 &rng)
{
    static std::normal_distribution<double> normal(0.0, 1.0);
    double dW = std::sqrt(dt) * normal(rng);
    return tcurrent - ((tcurrent - tmean) / tcorr) * dt + sigma * dW;
}

// -----------------------------------------------------------------------------
// Main: simulate N agents, each writes its own .dat file into existing "output/" folder
// -----------------------------------------------------------------------------
int main() {
    // Simulation parameters
    const int    N        = 100;     // number of agents
    const double t0       = 250.0;  // initial process value
    const double tcorr    =   125.0;   // relaxation time
    const double dt       =   1.0;   // time step
    const int    n_steps  = 3000;   // time steps per agent
    const double sigma    =   25.0;   // volatility

    // Log‑normal parameters (for t_mean sampling)
    double mu  = 3.4;
    double sigma_n = 1;
    //double var_lin   = sigma_lin * sigma_lin;
    //double mu        = std::log(mean_lin)
    //                 - 0.5 * std::log(1.0 + var_lin/(mean_lin*mean_lin));
    //double sigma_n   = std::sqrt(std::log(1.0 + var_lin/(mean_lin*mean_lin)));
    double min_lin   = 1000.0;  // minimum allowed mean

    // Prepare RNG for OU updates
    std::random_device rd;
    std::mt19937       rng(rd());

    // Model for sampling tmean
    Model model;

    // Storage for one agent’s trajectory
    std::vector<double> t_values(n_steps + 1);

    const std::string out_dir = "output";  // assume this directory already exists

    for (int agent = 0; agent < N; ++agent) {
        // 1) draw the long‑term mean for this agent
        double tmean = model.random_lognormal(mu, sigma_n, min_lin);

        // 2) initialize the process
        t_values[0] = t0;

        // 3) simulate OU over time
        for (int i = 0; i < n_steps; ++i) {
            t_values[i + 1] = update_ou(
                t_values[i],
                tmean,
                tcorr,
                sigma,
                dt,
                rng
            );
        }

        // 4) write out time series to file
        std::ostringstream filepath;
        filepath << out_dir << "/agent_" << agent << ".dat";
        std::ofstream ofs(filepath.str());
        if (!ofs) {
            std::cerr << "Error: could not open " << filepath.str() << "\n";
            continue;
        }

        ofs << "# time\tvalue\n";
        for (int i = 0; i <= n_steps; ++i) {
            ofs << (i * dt) << "\t" << t_values[i] << "\n";
        }

        if (agent % 10 == 0) {
            std::cout << "Wrote " << filepath.str()
                      << " (tmean=" << tmean << ")\n";
        }
    }

    std::cout << "All done. Check the '" << out_dir << "' directory for output files.\n";
    return 0;
}

