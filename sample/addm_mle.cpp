#include <iostream>
#include <vector> 
#include <fstream>
#include <addm/cpp_toolbox.h>

// Location of the aDDM simulations. 
const std::string SIMS = "results/addm_simulations.csv";
// Location to save the computed likelihoods to. 
const std::string SAVE = "results/addm_mle.csv";
// Parameter ranges. Change as desired. 
const std::vector<float> rangeD = {0.003, 0.004, 0.005, 0.006, 0.007};
const std::vector<float> rangeSigma = {0.05, 0.06, 0.07, 0.08, 0.09};
const std::vector<float> rangeTheta = {0.3, 0.4, 0.5, 0.6, 0.7};
const std::vector<float> rangeEta = {0};

int main() {
    // Load trials from a CSV. 
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV(SIMS);
    // Add additional arguments to specify computation mode, etc.. if desired. 
    MLEinfo info = aDDM::fitModelMLE(trials, rangeD, rangeSigma, rangeTheta, rangeEta, "thread", false, 1, 0, 10, 0.1);
    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << info.optimal.d << "\n" << 
    "sigma  : " << info.optimal.sigma << "\n" << 
    "theta  : " << info.optimal.theta << "\n" << 
    "eta      : " << info.optimal.eta << std::endl;

    // Save computed likelihoods to a CSV. 
    std::ofstream fp; 
    fp.open(SAVE); 
    fp << "d,sigma,theta,p\n"; 
    for (auto &i : info.likelihoods) {
        fp << i.first.d << "," << i.first.sigma << "," << i.first.theta << "," << i.second << "\n"; 
    }
    fp.close();
}