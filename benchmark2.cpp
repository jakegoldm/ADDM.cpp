#include <addm/cpp_toolbox.h>
#include <vector> 
#include <random>
#include <map>
#include <string>
#include <chrono>

int TIMESTEP = 10; 
int NDT = 0; 

float d = 0.006; 
float sigma = 0.06; 

vector<float> range_t = {0.3, 0.5, 0.7}; 
vector<float> range_e = {0.004, 0.006, 0.008};

vector<float> test_t = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}; 
vector<float> test_e = {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01}; 

vector<float> vals = {0, 1, 2, 3};
vector<int> timeSteps = {10, 5}; 
vector<float> stateSteps = {0.1, 0.01}; 
vector<int> trialCnts = {100, 500, 1000};

string EXP_DATA = "data/expdata.csv";
string FIX_DATA = "data/fixations.csv"; 


int main() {
  map<int, vector<aDDMTrial>> data = loadDataFromCSV(EXP_DATA, FIX_DATA);
  FixationData fixationData = getEmpiricalDistributions(data);

  mt19937 generator(random_device{}());
  uniform_int_distribution<size_t> distribution(0, vals.size() - 1);

  vector<pair<aDDM, vector<aDDMTrial>>> subj_data; 
  int maxTrials = *max_element(trialCnts.begin(), trialCnts.end()); 

  for (float t : range_t) {
    aDDM addm = aDDM(d, sigma, t, 0, 1, NDT);
    vector<aDDMTrial> trials;
    for (int i = 0; i < maxTrials; i++) {
      int r_i = distribution(generator);
      int l_i = distribution(generator);
      aDDMTrial trial = addm.simulateTrial(vals[l_i], vals[r_i], fixationData);
      trials.push_back(trial);
    }
    subj_data.push_back(make_pair(addm, trials));
  }

  for (float e : range_e) {
    aDDM addm = aDDM(d, sigma, 1, e, 1, NDT);
    vector<aDDMTrial> trials; 
    for (int i = 0; i < maxTrials; i++) {
      int r_i = distribution(generator); 
      int l_i = distribution(generator);
      aDDMTrial trial = addm.simulateTrial(vals[l_i], vals[r_i], fixationData); 
      trials.push_back(trial); 
    }
    subj_data.push_back(make_pair(addm, trials));
  }

  cout << "id,t,e,t_chosen,e_chosen,n_trials,timestep,statestep,time" << endl; 
  int id = 0; 
  for (const auto& [correct, trials] : subj_data) {
    for (int ts : timeSteps) { 
      for (float ss : stateSteps) {
        for (int tc : trialCnts) {
          vector<aDDMTrial> testTrials(trials.begin(), trials.begin() + tc); 

          auto start = chrono::high_resolution_clock::now(); 
          MLEinfo<aDDM> info_t = aDDM::fitModelMLE(
            testTrials, {d}, {sigma}, test_t, {0}, "thread", false, 1, NDT, ts, ss
          );
          MLEinfo<aDDM> info_e = aDDM::fitModelMLE(
            testTrials, {d}, {sigma}, {1}, test_e, "thread", false, 1, NDT, ts, ss
          ); 
          float t_optimal = info_t.likelihoods[info_t.optimal];
          float e_optimal = info_e.likelihoods[info_e.optimal];
          aDDM optimal = t_optimal < e_optimal ? info_t.optimal : info_e.optimal;

          auto stop = chrono::high_resolution_clock::now(); 
          const chrono::duration<double, milli> ms_time = stop - start; 
          double ms = ms_time.count(); 

          cout << id << "," 
               << correct.theta << ","
               << correct.eta << ","
               << optimal.theta << ","
               << optimal.eta << ","
               << tc << ","
               << ts << ","
               << ss << "," 
               << ms << endl; 
        }
      }
    }
    id++;
  }
}