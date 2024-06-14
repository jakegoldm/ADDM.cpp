#include <addm/cpp_toolbox.h>
#include <vector> 
#include <random>
#include <map>
#include <string>
#include <chrono>

int TIMESTEP = 10; 
int NDT = 0; 

vector<float> range_d = {0.004, 0.006, 0.008}; 
vector<float> range_s = {0.04, 0.06, 0.08}; 
vector<float> range_t = {0.3, 0.5, 0.7};

vector<float> test_d = {0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01};
vector<float> test_s = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};
vector<float> test_t = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};

vector<float> vals = {0, 1, 2, 3};
vector<int> timeSteps = {10, 5}; 
vector<float> stateSteps = {0.1, 0.01}; 
vector<int> trialCnts = {100, 500, 1000};

string EXP_DATA = "/central/groups/rnl/jgoldman/ADDM.cpp/data/expdata.csv";
string FIX_DATA = "/central/groups/rnl/jgoldman/ADDM.cpp/data/fixations.csv"; 


int main(int argc, char **argv) {
  assert(argc == 2);
  int id = atoi(argv[1]);
  assert(id >= 0 && id < 27);

  map<int, vector<aDDMTrial>> data = loadDataFromCSV(EXP_DATA, FIX_DATA);
  FixationData fixationData = getEmpiricalDistributions(data);

  mt19937 generator(random_device{}());
  uniform_int_distribution<size_t> distribution(0, vals.size() - 1);

  vector<pair<aDDM, vector<aDDMTrial>>> subj_data; 
  int maxTrials = *max_element(trialCnts.begin(), trialCnts.end()); 

  for (float d : range_d) {
    for (float s : range_s) {
      for (float t : range_t) {
        aDDM addm = aDDM(d, s, t, 0, 1, NDT);
        vector<aDDMTrial> trials;
        for (int i = 0; i < maxTrials; i++) {
          int r_i = distribution(generator);
          int l_i = distribution(generator);
          aDDMTrial trial = addm.simulateTrial(vals[l_i], vals[r_i], fixationData);
          trials.push_back(trial);
        }
        subj_data.push_back(make_pair(addm, trials));
      }
    }
  }

  cout << "id,d,s,t,d_chosen,s_chosen,t_chosen,n_trials,timestep,statestep,time" << endl; 
  aDDM correct = subj_data[id].first; 
  vector<aDDMTrial> trials = subj_data[id].second; 
  for (int ts : timeSteps) { 
    for (float ss : stateSteps) {
      for (int tc : trialCnts) {
        vector<aDDMTrial> testTrials(trials.begin(), trials.begin() + tc); 

        auto start = chrono::high_resolution_clock::now(); 
        MLEinfo<aDDM> info = aDDM::fitModelMLE(
          testTrials, test_d, test_s, test_t, {0}, "thread", false, 1, NDT, ts, ss
        );
        auto stop = chrono::high_resolution_clock::now(); 
        const chrono::duration<double, milli> ms_time = stop - start; 
        double ms = ms_time.count(); 

        cout << id << "," 
              << correct.d << ","
              << correct.sigma << ","
              << correct.theta << ","
              << info.optimal.d << ","
              << info.optimal.sigma << ","
              << info.optimal.theta << ","
              << tc << ","
              << ts << ","
              << ss << "," 
              << ms << endl; 
      }
    }
  }
}