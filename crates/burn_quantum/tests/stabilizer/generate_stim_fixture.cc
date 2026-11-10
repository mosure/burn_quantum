// Standalone independent reference generator for Stim 1.15.0 (Apache-2.0).
// Build against the pinned source's non-Python objects, excluding src/main.cc.
#include "stim/simulators/tableau_simulator.h"
#include <iostream>
#include <string>
#include <vector>
int main() {
    std::vector<std::vector<std::string>> circuits = {
        {"H 0", "CX 0 1", "CX 1 2", "CX 2 3", "S 1", "S 3"},
        {"H 0 1 2 3", "CZ 0 1", "CZ 1 2", "CZ 2 3", "S 0", "H 2", "CX 3 1"},
    };
    std::cout << "{\"stim_version\":\"1.15.0\",\"qubits\":4,\"cases\":[";
    bool first=true;
    for (const auto &gates: circuits) for (bool conditioned: {false,true}) {
        stim::TableauSimulator<64> state(std::mt19937_64(0),4);
        for (const auto &gate:gates) state.safe_do_circuit(stim::Circuit(gate));
        if (!first) std::cout << ','; first=false;
        std::cout << "{\"gates\":[";
        for (size_t i=0;i<gates.size();i++) { if(i) std::cout << ',';std::cout << '"' << gates[i] << '"'; }
        std::cout << "],\"postselections\":[";
        if (conditioned) {
            bool initial=true;
            for (const auto &p: {"XZY_","_XXY","Y_ZX"}) {
                stim::PauliString<64> observable(p);
                int expected=state.peek_observable_expectation(observable);
                bool negative=expected == -1 || (expected==0 && !initial);
                state.postselect_observable(observable,negative);
                if(!initial) std::cout << ',';initial=false;
                std::cout << "[\"" << p << "\"," << (negative?"true":"false") << ']';
            }
        }
        std::cout << "],\"expectations\":[";
        for (size_t code=0;code<256;code++) {
            size_t k=code;std::string p;
            for(size_t q=0;q<4;q++){p.push_back("_XYZ"[k%4]);k/=4;}
            if(code) std::cout << ',';
            std::cout << "[\"" << p << "\"," << int(state.peek_observable_expectation(stim::PauliString<64>(p))) << ']';
        }
        std::cout << "]}";
    }
    std::cout << "]}\n";
}
