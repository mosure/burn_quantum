// Stim 1.15.0 independent rotated six-ring sequential Bell-measurement oracle.
// Compile like generate_stim_fixture.cc; no burn_quantum implementation is linked.
#include "stim/simulators/tableau_simulator.h"
#include <iostream>
#include <string>
int main() {
    std::cout << "{\"stim_version\":\"1.15.0\",\"side\":2,\"cases\":[";
    for (size_t seed=0;seed<4;seed++) {
        if(seed) std::cout << ',';
        stim::TableauSimulator<64> state(std::mt19937_64(seed),48);
        for(size_t cell=0;cell<8;cell++) {
            for(size_t q=0;q<6;q++) state.safe_do_circuit(stim::Circuit("H "+std::to_string(6*cell+q)));
            for(size_t q=0;q<6;q++) state.safe_do_circuit(stim::Circuit("CZ "+std::to_string(6*cell+q)+" "+std::to_string(6*cell+(q+1)%6)));
        }
        std::cout << "[";size_t count=0;
        for(size_t x=0;x<2;x++) for(size_t y=0;y<2;y++) for(size_t z=0;z<2;z++) {
            const size_t cell=4*x+2*y+z;const size_t ports[]={1,2,0};
            for(size_t axis=0;axis<3;axis++) {
                size_t other=4*(x^(axis==0))+2*(y^(axis==1))+(z^(axis==2));
                size_t a=6*cell+ports[axis],b=6*other+ports[axis]+3;
                for(char basis: {'X','Z'}) {
                    std::string p(48,'_');p[a]=basis;p[b]=basis;
                    stim::PauliString<64> observable(p);
                    int prior=state.peek_observable_expectation(observable);
                    bool negative=prior<0 || (prior==0 && ((count*17+seed*13+count/3)%7<3));
                    state.postselect_observable(observable,negative);
                    if(count)std::cout<<',';
                    std::cout<<"{\"sites\":["<<a<<','<<b<<"],\"basis\":\""<<basis<<"\",\"prior\":"<<prior<<",\"negative\":"<<(negative?"true":"false")<<'}';
                    count++;
                }
            }
        }
        std::cout<<']';
    }
    std::cout<<"]}\n";
}
