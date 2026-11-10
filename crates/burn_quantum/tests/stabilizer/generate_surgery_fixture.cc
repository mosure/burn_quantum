// Pinned Stim1.15 independent local surgery oracle; no burn_quantum code linked.
// Initial encoded Bell inputs are prepared by code projection and logical CX,
// independently of burn_quantum's GF(2) unitary encoder.
#include "stim/simulators/tableau_simulator.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using P=std::string;
P join(P a,const P&b){for(size_t i=0;i<a.size();i++)if(b[i]!='_')a[i]=a[i]=='_'?b[i]:'_';return a;}
int main(){
 std::cout<<"{\"stim_version\":\"1.15.0\",\"cases\":[";bool first=true;
 for(size_t d:{3,5,7})for(bool xx:{false,true})for(size_t seed:{0,1,17}){
  size_t n=d*d+(d-1)*(d-1),N=2*n+2;
  auto v=[&](size_t x,size_t y){return x*d+y;};
  auto h=[&](size_t x,size_t y){return d*d+x*(d-1)+y-1;};
  auto single=[&](size_t q,char p){P s(N,'_');s[q]=p;return s;};
  std::vector<P> checks;size_t xc=d*(d-1);
  for(size_t off:{size_t(0),n}){
   for(size_t x=0;x<d;x++)for(size_t y=1;y<d;y++){P s(N,'_');s[off+v(x,y-1)]=s[off+v(x,y)]='X';if(x)s[off+h(x-1,y)]='X';if(x+1<d)s[off+h(x,y)]='X';checks.push_back(s);}
   for(size_t x=0;x+1<d;x++)for(size_t y=0;y<d;y++){P s(N,'_');s[off+v(x,y)]=s[off+v(x+1,y)]='Z';if(y)s[off+h(x,y)]='Z';if(y+1<d)s[off+h(x,y+1)]='Z';checks.push_back(s);}
  }
  std::vector<std::pair<size_t,size_t>> boundary;for(size_t i=0;i+1<d;i++){size_t a=xx?xc+i*d:i;boundary.push_back({a,a+n-1});}
  std::vector<P> merged;for(size_t i=0;i<d;i++){size_t q=xx?v(i,0):v(0,i);merged.push_back(join(single(q,xx?'X':'Z'),single(n+q,xx?'X':'Z')));}
  for(size_t i=0;i<checks.size();i++){bool boundary_check=false;for(auto[a,b]:boundary)boundary_check|=i==a||i==b;if(!boundary_check)merged.push_back(checks[i]);}
  for(auto[a,b]:boundary)merged.push_back(join(checks[a],checks[b]));
  stim::TableauSimulator<64> state(std::mt19937_64(seed),N);
  for(size_t patch=0;patch<2;patch++){
   for(size_t i=0;i<xc;i++)state.postselect_observable(stim::PauliString<64>(checks[patch*(n-1)+i]),false);
   state.safe_do_circuit(stim::Circuit("H "+std::to_string(2*n+patch)));
   for(size_t x=0;x<d;x++)state.safe_do_circuit(stim::Circuit("CX "+std::to_string(2*n+patch)+" "+std::to_string(patch*n+v(x,0))));
  }
  std::string signs;size_t random=0;bool parity=false;P correction(N,'_');
  auto measure=[&](const P&p){stim::PauliString<64> observable(p);int prior=state.peek_observable_expectation(observable);bool bit=prior<0;if(prior==0){bit=(random*17+seed*13+random/3)%7<3;random++;}state.postselect_observable(observable,bit);signs+=bit?'1':'0';return bit;};
  for(size_t round=0;round<d;round++)for(size_t i=0;i<merged.size();i++){bool bit=measure(merged[i]);if(round==0&&i<d)parity^=bit;}
  for(size_t round=0;round<d;round++){
   std::vector<bool> bits;for(auto&p:checks)bits.push_back(measure(p));
   if(round==0){bool bit=false;for(size_t i=0;i<boundary.size();i++){bit^=bits[boundary[i].first];if(bit)correction=join(correction,merged[i+1]);}for(size_t q=0;q<N;q++)if(correction[q]!='_')state.safe_do_circuit(stim::Circuit(std::string(1,correction[q])+" "+std::to_string(q)));}
  }
  if(!first)std::cout<<',';first=false;
  std::cout<<"{\"distance\":"<<d<<",\"basis\":\""<<(xx?"XX":"ZZ")<<"\",\"seed\":"<<seed<<",\"random_measurements\":"<<random<<",\"parity\":"<<(parity?"true":"false")<<",\"record_bits\":\""<<signs<<"\",\"correction\":\""<<correction.substr(0,2*n)<<"\"}";
 }
 std::cout<<"]}\n";
}
