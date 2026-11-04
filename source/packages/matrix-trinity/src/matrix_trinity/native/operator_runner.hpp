// General Hermitian operator bank. ECD/CPL, tensor spectra and interference
// use the same Gaussian and recurrence as positive electric-dipole spectra.
#include "vendor/Faddeeva.hh"

struct OperatorBank {
 const Model& m;int channels,forms;double direction,gamma,norm=0.;bool export_profile=true;C constant,linear,weights;V rules;std::unique_ptr<GaussianTraceWorkspace> trace;
 struct WeightedPair {int a,b;Z weight;};std::vector<std::vector<WeightedPair>> pairs;
 OperatorBank(const Model& model,std::istream& in):m(model) {
  channels=int(take(in));forms=int(take(in));direction=take(in);gamma=take(in);
  check(channels>0&&forms>0&&(direction==1||direction==-1)&&gamma>=0&&m.sigma+gamma>0,"Invalid operator bank");
  check(size_t(channels)*channels*forms<=16777216,"Observable form budget exceeded");
  C c(channels),h(size_t(channels)*m.n),p(h.size());weights.resize(size_t(forms)*channels*channels);
  for(Z& x:c)x=complex_take(in);for(Z& x:h)x=complex_take(in);for(Z& x:p)x=complex_take(in);for(Z& x:weights)x=complex_take(in);
  std::string output;if(in>>output){check(output=="STICKS_ONLY"&&m.method=="TI","Invalid operator output control");export_profile=false;}
  for(int f=0;f<forms;f++)for(int a=0;a<channels;a++)for(int b=0;b<channels;b++)
   check(std::abs(weight(f,a,b)-std::conj(weight(f,b,a)))<=1e-12*(1+std::abs(weight(f,a,b))),"Non-Hermitian observable form");
  int d=2*m.n;constant=c;linear.assign(size_t(channels)*d,Z{});rules.assign(forms,0.);
  for(int a=0;a<channels;a++)for(int i=0;i<m.n;i++) {
   Z insertion=h[a*m.n+i]+Z(0,1)*p[a*m.n+i];constant[a]+=insertion*m.coordinates[i];
   for(int k=0;k<d;k++)linear[a*d+k]+=insertion*m.coordinates[i+(k+1)*m.n];
   linear[a*d+i]-=Z(0,std::sqrt(2.))*p[a*m.n+i];
  }
  for(int a=0;a<channels;a++)for(int b=0;b<channels;b++) {
   Z gram=c[a]*std::conj(c[b]);
   for(int i=0;i<m.n;i++)gram+=((h[a*m.n+i]-Z(0,1)*p[a*m.n+i])*std::conj(h[b*m.n+i]-Z(0,1)*p[b*m.n+i])*m.q[i]
     +(h[a*m.n+i]+Z(0,1)*p[a*m.n+i])*std::conj(h[b*m.n+i]+Z(0,1)*p[b*m.n+i]))/(2*(1-m.q[i]));
   if(a==b)norm+=gram.real();for(int f=0;f<forms;f++)rules[f]+=(weight(f,a,b)*gram).real();
  }
  trace=std::make_unique<GaussianTraceWorkspace>(m,channels,linear);
  pairs.resize(forms);for(int f=0;f<forms;f++)for(int a=0;a<channels;a++)for(int b=0;b<channels;b++)
   if(weight(f,a,b)!=Z{})pairs[f].push_back({a,b,weight(f,a,b)});
 }
 Z weight(int f,int a,int b)const{return weights[(size_t(f)*channels+a)*channels+b];}
 V contract(const C& amplitude)const {
  V result(forms,0.);for(int f=0;f<forms;f++)for(int a=0;a<channels;a++)for(int b=0;b<channels;b++)
   result[f]+=(weight(f,a,b)*amplitude[a]*std::conj(amplitude[b])).real();return result;
 }
 C correlate(double time)const {
  auto& g=*trace;g.evaluate(direction*time);
  C left=constant,right(channels),result(forms,Z{});
  for(int b=0;b<channels;b++)right[b]=std::conj(constant[b]);
  for(int c:g.live)for(int i=0;i<g.d;i++) {
   left[c]+=g.insertion[c*g.d+i]*g.m1[i];right[c]+=std::conj(g.insertion[c*g.d+i])*g.m2[i];
  }
  // Only nonzero observable coefficients and nonzero linear insertions enter
  // a contraction. This is exact sparsity, with no numerical threshold.
  for(int f=0;f<forms;f++)for(const auto& pair:pairs[f]) {
   Z gram=left[pair.a]*right[pair.b];
   auto slot=std::find(g.live.begin(),g.live.end(),pair.b);
   if(slot!=g.live.end())for(int i=0;i<g.d;i++)
    gram+=g.insertion[pair.a*g.d+i]*g.rhs[i+(slot-g.live.begin()+1)*g.d];
   result[f]+=g.base*pair.weight*gram;
  }return result;
 }

};

double native_voigt(double x,double sigma,double gamma) {
 if(sigma==0)return gamma/(std::acos(-1.)*(x*x+gamma*gamma));
 if(gamma==0)return std::exp(-.5*x*x/(sigma*sigma))/(sigma*std::sqrt(2*std::acos(-1.)));
 return Faddeeva::w(Z(x,gamma)/(sigma*std::sqrt(2.))).real()/(sigma*std::sqrt(2*std::acos(-1.)));
}
void output_vector(std::ostream& stream,const V& values){stream<<'[';for(size_t i=0;i<values.size();i++){if(i)stream<<',';stream<<values[i];}stream<<']';}

int run_operator_bank(const Model& m,std::istream& in,const std::string& prefix) {
 auto start=ExecutionClock::now();OperatorBank bank(m,in);
 double operator_preparation_seconds=elapsed_since(start),correlation_seconds=0.,transform_seconds=0.,export_seconds=0.;
 size_t count=size_t(std::floor((m.high-m.low)/m.grain+.5))+1;check(count>1&&count<=10000000,"Output grid budget exceeded");
 if(!bank.export_profile)count=0;
 V axis(count),density(size_t(bank.forms)*count,0.),captured(bank.forms,0.),tails(bank.forms,0.);double captured_norm=0.;size_t transitions=0,cached=0;
 for(size_t i=0;i<count;i++)axis[i]=m.low+i*m.grain;
 if(m.method=="TI") {
  auto initial=occupations(m,true),final=occupations(m,false);
  check(initial.size()<=m.trans/final.size(),"Operator transition budget exceeded");
  Cache cache(m,std::min(m.integrals,initial.size()*final.size()));
  std::ofstream lines(prefix+"-lines.tsv");lines<<std::setprecision(17)<<"wavenumber_cm\tpopulation\tbank_norm";
  for(int f=0;f<bank.forms;f++)lines<<"\tstrength_"<<f;lines<<"\tinitial_quanta\tfinal_quanta\n";
  for(const auto& i:initial)if(i.pop>0)for(const auto& f:final) {
   Key k(2*m.n);for(int h=0;h<m.n;h++){k[h]=i.k[h];k[m.n+h]=f.k[h];}
   double overlap=cache.get(k);C amplitude=bank.constant;for(Z& x:amplitude)x*=overlap;
   for(int h=0;h<2*m.n;h++)if(k[h]) {
    bool live=false;for(int a=0;a<bank.channels;a++)live=live||bank.linear[a*2*m.n+h]!=Z{};
    if(live){double factor=std::sqrt(double(k[h]));k[h]--;double lower=cache.get(k);k[h]++;for(int a=0;a<bank.channels;a++)amplitude[a]+=bank.linear[a*2*m.n+h]*factor*lower;}
   }
   V strength=bank.contract(amplitude);double norm=0.;for(Z x:amplitude)norm+=std::norm(x);norm*=i.pop;captured_norm+=norm;
   double energy=bank.direction*(m.origin+f.energy-i.energy);lines<<energy<<'\t'<<i.pop<<'\t'<<norm;
   for(int a=0;a<bank.forms;a++){strength[a]*=i.pop;captured[a]+=strength[a];lines<<'\t'<<strength[a];}
   lines<<'\t';for(int h=0;h<m.n;h++)lines<<i.k[h]<<',';lines<<'\t';for(int h=0;h<m.n;h++)lines<<f.k[h]<<',';lines<<'\n';
   transitions++;if(!bank.export_profile)continue;
   auto first=bank.gamma==0?std::lower_bound(axis.begin(),axis.end(),energy-12*m.sigma):axis.begin();
   auto last=bank.gamma==0?std::upper_bound(axis.begin(),axis.end(),energy+12*m.sigma):axis.end();
   for(auto x=first;x!=last;++x){size_t j=x-axis.begin();double profile=native_voigt(*x-energy,m.sigma,bank.gamma);for(int a=0;a<bank.forms;a++)density[size_t(a)*count+j]+=strength[a]*profile;}
  }
  cached=cache.values.size();
 }else {
  size_t length=2*(m.points-1)*m.padding;check(length<=16777216&&size_t(bank.forms)*(length/2+1)<=16777216,"Operator FFT memory budget exceeded");
  std::vector<C> signal(bank.forms,C(length/2+1)),correlations(bank.forms,C(m.points));
  auto correlate_start=ExecutionClock::now();
  for(size_t i=0;i<m.points;i++) {
   C values=bank.correlate(i*m.dt);double tau=m.phase*i*m.dt,damping=std::exp(-bank.gamma*tau-.5*std::pow(m.sigma*tau,2));
   for(int f=0;f<bank.forms;f++) {
    if(i==0)check(std::abs(values[f]-bank.rules[f])<1e-7*std::max(bank.norm,1e-300),"Operator zero-time sum rule failed");
    signal[f][i]=std::conj(values[f])*damping;correlations[f][i]=values[f];
    if(i+16>=m.points)tails[f]=std::max(tails[f],std::abs(signal[f][i])/std::max(bank.norm,1e-300));
   }
  }
  correlation_seconds=elapsed_since(correlate_start);auto trace_export_start=ExecutionClock::now();
  std::ofstream trace(prefix+"-correlation.tsv");trace<<std::setprecision(17)<<"time_fs";for(int f=0;f<bank.forms;f++)trace<<"\treal_"<<f<<"\timag_"<<f;trace<<'\n';
  for(size_t i=0;i<m.points;i++) {
   trace<<i*m.dt;for(int f=0;f<bank.forms;f++)trace<<'\t'<<correlations[f][i].real()<<'\t'<<correlations[f][i].imag();trace<<'\n';
  }
  trace.close();export_seconds=elapsed_since(trace_export_start);auto transform_start=ExecutionClock::now();
  double spacing=m.phase*m.dt,step=2*std::acos(-1.)/(length*spacing);V x(length),transformed(length),ordered(length);
  for(size_t i=0;i<length;i++)x[i]=bank.direction*m.origin+(double(i)-double(length/2))*step;
  for(int f=0;f<bank.forms;f++) {
   pocketfft::c2r<double>({length},{std::ptrdiff_t(sizeof(Z))},{std::ptrdiff_t(sizeof(double))},0,true,signal[f].data(),transformed.data(),spacing/(2*std::acos(-1.)),1);
   for(size_t i=0;i<length;i++){ordered[i]=transformed[(i+length/2)%length];captured[f]+=ordered[i]*step;}
   for(size_t i=0;i<count;i++) {
    check(axis[i]>=x.front()-step*1e-7&&axis[i]<=x.back()+step*1e-7,"Output outside operator Nyquist interval");
    if(axis[i]<=x.front()){density[size_t(f)*count+i]=ordered.front();continue;}
    if(axis[i]>=x.back()){density[size_t(f)*count+i]=ordered.back();continue;}
    auto it=std::lower_bound(x.begin(),x.end(),axis[i]);size_t h=it-x.begin();double r=(axis[i]-x[h-1])/(x[h]-x[h-1]);density[size_t(f)*count+i]=(1-r)*ordered[h-1]+r*ordered[h];
   }
  }
  transform_seconds=elapsed_since(transform_start);
 }
 auto spectrum_export_start=ExecutionClock::now();
 if(bank.export_profile){std::ofstream spectrum(prefix+"-spectrum.tsv");spectrum<<std::setprecision(17)<<"wavenumber_cm";for(int f=0;f<bank.forms;f++)spectrum<<"\tdensity_"<<f;spectrum<<'\n';
  for(size_t i=0;i<count;i++){spectrum<<axis[i];for(int f=0;f<bank.forms;f++)spectrum<<'\t'<<density[size_t(f)*count+i];spectrum<<'\n';}}
 export_seconds+=elapsed_since(spectrum_export_start);double bank_seconds=elapsed_since(start);
 std::ofstream meta(prefix+"-result.json");meta<<std::setprecision(17)<<"{\"certified_LU_logdet\":"<<(m.certified_logdet?"true":"false")<<",\"logdet_absolute_bound\":"<<m.logdet_bound<<",\"schema\":\"matrix.trinity.native_operator_execution.v1\",\"method\":\""<<m.method<<"\",\"modes\":"<<m.n<<",\"channels\":"<<bank.channels<<",\"forms\":"<<bank.forms<<",\"bank_norm_sum_rule\":"<<bank.norm<<",\"captured_bank_norm\":"<<captured_norm<<",\"transitions\":"<<transitions<<",\"cached_integrals\":"<<cached<<",\"sum_rules\":";
 output_vector(meta,bank.rules);meta<<",\"captured_forms\":";output_vector(meta,captured);meta<<",\"time_window_tail_relative_bank_norm\":";output_vector(meta,tails);meta<<",\"seconds\":"<<bank_seconds<<",\"correlation_solver\":\""<<(m.method!="TD"?"not_applicable":bank.trace->half_phase.empty()?"product_LU":"accretive_resolvents")<<"\",\"correlation_evaluations\":{\"reduced\":"<<bank.trace->rank_evaluations<<",\"symmetric\":"<<bank.trace->symmetric_evaluations<<",\"legacy\":"<<bank.trace->legacy_evaluations<<"},\"seconds_including_input_and_spectral_export\":"<<m.input_seconds+m.preparation_seconds+bank_seconds
 <<",\"stage_seconds\":{\"input\":"<<m.input_seconds<<",\"gaussian_preparation\":"<<m.preparation_seconds
 <<",\"operator_preparation\":"<<operator_preparation_seconds<<",\"correlation\":"<<correlation_seconds
 <<",\"transform_and_interpolation\":"<<transform_seconds<<",\"export\":"<<export_seconds
 <<",\"remaining\":"<<bank_seconds-operator_preparation_seconds-correlation_seconds-transform_seconds-export_seconds<<"}}\n";
 return 0;
}
