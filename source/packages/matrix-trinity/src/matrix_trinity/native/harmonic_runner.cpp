// Standalone prepared harmonic FC/HT executor. The Gaussian preparation,
// Hermite recurrence and thermal contraction mirror their TRINITY owners.
// PocketFFT supplies the transform; LAPACK supplies dense factorizations.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "vendor/pocketfft_hdronly.h"

using Z=std::complex<double>;
using V=std::vector<double>;
using C=std::vector<Z>;
using ExecutionClock=std::chrono::steady_clock;
double elapsed_since(const ExecutionClock::time_point& start){return std::chrono::duration<double>(ExecutionClock::now()-start).count();}
extern "C" {
void dtrsm_(const char*,const char*,const char*,const char*,const int*,const int*,const double*,const double*,const int*,double*,const int*);
void dpotrf_(const char*,const int*,double*,const int*,int*);
void dpotrs_(const char*,const int*,const int*,const double*,const int*,double*,const int*,int*);
void dgetrf_(const int*,const int*,double*,const int*,int*,int*);
void zgetrf_(const int*,const int*,Z*,const int*,int*,int*);
void zsytrf_(const char*,const int*,Z*,const int*,int*,Z*,const int*,int*);
void zsytrs_(const char*,const int*,const int*,const Z*,const int*,const int*,Z*,const int*,int*);
void zgetrs_(const char*,const int*,const int*,const Z*,const int*,const int*,Z*,const int*,int*);
void zgeev_(const char*,const char*,const int*,Z*,const int*,Z*,Z*,const int*,Z*,const int*,Z*,const int*,double*,int*);
void dsyev_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*);
void zgemm_(const char*,const char*,const int*,const int*,const int*,const Z*,const Z*,const int*,const Z*,const int*,const Z*,Z*,const int*);
}
double take(std::istream& in) {double x;if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("Incomplete/nonfinite prepared input");return x;}
Z complex_take(std::istream& in){double a=take(in),b=take(in);return {a,b};}
void check(bool pass,const char* why){if(!pass)throw std::runtime_error(why);}
// Buffered three-column export avoids repeated locale-aware stream insertion.
// %.17g preserves double precision and the existing TSV numerical contract.
class NumericThreeColumnWriter {
 std::FILE* file;std::vector<char> buffer;
 public:
 NumericThreeColumnWriter(const std::string& path,const char* header):file(std::fopen(path.c_str(),"wb")),buffer(262144) {
  check(file!=nullptr,"Cannot open native numeric output");
  if(std::setvbuf(file,buffer.data(),_IOFBF,buffer.size())!=0){std::fclose(file);file=nullptr;throw std::runtime_error("Cannot buffer native output");}
  if(std::fputs(header,file)<0){std::fclose(file);file=nullptr;throw std::runtime_error("Cannot write native output header");}
 }
 NumericThreeColumnWriter(const NumericThreeColumnWriter&)=delete;
 NumericThreeColumnWriter& operator=(const NumericThreeColumnWriter&)=delete;
 void row(double x,double y,double z) {
  char text[128];int size=std::snprintf(text,sizeof(text),"%.17g\t%.17g\t%.17g\n",x,y,z);
  check(size>0&&size<int(sizeof(text)),"Numeric TSV row formatting failed");
  check(std::fwrite(text,1,size,file)==size_t(size),"Numeric TSV write failed");
 }
 void close(){if(file){std::FILE* stream=file;file=nullptr;check(std::fclose(stream)==0,"Numeric TSV flush failed");}}
 ~NumericThreeColumnWriter(){if(file)std::fclose(file);}
};
struct Model {
 int n,qi,qf,klass;size_t states,trans,integrals,points,padding;
 double temp,gap,sigma,low,high,grain,prefactor,floor,dt,hck,phase;
 std::string method,op;V wi,wf,j,shift,q,M,r,coordinates,gaussian_factor;C mu,grad,momentum,ht0,ht;
 double ground,logground,origin,partition,total,logdet_bound=0.;bool certified_logdet=false;
 double input_seconds=0.,preparation_seconds=0.;
 explicit Model(std::istream& in) {
  auto input_start=ExecutionClock::now();
  std::string version;in>>version>>method>>op;
  check(version=="MATRIX_HARMONIC_EXECUTOR_V1","Unsupported prepared input");
  n=int(take(in));temp=take(in);gap=take(in);sigma=take(in);low=take(in);high=take(in);grain=take(in);prefactor=take(in);
  qi=int(take(in));qf=int(take(in));klass=int(take(in));states=size_t(take(in));trans=size_t(take(in));integrals=size_t(take(in));floor=take(in);
  dt=take(in);points=size_t(take(in));padding=size_t(take(in));hck=take(in);phase=take(in);
  check(n>0&&n<=std::numeric_limits<int>::max()/2&&temp>=0&&sigma>=0&&high>low&&grain>0&&prefactor>0&&dt>0,"Invalid model/grid dimensions");
  check(qi>=0&&qf>=0&&klass>=0&&klass<=n&&states>0&&trans>0&&integrals>0&&floor>=0,"Invalid finite TI budget");
  check(points>=3&&points<=131073&&padding>0&&padding<=64,"Invalid TD grid budget");
  check((method=="TI"||method=="TD")&&(op=="FC"||op=="HT"||op=="FCHT"),"Unsupported route/operator");
  wi.resize(n);wf.resize(n);j.resize(n*n);shift.resize(n);mu.resize(3);grad.resize(3*n);momentum.resize(3*n);
  for(double& x:wi){x=take(in);check(x>0,"Nonpositive initial frequency");}
  for(double& x:wf){x=take(in);check(x>0,"Nonpositive final frequency");}
  for(double& x:j)x=take(in);for(double& x:shift)x=take(in);
  for(Z& x:mu)x=complex_take(in);for(Z& x:grad)x=complex_take(in);for(Z& x:momentum)x=complex_take(in);
  input_seconds=elapsed_since(input_start);auto prepare_start=ExecutionClock::now();
  prepare();preparation_seconds=elapsed_since(prepare_start);
 }
 void prepare() {
  int d=2*n,nrhs=d+1,info;char lower='L';
  V a(n*n),rhs(n*nrhs),L(n*d),k(n),b(n);
  for(int i=0;i<n;i++)k[i]=shift[i]/std::sqrt(wf[i]);
  for(int i=0;i<n;i++)for(int h=0;h<n;h++) {
   double x=(i==h?wi[i]:0.);for(int s=0;s<n;s++)x+=j[s*n+i]*wf[s]*j[s*n+h];a[i+h*n]=x;
  }
  for(int i=0;i<n;i++) {
   for(int s=0;s<n;s++)b[i]-=j[s*n+i]*wf[s]*k[s];rhs[i]=b[i];
   for(int h=0;h<d;h++) {
    double x=h<n?(i==h?std::sqrt(2*wi[i]):0.):j[(h-n)*n+i]*std::sqrt(2*wf[h-n]);
    L[i*d+h]=x;rhs[i+(h+1)*n]=x;
   }
  }
  dpotrf_(&lower,&n,a.data(),&n,&info);check(info==0,"Nonpositive Gaussian precision");
  double logdet=0.;for(int i=0;i<n;i++)logdet+=2*std::log(a[i+i*n]);
  dpotrs_(&lower,&n,&nrhs,a.data(),&n,rhs.data(),&n,&info);check(info==0,"Gaussian solve failed");
  if(method=="TD"&&temp>0&&n>=8) {
   // M+I=W^T W, W=L_chol^-1 L. Rank is n even for a 2n thermal trace.
   gaussian_factor.resize(size_t(n)*d);
   for(int h=0;h<d;h++)for(int i=0;i<n;i++)gaussian_factor[i+h*n]=L[i*d+h];
   char left='L',no_transpose='N',non_unit='N';double one=1.;
   dtrsm_(&left,&lower,&no_transpose,&non_unit,&n,&d,&one,a.data(),&n,gaussian_factor.data(),&n);
  }
  coordinates=rhs;for(int i=0;i<n;i++)for(int h=0;h<nrhs;h++)coordinates[i+h*n]*=std::sqrt(wi[i]);
  V jd=j;std::vector<int> piv(n);dgetrf_(&n,&n,jd.data(),&n,piv.data(),&info);check(info==0,"Singular mode map");
  double jlog=0.;for(int i=0;i<n;i++)jlog+=std::log(std::abs(jd[i+i*n]));
  logground=n*std::log(2.)/2+.5*jlog-.5*logdet;
  for(int i=0;i<n;i++)logground+=.25*std::log(wi[i]*wf[i])-.5*k[i]*wf[i]*k[i]+.5*b[i]*rhs[i];
  ground=std::exp(logground);check(std::isfinite(logground)&&(method!="TI"||ground>0),"Ground overlap underflow/nonfinite Gaussian");
  M.assign(d*d,0.);r.assign(d,0.);ht0.assign(3,Z{});ht.assign(3*d,Z{});
  for(int h=0;h<d;h++) {
   r[h]=h<n?0.:std::sqrt(2*wf[h-n])*k[h-n];
   for(int i=0;i<n;i++)r[h]+=L[i*d+h]*rhs[i];
   for(int l=0;l<d;l++){M[h*d+l]=(h==l?-1.:0.);for(int i=0;i<n;i++)M[h*d+l]+=L[i*d+h]*rhs[i+(l+1)*n];}
  }
  for(int h=0;h<d;h++)for(int l=h+1;l<d;l++){double x=.5*(M[h*d+l]+M[l*d+h]);M[h*d+l]=M[l*d+h]=x;}
  for(int c=0;c<3;c++)for(int i=0;i<n;i++) {
   Z insertion=grad[c*n+i]+Z(0,1)*momentum[c*n+i];
   ht0[c]+=insertion*std::sqrt(wi[i])*rhs[i];
   for(int h=0;h<d;h++)ht[c*d+h]+=insertion*std::sqrt(wi[i])*rhs[i+(h+1)*n];
   ht[c*d+i]-=Z(0,std::sqrt(2.))*momentum[c*n+i];
  }
  q.resize(n);partition=0.;origin=gap;double fc=0.,hp=0.;
  for(Z x:mu)fc+=std::norm(x);
  for(int i=0;i<n;i++) {
   q[i]=temp==0?0.:std::exp(-hck*wi[i]/temp);check(q[i]<1,"Unresolved thermal partition");
   partition+=std::log1p(-q[i]);origin+=(wf[i]-wi[i])/2;
   for(int c=0;c<3;c++)hp+=(std::norm(grad[c*n+i]-Z(0,1)*momentum[c*n+i])*q[i]+
     std::norm(grad[c*n+i]+Z(0,1)*momentum[c*n+i]))/(2*(1-q[i]));
  }
  total=op=="FC"?fc:op=="HT"?hp:fc+hp;
  // For C=D M D, |Tr[(E C E C)^k]| <= sum_j s_j(C)^(2k)
  // by Schatten Holder. Thus |log det(I-B)| <= -sum log(1-s_j(C)^2).
  // If this bound is below pi, the principal LU determinant has exactly the
  // analytic unit-disk logarithm at every time. Otherwise use the spectrum.
  V weighted(d*d),eigen(d),work(3*d+16);int lwork=work.size();char no='N';
  for(int i=0;i<d;i++)for(int j=0;j<d;j++)weighted[i+j*d]=std::sqrt(i<n?q[i]:1.)*M[i*d+j]*std::sqrt(j<n?q[j]:1.);
  dsyev_(&no,&lower,&d,weighted.data(),&d,eigen.data(),work.data(),&lwork,&info);check(info==0,"Gaussian determinant certificate failed");
  certified_logdet=true;for(double value:eigen){if(std::abs(value)>=1-1e-10){certified_logdet=false;break;}logdet_bound-=std::log1p(-value*value);}
  certified_logdet=certified_logdet&&logdet_bound<std::acos(-1.)-1e-8;
 }
};

C gaussian_product(const Model& m,const std::vector<int>& ids,const C& z) {
 int d=ids.size(),full=2*m.n;C p(d*d),column(d*d),row(d*d);char no='N';Z one=1.,zero=0.;
 for(int i=0;i<d;i++)for(int j=0;j<d;j++)p[i+j*d]=z[i]*m.M[ids[i]*full+ids[j]];
 zgemm_(&no,&no,&d,&d,&d,&one,p.data(),&d,p.data(),&d,&zero,column.data(),&d);
 for(int i=0;i<d;i++)for(int j=0;j<d;j++)row[i*d+j]=column[i+j*d];return row;
}
Z gaussian_logdet(const Model& m,const C& product,const C& lu,const std::vector<int>& piv,int d,int lu_dimension=0) {
 if(m.certified_logdet) {
  int ld=lu_dimension?lu_dimension:d;double magnitude=0.,phase=0.;for(int i=0;i<ld;i++){Z value=lu[i+i*ld];magnitude+=std::log(std::abs(value));phase+=std::arg(value);if(piv[i]!=i+1)phase+=std::acos(-1.);}
  return Z(magnitude,std::remainder(phase,2*std::acos(-1.)));
 }
 C matrix(d*d),eigen(d),work(8*d+16);V rwork(2*d);int info,lwork=work.size(),one=1;Z dummy;char no='N';
 for(int i=0;i<d;i++)for(int j=0;j<d;j++)matrix[i+j*d]=product[i*d+j];
 zgeev_(&no,&no,&d,matrix.data(),&d,eigen.data(),&dummy,&one,&dummy,&one,work.data(),&lwork,rwork.data(),&info);check(info==0,"Gaussian determinant spectrum failed");
 Z result=0.;for(Z value:eigen){check(std::abs(value)<=1+1e-7&&std::abs(1.-value)>1e-12,"Unresolved Gaussian contraction");result+=std::log(1.-value);}return result;
}
// Small inline storage is an allocation optimization, not a mode restriction.
// Larger molecules use the same exact keys, equality and recurrence dynamically.
struct Key {
 size_t dimension;std::array<unsigned int,12> local{};std::vector<unsigned int> heap;
 explicit Key(size_t n=0):dimension(n),heap(n>12?n:0,0){}
 unsigned int* data(){return dimension<=12?local.data():heap.data();}
 const unsigned int* data()const{return dimension<=12?local.data():heap.data();}
 unsigned int& operator[](size_t i){return data()[i];}
 unsigned int operator[](size_t i)const{return data()[i];}
 bool operator==(const Key& other)const{return dimension==other.dimension&&std::equal(data(),data()+dimension,other.data());}
};
struct Hash {size_t operator()(const Key& k)const{size_t v=1469598103934665603ull;for(size_t i=0;i<k.dimension;i++){v^=k[i];v*=1099511628211ull;}return v;}};
struct Cache {
 const Model& m;std::unordered_map<Key,double,Hash> values;
 Cache(const Model& x,size_t reserve):m(x){values.reserve(reserve);values[Key(2*m.n)]=m.ground;}
 double get(const Key& s) {
  auto it=values.find(s);if(it!=values.end())return it->second;
  std::vector<Key> pending{s};int d=2*m.n;
  while(!pending.empty()) {
   check(pending.size()<=m.integrals,"TI recurrence cache budget exceeded");
   const Key& current=pending.back();if(values.find(current)!=values.end()){pending.pop_back();continue;}
   int h=0;while(h<d&&!current[h])h++;check(h<d,"Invalid recurrence state");
   double denom=std::sqrt(double(current[h])),value=0.;Key parent=current;parent[h]--;
   if(m.r[h]!=0){auto found=values.find(parent);if(found==values.end()){pending.push_back(parent);continue;}value=m.r[h]*found->second;}
   bool missing=false;
   for(int l=0;l<d;l++)if(parent[l]&&m.M[h*d+l]!=0) {
    double factor=std::sqrt(double(parent[l]));parent[l]--;auto found=values.find(parent);
    if(found==values.end()){pending.push_back(parent);missing=true;break;}
    value+=m.M[h*d+l]*factor*found->second;parent[l]++;
   }
   if(missing)continue;
   check(values.size()<m.integrals,"TI recurrence cache budget exceeded");values.emplace(current,value/denom);pending.pop_back();
  }
  return values.at(s);
 }
};
struct Word {Key k;double energy=0,pop=1;explicit Word(size_t n):k(n){}};
std::vector<Word> occupations(const Model& m,bool initial) {
 std::vector<Word> out;Word current(m.n);int limit=initial?(m.temp==0?0:m.qi):m.qf;
 const V& w=initial?m.wi:m.wf;
 auto fill=[&](auto&& self,int i,int remaining,int active)->void{
  if(i==m.n){check(out.size()<m.states,"TI occupation budget exceeded");out.push_back(current);return;}
  for(int v=0;v<=remaining;v++)if(active+(v!=0)<=m.klass){current.k[i]=v;self(self,i+1,remaining-v,active+(v!=0));}
 };
 fill(fill,0,limit,0);
 for(auto& row:out)for(int i=0;i<m.n;i++){row.energy+=w[i]*row.k[i];if(initial)row.pop*=std::pow(m.q[i],row.k[i])*(1-m.q[i]);}
 std::stable_sort(out.begin(),out.end(),[](const Word& a,const Word& b){return a.energy<b.energy;});return out;
}
struct Result {V density,energy_density;double computed=0,reported=0;size_t transitions=0,cache=0;double tail=0;};
Result ti(const Model& m,const V& axis,const std::string& prefix) {
 auto initial=occupations(m,true),final=occupations(m,false);Result result;result.density.assign(axis.size(),0);result.energy_density=result.density;
 Cache cache(m,std::min(m.integrals,std::min(m.trans,initial.size()*final.size())));
  std::ofstream lines(prefix+"-lines.tsv");lines<<std::setprecision(17)<<"wavenumber_cm\tweighted_strength_au2\tpopulation\tFC_overlap\tFC_strength_au2\tHT_strength_au2\tinterference_au2\tstrength_au2\tinitial_quanta\tfinal_quanta\n";
 for(const auto& i:initial)if(i.pop>0)for(const auto& f:final) {
  if(result.transitions>=m.trans)goto finished;
  Key k(2*m.n);for(int h=0;h<m.n;h++){k[h]=i.k[h];k[m.n+h]=f.k[h];}double overlap=cache.get(k);C fc(3),ht(3);double fs=0,hs=0,combined=0;
  for(int c=0;c<3;c++){fc[c]=m.mu[c]*overlap;ht[c]=m.ht0[c]*overlap;}
  for(int h=0;h<2*m.n;h++)if(k[h]) {
   bool live=false;for(int c=0;c<3;c++)live=live||m.ht[c*2*m.n+h]!=Z{};
   if(live){double factor=std::sqrt(double(k[h]));k[h]--;double lower=cache.get(k);k[h]++;
    for(int c=0;c<3;c++)ht[c]+=m.ht[c*2*m.n+h]*factor*lower;}
  }
  for(int c=0;c<3;c++){fs+=std::norm(fc[c]);hs+=std::norm(ht[c]);combined+=std::norm(fc[c]+ht[c]);}
  double strength=i.pop*(m.op=="FC"?fs:m.op=="HT"?hs:combined);double energy=m.origin+f.energy-i.energy;
  result.computed+=strength;result.transitions++;if(strength<=m.floor)continue;result.reported+=strength;
  lines<<energy<<'\t'<<strength<<'\t'<<i.pop<<'\t'<<overlap<<'\t'<<fs<<'\t'<<hs<<'\t'<<combined-fs-hs<<'\t'<<(m.op=="FC"?fs:m.op=="HT"?hs:combined)<<'\t';
  for(int h=0;h<m.n;h++)lines<<i.k[h]<<',';lines<<'\t';for(int h=0;h<m.n;h++)lines<<f.k[h]<<',';lines<<'\n';
  // Certified Gaussian tail below 4e-33 of retained strength; full recovery
  // is counted before this profile-only, twelve-sigma support restriction.
  auto first=std::lower_bound(axis.begin(),axis.end(),energy-12*m.sigma);
  auto last=std::upper_bound(axis.begin(),axis.end(),energy+12*m.sigma);
  double factor=strength/(m.sigma*std::sqrt(2*std::acos(-1.)));
  for(auto it=first;it!=last;++it){size_t j=it-axis.begin();double x=(*it-energy)/m.sigma;double shape=factor*std::exp(-.5*x*x);result.density[j]+=shape;result.energy_density[j]+=energy*shape;}
 }
 finished:result.cache=cache.values.size();return result;
}
// One workspace per correlation stream: all geometry, active indices and
// matrix buffers are reused. The small-matrix route is exact pivoted LU;
// its crossover changes allocation/BLAS overhead, never the physical model.
struct GaussianTraceWorkspace {
 const Model& m;int d,channels,nrhs;std::vector<int> ids,live,piv;V population,frequency,real_matrix,source;
 C z,product,a,p,rhs,m1,m2,insertion;Z base;char no='N';
 C half_phase,plus_factor,plus_rhs,symmetric_work;std::vector<int> plus_piv;int symmetric_lwork=0;
 C rank_factor,weighted_minus,weighted_plus,rank_rhs_minus,rank_rhs_plus,right_insertions,minus_weight,plus_weight;
 size_t rank_evaluations=0,symmetric_evaluations=0,legacy_evaluations=0;
 V root_population;
 GaussianTraceWorkspace(const Model& model,int count,const C& linear):m(model),channels(count) {
  int full=2*m.n;
  for(int i=0;i<full;i++){double q=i<m.n?m.q[i]:1.;if(q!=0){ids.push_back(i);population.push_back(q);frequency.push_back(i<m.n?-m.wi[i]:m.wf[i-m.n]);}}
  d=ids.size();real_matrix.resize(d*d);source.resize(d);insertion.resize(size_t(channels)*d);
  for(int i=0;i<d;i++){source[i]=m.r[ids[i]];for(int j=0;j<d;j++)real_matrix[i+j*d]=m.M[ids[i]*full+ids[j]];}
  for(int c=0;c<channels;c++) {
   bool active=false;for(int i=0;i<d;i++){insertion[c*d+i]=linear[c*full+ids[i]];active=active||insertion[c*d+i]!=Z{};}
   if(active)live.push_back(c);
  }
  nrhs=1+live.size();z.resize(d);product.resize(d*d);a.resize(d*d);p.resize(d*d);
  rhs.resize(size_t(d)*nrhs);m1.resize(d);m2.resize(d);piv.resize(d);
  if(m.method=="TD"&&d>8&&d>m.n) {
   half_phase.resize(d);plus_factor.resize(d*d);plus_rhs.resize(size_t(d)*nrhs);plus_piv.resize(d);
   root_population.resize(d);for(int i=0;i<d;i++)root_population[i]=std::sqrt(population[i]);
   if(!m.gaussian_factor.empty()&&m.n>=8&&d>m.n) {
    rank_factor.resize(size_t(m.n)*d);weighted_minus.resize(rank_factor.size());weighted_plus.resize(rank_factor.size());
    rank_rhs_minus.resize(size_t(m.n)*nrhs);rank_rhs_plus.resize(size_t(m.n)*nrhs);
    right_insertions.resize(size_t(d)*nrhs);minus_weight.resize(d);plus_weight.resize(d);
    for(int j=0;j<d;j++) {
     for(int i=0;i<m.n;i++)rank_factor[i+j*m.n]=m.gaussian_factor[i+ids[j]*m.n];
     right_insertions[j]=source[j];
     for(size_t c=0;c<live.size();c++)right_insertions[j+(c+1)*d]=std::conj(insertion[live[c]*d+j]);
    }
   }
   int query=-1,info;Z size;char lower='L';
   zsytrf_(&lower,&d,a.data(),&d,piv.data(),&size,&query,&info);
   check(info==0,"Symmetric Gaussian workspace query failed");
   symmetric_lwork=std::max(d,int(size.real()));symmetric_work.resize(symmetric_lwork);
  }
 }
 void factor_and_solve() {
  if(d>8) {
   int info;zgetrf_(&d,&d,a.data(),&d,piv.data(),&info);check(info==0,"TD Gaussian factorization failed");
   // The logarithm is taken from this factorization before any buffer changes.
  } else {
   for(int k=0;k<d;k++) {
    int pivot=k;double magnitude=std::norm(a[k+k*d]);
    for(int i=k+1;i<d;i++)if(std::norm(a[i+k*d])>magnitude){pivot=i;magnitude=std::norm(a[i+k*d]);}
    check(magnitude>0,"Singular TD Gaussian factorization");piv[k]=pivot+1;
    if(pivot!=k)for(int j=0;j<d;j++)std::swap(a[k+j*d],a[pivot+j*d]);
    for(int i=k+1;i<d;i++){a[i+k*d]/=a[k+k*d];for(int j=k+1;j<d;j++)a[i+j*d]-=a[i+k*d]*a[k+j*d];}
   }
  }
 }
 void solve() {
  if(d>8) {int info;zgetrs_(&no,&d,&nrhs,a.data(),&d,piv.data(),rhs.data(),&d,&info);check(info==0,"TD Gaussian solve failed");}
  else for(int c=0;c<nrhs;c++) {
   Z* column=rhs.data()+c*d;
   for(int k=0;k<d;k++)if(piv[k]!=k+1)std::swap(column[k],column[piv[k]-1]);
   for(int i=0;i<d;i++)for(int k=0;k<i;k++)column[i]-=a[i+k*d]*column[k];
   for(int i=d-1;i>=0;i--){for(int k=i+1;k<d;k++)column[i]-=a[i+k*d]*column[k];column[i]/=a[i+i*d];}
  }
 }
 // With E=diag(sqrt(z)), S=E M E is complex symmetric and P=E S E^-1.
 // (I-P^2)^-1(I+P)z*r = E (I-S)^-1 E*r; the two old means coincide.
 // Covariance insertions use 1/2[(I-S)^-1+(I+S)^-1]. This removes P^2
 // and uses pivoted complex-symmetric solves, without dropping populations.
 // Since ||S||<1, I+/-S have positive-definite real part. Positive-real
 // LDL^T blocks give the analytic (unwrapped) log determinant, including
 // cases outside the older global LU-phase bound. Unsafe pivots fall back.
 Z symmetric_factor_logdet(C& factor,std::vector<int>& pivot,int dimension=0) {
  int order=dimension?dimension:d;char lower='L';int info;
  zsytrf_(&lower,&order,factor.data(),&order,pivot.data(),symmetric_work.data(),&symmetric_lwork,&info);
  check(info>=0,"Invalid symmetric Gaussian factorization argument");
  if(info>0)return Z(std::numeric_limits<double>::quiet_NaN(),0.);
  Z result=0.;
  for(int k=0;k<order;) {
   Z determinant=factor[k+k*order];
   if(pivot[k]<0) {
    check(k+1<order&&pivot[k+1]==pivot[k],"Invalid symmetric pivot block");
    double x=determinant.real(),y=factor[k+1+(k+1)*order].real(),off=factor[k+1+k*order].real();
    if(!(x>0.&&y>0.&&x*y>off*off))return Z(std::numeric_limits<double>::quiet_NaN(),0.);
    determinant=determinant*factor[k+1+(k+1)*order]-factor[k+1+k*order]*factor[k+1+k*order];k+=2;
   }else {
    if(!(determinant.real()>0.))return Z(std::numeric_limits<double>::quiet_NaN(),0.);k++;
   }
   if(!(std::abs(determinant)>0.))return Z(std::numeric_limits<double>::quiet_NaN(),0.);
   result+=Z(std::log(std::abs(determinant)),std::arg(determinant));
  }return result;
 }
 void symmetric_solve(const C& factor,const std::vector<int>& pivot,int order,int columns,Z* right) {
  if(!columns)return;
  int info;char lower='L';
  zsytrs_(&lower,&order,&columns,factor.data(),&order,pivot.data(),right,&order,&info);
  check(info==0,"Symmetric Gaussian solve failed");
 }
 // Woodbury reduces each accretive thermal resolvent from d to rank n.
 // A_sigma=I+sigma*S=D_sigma+sigma*U^T U, D_sigma=I-sigma*z.
 // Near a diagonal pole use the full symmetric solver; this crossover only
 // changes numerical conditioning, never populations or the Hamiltonian.
 bool evaluate_rank_resolvents(double time) {
  if(rank_factor.empty())return false;
  Z diagonal_logdet=0.;
  for(int j=0;j<d;j++) {
   z[j]=population[j]*std::exp(Z(0,-m.phase*time*frequency[j]));
   Z minus=1.+z[j],plus=1.-z[j];
   if(std::min(std::abs(minus),std::abs(plus))<1e-3)return false;
   minus_weight[j]=z[j]/minus;plus_weight[j]=z[j]/plus;
   diagonal_logdet+=std::log(minus*plus);
   for(int i=0;i<m.n;i++) {
    weighted_minus[i+j*m.n]=rank_factor[i+j*m.n]*minus_weight[j];
    weighted_plus[i+j*m.n]=rank_factor[i+j*m.n]*plus_weight[j];
   }
  }
  int rank=m.n;char trans='T';Z one=1.,negative=-1.,zero=0.;
  zgemm_(&no,&trans,&rank,&rank,&d,&negative,weighted_minus.data(),&rank,rank_factor.data(),&rank,&zero,a.data(),&rank);
  zgemm_(&no,&trans,&rank,&rank,&d,&one,weighted_plus.data(),&rank,rank_factor.data(),&rank,&zero,plus_factor.data(),&rank);
  for(int i=0;i<rank;i++){a[i+i*rank]+=1.;plus_factor[i+i*rank]+=1.;}
  zgemm_(&no,&no,&rank,&nrhs,&d,&one,weighted_minus.data(),&rank,right_insertions.data(),&d,&zero,rank_rhs_minus.data(),&rank);
  zgemm_(&no,&no,&rank,&nrhs,&d,&one,weighted_plus.data(),&rank,right_insertions.data(),&d,&zero,rank_rhs_plus.data(),&rank);
  Z logdet=diagonal_logdet+symmetric_factor_logdet(a,piv,rank)+symmetric_factor_logdet(plus_factor,plus_piv,rank);
  if(!std::isfinite(logdet.real()))return false;
  int plus_columns=live.size();
  symmetric_solve(a,piv,rank,nrhs,rank_rhs_minus.data());
  symmetric_solve(plus_factor,plus_piv,rank,plus_columns,rank_rhs_plus.data()+rank);
  zgemm_(&trans,&no,&d,&nrhs,&rank,&one,rank_factor.data(),&rank,rank_rhs_minus.data(),&rank,&zero,rhs.data(),&d);
  if(plus_columns)zgemm_(&trans,&no,&d,&plus_columns,&rank,&one,rank_factor.data(),&rank,rank_rhs_plus.data()+rank,&rank,&zero,plus_rhs.data()+d,&d);
  Z exponent=2*m.logground+m.partition-.5*logdet;
  for(int i=0;i<d;i++) {
   m1[i]=m2[i]=minus_weight[i]*(source[i]+rhs[i]);exponent+=source[i]*m1[i];
   for(size_t c=0;c<live.size();c++) {
    size_t slot=i+(c+1)*d;Z value=right_insertions[slot];
    rhs[slot]=.5*(minus_weight[i]*(value+rhs[slot])+plus_weight[i]*(value-plus_rhs[slot]));
   }
  }
  base=std::exp(exponent);rank_evaluations++;return true;
 }
 bool evaluate_symmetric(double time) {
  if(evaluate_rank_resolvents(time))return true;
  for(int i=0;i<d;i++)half_phase[i]=root_population[i]*std::exp(Z(0,-.5*m.phase*time*frequency[i]));
  for(int j=0;j<d;j++)for(int i=j;i<d;i++) {
   Z value=half_phase[i]*real_matrix[i+j*d]*half_phase[j];
   a[i+j*d]=(i==j?1.:0.)-value;plus_factor[i+j*d]=(i==j?1.:0.)+value;
  }
  for(int i=0;i<d;i++) {
   rhs[i]=half_phase[i]*source[i];
   for(size_t c=0;c<live.size();c++) {
    Z value=half_phase[i]*std::conj(insertion[live[c]*d+i]);rhs[i+(c+1)*d]=value;plus_rhs[i+(c+1)*d]=value;
   }
  }
  Z logdet=symmetric_factor_logdet(a,piv)+symmetric_factor_logdet(plus_factor,plus_piv);
  if(!std::isfinite(logdet.real()))return false;
  int plus_columns=live.size();
  symmetric_solve(a,piv,d,nrhs,rhs.data());
  symmetric_solve(plus_factor,plus_piv,d,plus_columns,plus_rhs.data()+d);
  Z exponent=2*m.logground+m.partition-.5*logdet;
  for(int i=0;i<d;i++) {
   m1[i]=m2[i]=half_phase[i]*rhs[i];exponent+=source[i]*m1[i];
   for(size_t c=0;c<live.size();c++) {
    size_t slot=i+(c+1)*d;rhs[slot]=.5*half_phase[i]*(rhs[slot]+plus_rhs[slot]);
   }
  }base=std::exp(exponent);symmetric_evaluations++;return true;
 }
 void evaluate(double time) {
  if(!half_phase.empty()&&evaluate_symmetric(time))return;
  legacy_evaluations++;
  for(int i=0;i<d;i++)z[i]=population[i]*std::exp(Z(0,-m.phase*time*frequency[i]));
  for(int j=0;j<d;j++)for(int i=0;i<d;i++)p[i+j*d]=z[i]*real_matrix[i+j*d];
  if(d>8){Z one=1.,zero=0.;zgemm_(&no,&no,&d,&d,&d,&one,p.data(),&d,p.data(),&d,&zero,a.data(),&d);}
  else for(int j=0;j<d;j++)for(int i=0;i<d;i++){Z value=0.;for(int k=0;k<d;k++)value+=p[i+k*d]*p[k+j*d];a[i+j*d]=value;}
  for(int j=0;j<d;j++)for(int i=0;i<d;i++){product[i*d+j]=a[i+j*d];a[i+j*d]=(i==j?1.:0.)-a[i+j*d];}
  for(int i=0;i<d;i++) {
   rhs[i]=z[i]*source[i];for(int j=0;j<d;j++)rhs[i]+=p[i+j*d]*z[j]*source[j];
   for(size_t c=0;c<live.size();c++)rhs[i+(c+1)*d]=z[i]*std::conj(insertion[live[c]*d+i]);
  }
  factor_and_solve();Z logdet=gaussian_logdet(m,product,a,piv,d);solve();
  Z exponent=2*m.logground+m.partition-.5*logdet;
  for(int i=0;i<d;i++) {
   m1[i]=rhs[i];m2[i]=z[i]*source[i];for(int j=0;j<d;j++)m2[i]+=p[i+j*d]*rhs[j];
   exponent+=.5*source[i]*(m1[i]+m2[i]);
  }base=std::exp(exponent);
 }
};
struct DipoleTraceWorkspace {
 const Model& m;GaussianTraceWorkspace gaussian;double fc=0.;
 DipoleTraceWorkspace(const Model& model):m(model),gaussian(model,3,
     model.op=="FC"?C(size_t(3)*2*model.n,Z{}):model.ht){for(Z value:m.mu)fc+=std::norm(value);}
 std::array<Z,3> evaluate(double time) {
  auto& g=gaussian;g.evaluate(time);Z ht=0.,cross=0.;
  if(m.op!="FC")for(int c=0;c<3;c++) {
   Z left=m.ht0[c],right=std::conj(m.ht0[c]);
   for(int i=0;i<g.d;i++){left+=g.insertion[c*g.d+i]*g.m1[i];right+=std::conj(g.insertion[c*g.d+i])*g.m2[i];}
   for(size_t slot=0;slot<g.live.size();slot++)if(g.live[slot]==c)
    for(int i=0;i<g.d;i++)ht+=g.insertion[c*g.d+i]*g.rhs[i+(slot+1)*g.d];
   ht+=left*right;cross+=m.mu[c]*right+left*std::conj(m.mu[c]);
  }
  return {g.base*fc,g.base*ht,g.base*cross};
 }
};
std::array<Z,3> correlation(const Model& m,double time) {return DipoleTraceWorkspace(m).evaluate(time);}
Result td(const Model& m,const V& axis,const std::string& prefix) {
 size_t length=2*(m.points-1)*m.padding;check(length<=16777216,"FFT allocation budget exceeded");
 DipoleTraceWorkspace workspace(m);C positive(length/2+1);NumericThreeColumnWriter trace(prefix+"-correlation.tsv","time_fs\treal\timag\n");double reference=0,tail=0;
 for(size_t i=0;i<m.points;i++) {
  auto parts=workspace.evaluate(i*m.dt);Z value=m.op=="FC"?parts[0]:m.op=="HT"?parts[1]:parts[0]+parts[1]+parts[2];
  double tau=m.phase*i*m.dt;positive[i]=std::conj(value)*std::exp(-.5*std::pow(m.sigma*tau,2));
  if(i==0){reference=std::abs(value);check(std::abs(value-m.total)<1e-7*std::max(1.,m.total),"TD zero-time sum rule failed");}
  if(i+16>=m.points)tail=std::max(tail,std::abs(positive[i]));trace.row(i*m.dt,value.real(),value.imag());
 }
 trace.close();check(tail/std::max(reference,1e-300)<1e-6,"Unconverged TD damping window");
 V density(length);double spacing=m.phase*m.dt;
 pocketfft::c2r<double>({length},{std::ptrdiff_t(sizeof(Z))},{std::ptrdiff_t(sizeof(double))},0,true,positive.data(),density.data(),spacing/(2*std::acos(-1.)),1);
 V shifted(length),x(length);double step=2*std::acos(-1.)/(length*spacing);double integral=0.;
 for(size_t i=0;i<length;i++){shifted[i]=density[(i+length/2)%length];x[i]=m.origin+(double(i)-double(length/2))*step;integral+=shifted[i]*step;}
 Result result;result.computed=integral;result.reported=integral;result.tail=tail/std::max(reference,1e-300);result.density.resize(axis.size());result.energy_density.resize(axis.size());
 for(size_t i=0;i<axis.size();i++){
  auto it=std::lower_bound(x.begin(),x.end(),axis[i]);check(it!=x.begin()&&it!=x.end(),"Output window outside TD Nyquist interval");size_t h=it-x.begin();double f=(axis[i]-x[h-1])/(x[h]-x[h-1]);
  result.density[i]=(1-f)*shifted[h-1]+f*shifted[h];result.energy_density[i]=axis[i]*result.density[i];
 }
 return result;
}
#include "operator_runner.hpp"
#include "raman_runner.hpp"
#ifndef MATRIX_BUILD_SHARED
int main(int argc,char** argv) {
 try {
  check(argc==3,"Usage: matrix-vibronic prepared-model.dat output-prefix");auto start=std::chrono::steady_clock::now();std::ifstream in(argv[1]);Model m(in);
  std::string extension;if(in>>extension){if(extension=="MATRIX_RAMAN_V1")return run_native_raman(m,in,argv[2]);check(extension=="MATRIX_OPERATOR_BANK_V1","Unsupported operator extension");return run_operator_bank(m,in,argv[2]);}
  check(m.sigma>0,"Positive Gaussian broadening required for dipole input");
  size_t count=size_t(std::floor((m.high-m.low)/m.grain+.5))+1;check(count>1&&count<=10000000,"Output grid budget exceeded");V axis(count);
  for(size_t i=0;i<count;i++)axis[i]=m.low+i*m.grain;
  Result result=m.method=="TI"?ti(m,axis,argv[2]):td(m,axis,argv[2]);
  NumericThreeColumnWriter out(std::string(argv[2])+"-spectrum.tsv","wavenumber_cm\tmolar_epsilon\tstrength_density\n");
  for(size_t i=0;i<count;i++)out.row(axis[i],m.prefactor*result.energy_density[i],result.density[i]);out.close();
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::ofstream meta(std::string(argv[2])+"-result.json");meta<<std::setprecision(17)<<"{\"certified_LU_logdet\":"<<(m.certified_logdet?"true":"false")<<",\"logdet_absolute_bound\":"<<m.logdet_bound<<",\"schema\":\"matrix.trinity.native_harmonic_execution.v1\",\"method\":\""<<m.method<<"\",\"operator\":\""<<m.op<<"\",\"modes\":"<<m.n<<",\"total_strength\":"<<m.total<<",\"computed_strength\":"<<result.computed<<",\"reported_strength\":"<<result.reported<<",\"fraction\":"<<(m.total==0?1:result.reported/m.total)<<",\"transitions\":"<<result.transitions<<",\"cached_integrals\":"<<result.cache<<",\"seconds_including_input_and_spectral_export\":"<<seconds<<",\"tail_ratio\":"<<result.tail<<",\"Gaussian_profile_tail_bound_fraction\":"<<std::erfc(12/std::sqrt(2.))<<",\"TD_positive_points\":"<<m.points<<",\"FFT_padding_factor\":"<<m.padding<<"}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
#endif
