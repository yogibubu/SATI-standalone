// Coherent KHD Raman. External number states are explicit in both routes;
// TD contracts the entire intermediate harmonic space analytically.
struct HermiteEdge {size_t index;int axis;double factor;};
struct HermiteNode {int axis;size_t parent;double factor;std::vector<HermiteEdge> edges;};
struct PolynomialTerm {size_t index;int first,second;double factor;};
struct ComplexHermitePlan {
 int dimension;size_t budget;std::unordered_map<Key,size_t,Hash> known;
 std::vector<HermiteNode> nodes;std::vector<std::vector<PolynomialTerm>> terms;
 ComplexHermitePlan(int d,size_t maximum):dimension(d),budget(maximum){known[Key(d)]=0;nodes.push_back({-1,0,0.,{}});}
 size_t ensure(const Key& state) {
  std::vector<Key> pending{state};
  while(!pending.empty()) {
   const Key& current=pending.back();if(known.find(current)!=known.end()){pending.pop_back();continue;}
   int axis=0;while(axis<dimension&&!current[axis])axis++;check(axis<dimension,"Invalid complex Hermite state");
   Key parent=current;double inverse=1/std::sqrt(double(parent[axis]));parent[axis]--;
   auto p=known.find(parent);if(p==known.end()){pending.push_back(parent);continue;}
   HermiteNode node{axis,p->second,inverse,{}};bool missing=false;
   for(int k=0;k<dimension;k++)if(parent[k]) {
    double factor=std::sqrt(double(parent[k]))*inverse;parent[k]--;auto dependency=known.find(parent);
    if(dependency==known.end()){pending.push_back(parent);missing=true;break;}
    node.edges.push_back({dependency->second,k,factor});parent[k]++;
   }
   if(missing)continue;check(nodes.size()<budget,"Raman external Hermite cache budget exceeded");
   known.emplace(current,nodes.size());nodes.push_back(std::move(node));pending.pop_back();
  }return known.at(state);
 }
 void add(Key state,bool degree) {
  std::vector<PolynomialTerm> row{{ensure(state),-1,-1,1.}};
  if(degree)for(int k=0;k<dimension;k++)if(state[k]) {
   double first=std::sqrt(double(state[k]));state[k]--;row.push_back({ensure(state),k,-1,first});
   for(int l=0;l<dimension;l++)if(state[l]){double factor=first*std::sqrt(double(state[l]));state[l]--;row.push_back({ensure(state),k,l,factor});state[l]++;}
   state[k]++;
  }terms.push_back(std::move(row));
 }
 C evaluate(const C& matrix,const C& linear,Z constant,const C& left,const C& right,const C& covariance)const {
  C values(nodes.size()),result(terms.size()*9,Z{});values[0]=constant;
  for(size_t i=1;i<nodes.size();i++){const auto& node=nodes[i];Z value=linear[node.axis]*values[node.parent]*node.factor;for(const auto& edge:node.edges)value+=matrix[node.axis*dimension+edge.axis]*values[edge.index]*edge.factor;values[i]=value;}
  for(size_t row=0;row<terms.size();row++)for(const auto& term:terms[row])for(int a=0;a<3;a++)for(int b=0;b<3;b++) {
   Z polynomial;
   if(term.first<0)polynomial=left[a*(dimension+1)]*right[b*(dimension+1)]+covariance[a*3+b];
   else if(term.second<0)polynomial=left[a*(dimension+1)+term.first+1]*right[b*(dimension+1)]+left[a*(dimension+1)]*right[b*(dimension+1)+term.first+1];
   else polynomial=left[a*(dimension+1)+term.first+1]*right[b*(dimension+1)+term.second+1];
   result[row*9+a*3+b]+=term.factor*values[term.index]*polynomial;
  }return result;
 }
};

C row_product(const C& a,const C& b,int rows,int shared,int columns) {
 C result(size_t(rows)*columns);Z one=1.,zero=0.;char no='N';
 zgemm_(&no,&no,&columns,&rows,&shared,&one,b.data(),&columns,a.data(),&shared,&zero,result.data(),&columns);return result;
}
Z optical_constant(const Model& m,int a){return (m.op=="HT"?Z{}:m.mu[a])+(m.op=="FC"?Z{}:m.ht0[a]);}
Z optical_linear(const Model& m,int a,int h){return m.op=="FC"?Z{}:m.ht[a*2*m.n+h];}

C raman_partial_correlation(const Model& m,double tau,const ComplexHermitePlan& plan) {
 int n=m.n,d=2*n,info,nrhs=d;char no='N';C z(n),precision(d*d),covariance(d*d),response(d*d),rbt(d),mean(d),linear(d),matrix(d*d);std::vector<int> indices(n),piv(d);
 for(int i=0;i<n;i++){z[i]=std::exp(Z(0,-m.wf[i]*tau));indices[i]=n+i;}
 for(int i=0;i<d;i++)for(int j=0;j<d;j++) {
  precision[i+j*d]=(i==j?1.:0.);
  if((i<n)!=(j<n)){precision[i+j*d]-=z[i%n]*m.M[(n+i%n)*d+n+j%n];if(i%n==j%n)covariance[i+j*d]=z[i%n];}
  if((i<n)==(j<n))response[i*d+j]=m.M[(j%n)*d+n+i%n];
 }
 zgetrf_(&d,&d,precision.data(),&d,piv.data(),&info);check(info==0,"Raman Gaussian precision failed");
 C product=gaussian_product(m,indices,z);Z logdet=gaussian_logdet(m,product,precision,piv,n,d);
 zgetrs_(&no,&d,&nrhs,precision.data(),&d,piv.data(),covariance.data(),&d,&info);check(info==0,"Raman Gaussian covariance failed");
 C cov(d*d),rt(d*d);for(int i=0;i<d;i++)for(int j=0;j<d;j++){cov[i*d+j]=covariance[i+j*d];rt[i*d+j]=response[j*d+i];}
 C propagated=row_product(cov,response,d,d,d);matrix=row_product(rt,propagated,d,d,d);
 for(int i=0;i<d;i++)rbt[i]=m.r[n+i%n];for(int i=0;i<d;i++)for(int j=0;j<d;j++)mean[i]+=cov[i*d+j]*rbt[j];
 for(int i=0;i<d;i++) {
  linear[i]=m.r[i%n];for(int j=0;j<d;j++){linear[i]+=response[j*d+i]*mean[j];if((i<n)==(j<n))matrix[i*d+j]+=m.M[(i%n)*d+j%n];}
 }
 Z exponent=2*m.logground-.5*logdet;for(int i=0;i<d;i++)exponent+=.5*rbt[i]*mean[i];
 C left(3*(d+1)),right(left.size()),insertion(9);
 for(int a=0;a<3;a++) {
  left[a*(d+1)]=std::conj(optical_constant(m,a));right[a*(d+1)]=optical_constant(m,a);
  for(int i=0;i<n;i++){left[a*(d+1)]+=std::conj(optical_linear(m,a,n+i))*mean[i];right[a*(d+1)]+=optical_linear(m,a,n+i)*mean[n+i];}
  for(int h=0;h<d;h++) {
   for(int i=0;i<n;i++){left[a*(d+1)+h+1]+=std::conj(optical_linear(m,a,n+i))*propagated[i*d+h];right[a*(d+1)+h+1]+=optical_linear(m,a,n+i)*propagated[(n+i)*d+h];}
   if(h<n)left[a*(d+1)+h+1]+=std::conj(optical_linear(m,a,h));else right[a*(d+1)+h+1]+=optical_linear(m,a,h-n);
  }
  for(int b=0;b<3;b++)for(int i=0;i<n;i++)for(int j=0;j<n;j++)insertion[a*3+b]+=std::conj(optical_linear(m,a,n+i))*cov[i*d+n+j]*optical_linear(m,b,n+j);
 }
 return plan.evaluate(matrix,linear,std::exp(exponent),left,right,insertion);
}

C carrier_weights(Z s,double step,size_t count) {
 Z z=s*step;std::array<Z,4> moments{};
 if(std::abs(z)<1.){Z term=1.;for(int order=0;order<40;order++){for(int k=0;k<4;k++)moments[k]+=term/double(order+k+1);term*=z/double(order+1);}}
 else{moments[0]=(std::exp(z)-1.)/z;for(int k=1;k<4;k++)moments[k]=std::exp(z)/z-double(k)*moments[k-1]/z;}
 C result(count,Z{});
 for(size_t interval=0;interval<count-1;interval++) {
  int degree=count==3?2:3;std::vector<int> nodes;
  if(count==3)nodes=interval==0?std::vector<int>{0,1,2}:std::vector<int>{-1,0,1};
  else nodes=interval==0?std::vector<int>{0,1,2,3}:interval==count-2?std::vector<int>{-2,-1,0,1}:std::vector<int>{-1,0,1,2};
  for(int slot=0;slot<=degree;slot++) {
   V polynomial{1.};double denominator=1.;
   for(int k=0;k<=degree;k++)if(k!=slot){V next(polynomial.size()+1,0.);for(size_t j=0;j<polynomial.size();j++){next[j]-=nodes[k]*polynomial[j];next[j+1]+=polynomial[j];}polynomial=std::move(next);denominator*=nodes[slot]-nodes[k];}
   Z coefficient=0.;for(size_t j=0;j<polynomial.size();j++)coefficient+=moments[j]*polynomial[j]/denominator;
   result[size_t(int(interval)+nodes[slot])]+=step*std::exp(s*(interval*step))*coefficient;
  }
 }return result;
}

double raman_state_norm(const Model& m,const Key& state) {
 double result=0.;if(m.op!="HT")for(Z value:m.mu)result+=std::norm(value);
 if(m.op!="FC")for(int a=0;a<3;a++)for(int i=0;i<m.n;i++)result+=std::norm(m.grad[a*m.n+i])*(state[i]+.5);return result;
}
C raman_dipoles(const Model& m,const std::vector<Word>& ground,const std::vector<Word>& excited,Cache& cache) {
 C result(ground.size()*excited.size()*3);
 for(size_t i=0;i<ground.size();i++)for(size_t f=0;f<excited.size();f++) {
  Key state(2*m.n);for(int h=0;h<m.n;h++){state[h]=ground[i].k[h];state[m.n+h]=excited[f].k[h];}
  double overlap=cache.get(state);for(int a=0;a<3;a++)result[(i*excited.size()+f)*3+a]=optical_constant(m,a)*overlap;
  if(m.op!="FC")for(int h=0;h<2*m.n;h++)if(state[h]){double factor=std::sqrt(double(state[h]));state[h]--;double value=cache.get(state);state[h]++;for(int a=0;a<3;a++)result[(i*excited.size()+f)*3+a]+=optical_linear(m,a,h)*factor*value;}
 }return result;
}

int run_native_raman(const Model& first,std::istream& in,const std::string& prefix) {
 auto start=std::chrono::steady_clock::now();int ns=int(take(in));check(ns>0,"Positive Raman surface count required");
 std::vector<std::unique_ptr<Model>> storage;std::vector<const Model*> models{&first};
 for(int k=1;k<ns;k++){storage.push_back(std::make_unique<Model>(in));models.push_back(storage.back().get());}
 int nl=int(take(in)),ni=int(take(in)),nf=int(take(in));size_t nr=size_t(take(in));bool anti=int(take(in))!=0;size_t maximum_pairs=size_t(take(in)),memory=size_t(take(in));double cm=take(in),photon_prefactor=take(in);
 check(nl>0&&ni>0&&nf>0&&nr>0&&nr<=first.trans&&size_t(nl)*nr<=memory/(6*9*sizeof(Z)),"Raman output budget exceeded");
 V laser(nl),gamma(ns),population(ni);for(double& value:laser){value=take(in);check(value>0,"Nonpositive Raman laser");}for(double& value:gamma){value=take(in);check(value>0,"Nonpositive Raman lifetime width");}
 auto read_states=[&](int count){std::vector<Word> states;for(int i=0;i<count;i++){Word row(first.n);for(int h=0;h<first.n;h++){double value=take(in);check(value>=0&&value==std::floor(value)&&value<=std::numeric_limits<unsigned int>::max(),"Invalid Raman occupation");row.k[h]=unsigned(value);row.energy+=value*first.wi[h];}states.push_back(std::move(row));}return states;};
 auto initial=read_states(ni),final=read_states(nf);for(double& value:population){value=take(in);check(value>=0,"Negative Raman population");}
 std::vector<int> ii(nr),ff(nr);for(size_t r=0;r<nr;r++){ii[r]=int(take(in));ff[r]=int(take(in));check(ii[r]>=0&&ii[r]<ni&&ff[r]>=0&&ff[r]<nf,"Invalid Raman transition index");}
 C tensor(size_t(nl)*nr*9,Z{});V bounds(nr,0.),captures(ns,1.),tails(ns,0.);size_t cached=0;
 for(int surface=0;surface<ns;surface++) {
  const Model& m=*models[surface];check(m.n==first.n&&m.method==first.method,"Incompatible Raman electronic surfaces");
  for(int i=0;i<m.n;i++)check(std::abs(m.wi[i]-first.wi[i])<1e-10*(1+first.wi[i]),"Different Raman initial frequencies");for(Z p:m.momentum)check(p==Z{},"Raman requires electric position operators");
  if(m.method=="TI") {
   std::unordered_map<Key,size_t,Hash> lookup;std::vector<Word> ground;auto add=[&](const Word& word){if(lookup.find(word.k)==lookup.end()){lookup[word.k]=ground.size();ground.push_back(word);}};
   for(const auto& state:initial)add(state);for(const auto& state:final)add(state);auto excited=occupations(m,false);
   check(ground.size()<=maximum_pairs/excited.size(),"Raman dipole-pair budget exceeded");check(ground.size()*excited.size()*3*sizeof(Z)<memory/2,"Raman dipole workspace budget exceeded");
   Cache cache(m,std::min(m.integrals,ground.size()*excited.size()));C dipole=raman_dipoles(m,ground,excited,cache);cached+=cache.values.size();V missing(ground.size());
   for(size_t g=0;g<ground.size();g++){double got=0.,total=raman_state_norm(m,ground[g].k);for(size_t f=0;f<excited.size();f++)for(int a=0;a<3;a++)got+=std::norm(dipole[(g*excited.size()+f)*3+a]);missing[g]=std::max(0.,total-got);captures[surface]=std::min(captures[surface],got/std::max(total,1e-300));}
   for(size_t r=0;r<nr;r++) {
    size_t gi=lookup.at(initial[ii[r]].k),gf=lookup.at(final[ff[r]].k);bounds[r]+=cm*std::sqrt(missing[gi]*missing[gf])/gamma[surface]*(anti?2:1);
    for(int l=0;l<nl;l++)for(size_t e=0;e<excited.size();e++) {
     Z resonant=cm/Z(m.origin+excited[e].energy-initial[ii[r]].energy-laser[l],-gamma[surface]);
     Z nonresonant=anti?cm/Z(m.origin+excited[e].energy+laser[l]-final[ff[r]].energy,gamma[surface]):Z{};
     for(int a=0;a<3;a++)for(int b=0;b<3;b++)tensor[(size_t(l)*nr+r)*9+a*3+b]+=resonant*std::conj(dipole[(gf*excited.size()+e)*3+a])*dipole[(gi*excited.size()+e)*3+b]+nonresonant*std::conj(dipole[(gf*excited.size()+e)*3+b])*dipole[(gi*excited.size()+e)*3+a];
    }
   }
  }else {
   ComplexHermitePlan plan(2*m.n,m.integrals);for(size_t r=0;r<nr;r++){Key state(2*m.n);for(int h=0;h<m.n;h++){state[h]=final[ff[r]].k[h];state[m.n+h]=initial[ii[r]].k[h];}plan.add(state,m.op!="FC");}cached+=plan.nodes.size();
   check(size_t(2)*nl*m.points*sizeof(Z)+plan.nodes.size()*sizeof(Z)+size_t(96)*m.n*m.n*sizeof(Z)<memory,"Raman TD workspace budget exceeded");
   C positive(size_t(nl)*m.points),negative(anti?positive.size():0);double step=m.phase*m.dt;
   for(int l=0;l<nl;l++){C weights=carrier_weights(Z(-gamma[surface],laser[l]-m.origin),step,m.points);std::copy(weights.begin(),weights.end(),positive.begin()+l*m.points);if(anti){weights=carrier_weights(Z(-gamma[surface],laser[l]+m.origin),step,m.points);std::copy(weights.begin(),weights.end(),negative.begin()+l*m.points);}}
   tails[surface]=std::exp(-gamma[surface]*step*(m.points-1));for(size_t r=0;r<nr;r++)bounds[r]+=cm*std::sqrt(raman_state_norm(m,initial[ii[r]].k)*raman_state_norm(m,final[ff[r]].k))*tails[surface]/gamma[surface]*(anti?2:1);
   for(size_t t=0;t<m.points;t++) {
    C value=raman_partial_correlation(m,t*step,plan),backward=anti?raman_partial_correlation(m,-double(t)*step,plan):C{};
    for(size_t r=0;r<nr;r++) {
     Z phase=std::exp(Z(0,t*step*initial[ii[r]].energy)),reverse=std::exp(Z(0,-double(t)*step*final[ff[r]].energy));
     for(int l=0;l<nl;l++)for(int a=0;a<3;a++)for(int b=0;b<3;b++)tensor[(size_t(l)*nr+r)*9+a*3+b]+=Z(0,cm)*positive[l*m.points+t]*phase*value[r*9+a*3+b]-(anti?Z(0,cm)*negative[l*m.points+t]*reverse*backward[r*9+b*3+a]:Z{});
    }
   }
  }
 }
 std::ofstream out(prefix+"-raman.tsv");out<<std::setprecision(17)<<"laser_index\ttransition_index\tshift_cm\tscattered_cm\tpopulation\tparallel_m2_sr\tperpendicular_m2_sr";for(int k=0;k<9;k++)out<<"\treal_"<<k<<"\timag_"<<k;out<<'\n';
 for(int l=0;l<nl;l++)for(size_t r=0;r<nr;r++) {
  size_t base=(size_t(l)*nr+r)*9;Z trace=tensor[base]+tensor[base+4]+tensor[base+8];double snorm=0.,anorm=0.;
  for(int a=0;a<3;a++)for(int b=0;b<3;b++){Z sym=.5*(tensor[base+a*3+b]+tensor[base+b*3+a]),asym=.5*(tensor[base+a*3+b]-tensor[base+b*3+a]);if(a==b)sym-=trace/3.;snorm+=std::norm(sym);anorm+=std::norm(asym);}
  double shift=final[ff[r]].energy-initial[ii[r]].energy,scattered=laser[l]-shift,photons=photon_prefactor*laser[l]*std::pow(std::max(scattered,0.),3)*population[ii[r]];
  out<<l<<'\t'<<r<<'\t'<<shift<<'\t'<<scattered<<'\t'<<population[ii[r]]<<'\t'<<photons*(std::norm(trace)/9+2*snorm/15)<<'\t'<<photons*(snorm/10+anorm/6);for(int k=0;k<9;k++)out<<'\t'<<tensor[base+k].real()<<'\t'<<tensor[base+k].imag();out<<'\n';
 }
 std::ofstream meta(prefix+"-result.json");meta<<std::setprecision(17)<<"{\"schema\":\"matrix.trinity.native_raman_execution.v1\",\"method\":\""<<first.method<<"\",\"surfaces\":"<<ns<<",\"modes\":"<<first.n<<",\"cached_integrals\":"<<cached<<",\"intermediate_capture_minimum\":";output_vector(meta,captures);meta<<",\"lifetime_window_tail_factors\":";output_vector(meta,tails);meta<<",\"polarizability_error_bound_au\":";output_vector(meta,bounds);meta<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";return 0;
}
