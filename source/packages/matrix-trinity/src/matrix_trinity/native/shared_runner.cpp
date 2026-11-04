#define MATRIX_BUILD_SHARED 1
#include "harmonic_runner.cpp"
#include "selection_kernels.cpp"

extern "C" int matrix_prepare_operator_gaussian(const char* path,size_t modes,size_t channels,size_t forms,
        double* matrix,double* linear,double* log_ground,Z* constant,Z* insertion,double* rules,double* norm,double* q) {
 try {
  std::ifstream input(path);Model m(input);std::string extension;input>>extension;
  check(extension=="MATRIX_OPERATOR_BANK_V1","Operator bank preparation input required");OperatorBank bank(m,input);
  check(size_t(m.n)==modes&&size_t(bank.channels)==channels&&size_t(bank.forms)==forms,"Operator preparation buffer dimensions differ");
  std::copy(m.M.begin(),m.M.end(),matrix);std::copy(m.r.begin(),m.r.end(),linear);*log_ground=m.logground;
  std::copy(bank.constant.begin(),bank.constant.end(),constant);std::copy(bank.linear.begin(),bank.linear.end(),insertion);
  std::copy(bank.rules.begin(),bank.rules.end(),rules);*norm=bank.norm;std::copy(m.q.begin(),m.q.end(),q);return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}

// Dipole APIs share the same preparation, contraction and exact-key cache as
// the standalone executor. Buffers are owned by the public Python workflow.
extern "C" int matrix_prepare_dipole_gaussian(const char* path,size_t modes,
        double* matrix,double* linear,double* log_ground,Z* constant,Z* insertion) {
 try {
  std::ifstream input(path);Model m(input);check(size_t(m.n)==modes,"Dipole preparation dimension differs");
  std::copy(m.M.begin(),m.M.end(),matrix);std::copy(m.r.begin(),m.r.end(),linear);*log_ground=m.logground;
  std::copy(m.ht0.begin(),m.ht0.end(),constant);std::copy(m.ht.begin(),m.ht.end(),insertion);return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_dipole_amplitudes(size_t rows,size_t dimension,const int64_t* states,
        const int64_t* ids,const double* values,const Z* mu,const Z* ht0,const Z* ht,double* output) {
 try {
  for(size_t row=0;row<rows;row++) {
   double overlap=values[ids[row*(dimension+1)]],fc=0.,h=0.,cross=0.,total=0.;
   for(size_t axis=0;axis<3;axis++) {
    Z f=mu[axis]*overlap,v=ht0[axis]*overlap;
    for(size_t k=0;k<dimension;k++) {
     int64_t id=ids[row*(dimension+1)+k+1];
     if(id>=0)v+=ht[axis*dimension+k]*std::sqrt(double(states[row*dimension+k]))*values[id];
    }
    fc+=std::norm(f);h+=std::norm(v);cross+=2.*std::real(f*std::conj(v));total+=std::norm(f+v);
   }
   output[row*5]=overlap;output[row*5+1]=fc;output[row*5+2]=h;output[row*5+3]=cross;output[row*5+4]=total;
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_fill_occupations(size_t rows,size_t modes,size_t tasks,size_t columns,
        const int64_t* totals,const int64_t* classes,const int64_t* offsets,const int64_t* binomial,int64_t* output) {
 try {
  auto unrank=[&](int64_t rank,size_t size,size_t count) {
   std::vector<size_t> result(count);size_t start=0;
   for(size_t slot=0;slot<count;slot++)for(size_t v=start;v<size;v++) {
    int64_t ways=binomial[(size-v-1)*columns+count-slot-1];
    if(rank<ways){result[slot]=v;start=v+1;break;}rank-=ways;
   }return result;
  };
  for(size_t row=1;row<rows;row++) {
   size_t task=std::upper_bound(offsets,offsets+tasks,int64_t(row))-offsets-1;
   size_t total=totals[task],count=classes[task];int64_t partitions=binomial[(total-1)*columns+count-1];
   int64_t rank=row-offsets[task];auto active=unrank(rank/partitions,modes,count),cuts=unrank(rank%partitions,total-1,count-1);
   int64_t previous=-1;
   for(size_t slot=0;slot<count;slot++){int64_t cut=slot<count-1?cuts[slot]:total-1;output[row*modes+active[slot]]=cut-previous;previous=cut;}
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_dipole_correlation(const char* path,size_t count,const double* times,
        double direction,Z* output,double* logdet_bound,int* certified) {
 try {
  std::ifstream input(path);Model m(input);*logdet_bound=m.logdet_bound;*certified=m.certified_logdet;
  DipoleTraceWorkspace workspace(m);
  for(size_t i=0;i<count;i++) {
   auto values=workspace.evaluate(direction*times[i]);
   for(size_t k=0;k<3;k++)output[k*count+i]=values[k];
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_dipole_fft(size_t points,double spacing,const Z* signal,double* output) {
 try {
  check(points>=3&&points<=131073&&spacing>0.,"Invalid native Fourier grid");size_t length=2*(points-1);
  C positive(points);V density(length);
  for(size_t k=0;k<3;k++) {
   for(size_t i=0;i<points;i++)positive[i]=std::conj(signal[k*points+i]);
   pocketfft::c2r<double>({length},{std::ptrdiff_t(sizeof(Z))},{std::ptrdiff_t(sizeof(double))},0,true,
     positive.data(),density.data(),spacing/(2*std::acos(-1.)),1);
   for(size_t i=0;i<length;i++)output[k*length+i]=density[(i+length/2)%length];
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_rate_moments(size_t count,const double* energies,const double* strengths,
        double sigma,double gamma,double* einstein,double* density,double* peak) {
 try {
  check(sigma>=0&&gamma>=0&&sigma+gamma>0,"Nonzero Voigt width required");*einstein=0.;*density=0.;
  auto profile=[&](double x){return sigma==0?gamma/(std::acos(-1.)*(x*x+gamma*gamma)):
    std::real(Faddeeva::w(Z(x,gamma)/(sigma*std::sqrt(2.))))/(sigma*std::sqrt(2*std::acos(-1.)));};
  *peak=profile(0.);
  for(size_t i=0;i<count;i++) {
   if(energies[i]>0)*einstein+=std::pow(energies[i],3)*strengths[i];
   *density+=profile(energies[i])*strengths[i];
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}

extern "C" int matrix_overlap_grid(size_t dimension,size_t entries,const double* matrix,const double* linear,
        const int64_t* dimensions,const int64_t* strides,double ground,double* output) {
 try {
  output[0]=ground;std::vector<int64_t> counts(dimension);
  for(size_t index=1;index<entries;index++) {
   for(size_t k=0;k<dimension;k++)counts[k]=(index/strides[k])%dimensions[k];
   size_t j=0;while(j<dimension&&counts[j]==0)j++;check(j<dimension,"Invalid overlap grid strides");
   double denominator=std::sqrt(double(counts[j]));size_t previous=index-strides[j];counts[j]--;
   double value=linear[j]*output[previous];
   for(size_t k=0;k<dimension;k++)if(counts[k])value+=matrix[j*dimension+k]*std::sqrt(double(counts[k]))*output[previous-strides[k]];
   output[index]=value/denominator;
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_operator_rows(size_t rows,size_t dimension,size_t channels,const int64_t* states,
        const int64_t* ids,const double* values,const Z* constant,const Z* linear,Z* output) {
 try {
  for(size_t row=0;row<rows;row++)for(size_t c=0;c<channels;c++) {
   Z value=constant[c]*values[ids[row*(dimension+1)]];
   for(size_t k=0;k<dimension;k++) {
    int64_t id=ids[row*(dimension+1)+k+1];
    if(id>=0)value+=linear[c*dimension+k]*std::sqrt(double(states[row*dimension+k]))*values[id];
   }output[row*channels+c]=value;
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_contract_orders(size_t rows,size_t channels,const int64_t* offsets,const int64_t* indices,
        const Z* coefficients,const Z* moments,Z* output) {
 try {
  for(size_t row=0;row<3*rows;row++)for(size_t c=0;c<channels;c++) {
   Z value=0.;for(int64_t edge=offsets[row];edge<offsets[row+1];edge++)value+=coefficients[edge]*moments[indices[edge]*channels+c];
   output[row*channels+c]=value;
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}

// One boundary crossing owns conditional selection, exact recurrence,
// amplitudes, thermal weights, energies and fixed-order batch reductions.
int dipole_ti_batch_impl(matrix_native_selection::PackedRequest* request,void* handle,
        size_t batch,size_t pending,int64_t* states,const Z* mu,const Z* ht0,const Z* ht,
        const double* q,const double* wi,const double* wf,double origin,double direction,double floor,int op,
        double* amplitudes,double* columns,double* totals,double* class_weights,int64_t* class_counts,
        size_t* tree_counts,double* timings,void* context) {
 try {
  auto& r=*request;auto& tree=*static_cast<matrix_native_selection::Tree*>(handle);
  auto start=std::chrono::steady_clock::now();
  size_t selected=pending?pending:tree.batch(batch,states),n=r.dimension/2;
  tree_counts[0]=selected;tree_counts[1]=tree.r.allocated;tree_counts[2]=tree.r.visited;
  r.states=states;r.rows=selected;r.maximum_depth=2;
  for(size_t row=0;row<selected;row++){size_t degree=0;for(size_t k=0;k<r.dimension;k++)degree+=states[row*r.dimension+k];r.maximum_depth=std::max(r.maximum_depth,degree+2);}
  auto selected_at=std::chrono::steady_clock::now();
  if((context?matrix_packed_context_evaluate(context,&r):matrix_packed_evaluate(&r))<0)return -1;
  auto evaluated_at=std::chrono::steady_clock::now();
  if(matrix_dipole_amplitudes(r.completed,r.dimension,states,r.ids,r.values,mu,ht0,ht,amplitudes)<0)return -1;
  std::fill(totals,totals+5,0.);std::fill(class_weights,class_weights+n+1,0.);std::fill(class_counts,class_counts+n+1,0);
  double partition=1.;for(size_t k=0;k<n;k++)partition*=1.-q[k];
  for(size_t row=0;row<r.completed;row++) {
   double population=partition,energy=origin;size_t cls=0;
   for(size_t k=0;k<n;k++) {
    int64_t i=states[row*r.dimension+k],f=states[row*r.dimension+n+k];
    if(i)population*=std::pow(q[k],double(i));energy+=wf[k]*f-wi[k]*i;cls+=f!=0;
   }
   double strength=amplitudes[row*5+op],weighted=strength*population;
   columns[row*4]=direction*energy;columns[row*4+1]=population;columns[row*4+2]=strength;columns[row*4+3]=weighted;
   totals[0]+=weighted;if(weighted>floor)totals[1]+=weighted;
   for(size_t c=0;c<3;c++)totals[c+2]+=amplitudes[row*5+c+1]*population;
   class_weights[cls]+=weighted;class_counts[cls]++;
  }
  auto end=std::chrono::steady_clock::now();timings[0]=std::chrono::duration<double>(selected_at-start).count();
  timings[1]=std::chrono::duration<double>(evaluated_at-selected_at).count();timings[2]=std::chrono::duration<double>(end-evaluated_at).count();return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}

extern "C" int matrix_dipole_ti_batch(matrix_native_selection::PackedRequest* request,void* handle,
        size_t batch,size_t pending,int64_t* states,const Z* mu,const Z* ht0,const Z* ht,
        const double* q,const double* wi,const double* wf,double origin,double direction,double floor,int op,
        double* amplitudes,double* columns,double* totals,double* class_weights,int64_t* class_counts,
        size_t* tree_counts,double* timings) {return dipole_ti_batch_impl(request,handle,batch,pending,states,mu,ht0,ht,q,wi,wf,origin,direction,floor,op,amplitudes,columns,totals,class_weights,class_counts,tree_counts,timings,nullptr);}

extern "C" int matrix_dipole_ti_batch_context(matrix_native_selection::PackedRequest* request,void* handle,
        size_t batch,size_t pending,int64_t* states,const Z* mu,const Z* ht0,const Z* ht,
        const double* q,const double* wi,const double* wf,double origin,double direction,double floor,int op,
        double* amplitudes,double* columns,double* totals,double* class_weights,int64_t* class_counts,
        size_t* tree_counts,double* timings,void* context) {return dipole_ti_batch_impl(request,handle,batch,pending,states,mu,ht0,ht,q,wi,wf,origin,direction,floor,op,amplitudes,columns,totals,class_weights,class_counts,tree_counts,timings,context);}

extern "C" int matrix_prepare_gaussian_geometry(const char* path,size_t modes,double* matrix,
        double* linear,double* log_ground,double* coordinates) {
 try {
  std::ifstream input(path);Model m(input);check(size_t(m.n)==modes,"Gaussian geometry dimension differs");
  std::copy(m.M.begin(),m.M.end(),matrix);std::copy(m.r.begin(),m.r.end(),linear);*log_ground=m.logground;
  std::copy(m.coordinates.begin(),m.coordinates.end(),coordinates);return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
extern "C" int matrix_insert_dipole(size_t modes,const double* coordinates,const Z* gradient,
        const Z* momentum,Z* constant,Z* linear) {
 try {
  size_t d=2*modes;std::fill(constant,constant+3,Z{});std::fill(linear,linear+3*d,Z{});
  for(size_t c=0;c<3;c++)for(size_t i=0;i<modes;i++) {
   Z insertion=gradient[c*modes+i]+Z(0,1)*momentum[c*modes+i];constant[c]+=insertion*coordinates[i];
   for(size_t k=0;k<d;k++)linear[c*d+k]+=insertion*coordinates[i+(k+1)*modes];
   linear[c*d+i]-=Z(0,std::sqrt(2.))*momentum[c*modes+i];
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
