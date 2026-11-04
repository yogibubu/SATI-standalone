// Native adaptive-TI kernels over the existing checkpoint-owned typed arrays.
// No checkpoint schema, key packing, proposal or stopping criterion is changed.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "vendor/pocketfft_hdronly.h"

namespace matrix_native_selection {
thread_local std::string error;
void require(bool pass,const char* message){if(!pass)throw std::runtime_error(message);}
using Item=std::tuple<double,int64_t,int64_t>;
struct TreeRequest {
 size_t dimension,length,budget,capacity;const double* costs;const int64_t *values,*parents,*original;
 int64_t *ancestors,*axes,*ranks;size_t allocated,visited;const double* heap_cost;
 const int64_t *heap_node,*heap_axis;size_t heap_size,modes,qi,qf,maximum_class;
};
struct Tree {
 TreeRequest r;std::vector<Item> heap;std::vector<int64_t> selection,physical,state;std::vector<uint8_t> assigned;
 explicit Tree(const TreeRequest& input):r(input),selection(r.dimension),physical(r.dimension),state(r.dimension),assigned(r.dimension) {
  require(r.dimension==2*r.modes&&r.length>1&&r.capacity>=2*r.budget+2,"Invalid native tree dimensions");
  for(size_t i=0;i<r.heap_size;i++)heap.emplace_back(r.heap_cost[i],r.heap_node[i],r.heap_axis[i]);
  std::make_heap(heap.begin(),heap.end(),std::greater<Item>());
 }
 double cost(size_t axis,int64_t parent,size_t rank)const{return r.costs[(axis*r.length+parent)*r.length+rank];}
 void push(double cost,int64_t node,int64_t axis){heap.emplace_back(cost,node,axis);std::push_heap(heap.begin(),heap.end(),std::greater<Item>());}
 void decode(int64_t node) {
  std::fill(assigned.begin(),assigned.end(),0);std::fill(selection.begin(),selection.end(),0);
  while(node){size_t axis=r.axes[node];if(!assigned[axis]){selection[axis]=r.ranks[node];assigned[axis]=1;}node=r.ancestors[node];}
  update_physical(0);
 }
 // Appending changes only this axis and its descendants: proposal parents
 // precede their children. Keep the previously decoded prefix intact.
 void update_physical(size_t first) {
  for(size_t axis=first;axis<r.dimension;axis++){int64_t parent=r.parents[axis]<0?0:physical[r.parents[axis]];physical[axis]=r.values[(axis*r.length+parent)*r.length+selection[axis]];}
 }
 size_t allocate(int64_t parent,int64_t axis,int64_t rank){require(r.allocated+1<r.capacity,"Conditional tree allocation budget exceeded");size_t node=++r.allocated;r.ancestors[node]=parent;r.axes[node]=axis;r.ranks[node]=rank;return node;}
 size_t batch(size_t count,int64_t* output) {
  size_t kept=0;
  while(!heap.empty()&&r.visited<r.budget&&kept<count) {
   std::pop_heap(heap.begin(),heap.end(),std::greater<Item>());auto [value,node,append]=heap.back();heap.pop_back();decode(node);
   if(append>=0) {
    int64_t base=node,last=base==0?-1:r.axes[base],parent=r.parents[append]<0?0:physical[r.parents[append]];
    double previous=cost(append,parent,1),next=INFINITY;int64_t next_axis=-1;
    for(size_t j=last+1;j<r.dimension;j++){int64_t pv=r.parents[j]<0?0:physical[r.parents[j]];double penalty=cost(j,pv,1);if((penalty>previous||(penalty==previous&&int64_t(j)>append))&&(penalty<next||(penalty==next&&(next_axis<0||int64_t(j)<next_axis)))){next_axis=j;next=penalty;}}
    if(next_axis>=0&&std::isfinite(next))push(value-previous+next,base,next_axis);
    node=allocate(base,append,1);selection[append]=1;update_physical(size_t(append));
   }
   r.visited++;for(size_t j=0;j<r.dimension;j++)state[r.original[j]]=physical[j];
   size_t qi=0,qf=0,ci=0,cf=0;for(size_t j=0;j<r.modes;j++){qi+=state[j];qf+=state[r.modes+j];ci+=state[j]!=0;cf+=state[r.modes+j]!=0;}
   if(qi<=r.qi&&qf<=r.qf&&ci<=r.maximum_class&&cf<=r.maximum_class){std::copy(state.begin(),state.end(),output+kept*r.dimension);kept++;}
   int64_t last=node==0?-1:r.axes[node];
   if(last>=0){size_t rank=r.ranks[node];int64_t pv=r.parents[last]<0?0:physical[r.parents[last]];if(rank+1<r.length&&std::isfinite(cost(last,pv,rank+1))){size_t child=allocate(node,last,rank+1);push(value+cost(last,pv,rank+1)-cost(last,pv,rank),child,-1);}}
   double next=INFINITY;int64_t next_axis=-1;
   for(size_t j=last+1;j<r.dimension;j++){int64_t pv=r.parents[j]<0?0:physical[r.parents[j]];double penalty=cost(j,pv,1);if(penalty<next){next_axis=j;next=penalty;}}
   if(next_axis>=0&&std::isfinite(next))push(value+next,node,next_axis);
  }return kept;
 }
};
struct PackedRequest {
 const int64_t* states;size_t rows,dimension;uint64_t* keys;size_t words,maximum_integrals;
 int64_t* slots;size_t slot_count;double* values;size_t count,edge_count;
 const double *matrix,*linear;const uint8_t* ht;unsigned bits;uint64_t mask;
 size_t maximum_depth;int64_t* ids;size_t completed;int exhausted;
};
// Hash and probing are separate so a recurrence frame can retain its hash
// while dependencies are inserted. Probe again on insertion: nested insertions
// may have occupied the formerly empty slot. Key equality remains exact.
uint64_t append_hash(uint64_t hash,uint64_t word) {
 uint64_t mixed=word+0x9e3779b97f4a7c15ull;
 mixed=(mixed^(mixed>>30))*0xbf58476d1ce4e5b9ull;
 mixed=(mixed^(mixed>>27))*0x94d049bb133111ebull;mixed^=mixed>>31;
 return (hash^mixed)*1099511628211ull;
}
uint64_t key_hash(const uint64_t* key,size_t words) {
 uint64_t hash=1469598103934665603ull;
 for(size_t k=0;k<words;k++)hash=append_hash(hash,key[k]);
 return hash;
}
std::pair<int64_t,size_t> find_hashed(const uint64_t* key,uint64_t hash,const uint64_t* keys,size_t words,const int64_t* slots,size_t capacity) {
 size_t slot=hash&(capacity-1);
 while(slots[slot]>=0){int64_t node=slots[slot];const uint64_t* candidate=keys+node*words;
  if(candidate[words-1]==key[words-1]&&std::equal(key,key+words-1,candidate))return {node,slot};
  slot=(slot+1)&(capacity-1);
 }return {-1,slot};
}
std::pair<int64_t,size_t> find(const uint64_t* key,const uint64_t* keys,size_t words,const int64_t* slots,size_t capacity) {
 return find_hashed(key,key_hash(key,words),keys,words,slots,capacity);
}
// A persistent executor is used only for sufficiently populated degree levels.
// Each node owns its ordered sum; threads never write the same cache value.
class LevelExecutor {
 size_t workers,generation=0,remaining=0;bool stopping=false;
 std::mutex mutex;std::condition_variable ready,finished;
 std::function<void(size_t,size_t)> job;std::vector<std::thread> threads;
 public:
 explicit LevelExecutor(size_t count):workers(count) {
  require(count>=1&&count<=32,"Recurrence workers must be between 1 and 32");
  try {for(size_t index=1;index<workers;index++)threads.emplace_back([this,index]{
   size_t seen=0;std::unique_lock<std::mutex> lock(mutex);
   while(true) {
    ready.wait(lock,[&]{return stopping||generation!=seen;});if(stopping)return;
    seen=generation;lock.unlock();job(index,workers);lock.lock();
    if(--remaining==0)finished.notify_one();
   }
  });}catch(...){stop();throw;}
 }
 void stop() {
  {std::lock_guard<std::mutex> lock(mutex);stopping=true;}ready.notify_all();
  for(auto& thread:threads)if(thread.joinable())thread.join();
 }
 ~LevelExecutor(){stop();}
 void run(std::function<void(size_t,size_t)> action) {
  if(workers==1){action(0,1);return;}
  {std::lock_guard<std::mutex> lock(mutex);job=std::move(action);remaining=workers-1;generation++;}
  ready.notify_all();job(0,workers);
  std::unique_lock<std::mutex> lock(mutex);finished.wait(lock,[&]{return remaining==0;});
 }
 size_t size()const{return workers;}
};
struct RecurrenceEdge {int64_t node;double factor;};
struct Workspace {
 PackedRequest r{};bool levels;
 std::unique_ptr<LevelExecutor> executor;
 std::vector<uint64_t> stack,parent,key,hashes,prefixes,lowering;
 std::vector<size_t> word,shift,active,active_count,degree;
 std::vector<int64_t> cursor,axis;
 std::vector<double> denominator,sums,pending,roots;
 // Tags are transient metadata; slots/keys/values retain their checkpoint ABI.
 std::vector<uint32_t> compact_index;std::vector<uint64_t> wide_index;
 unsigned index_bits=24;uint64_t index_mask=(uint64_t(1)<<24)-1;
 std::vector<std::vector<RecurrenceEdge>> frame_edges;
 std::vector<RecurrenceEdge> edges;
 std::vector<size_t> offsets;
 std::vector<std::vector<size_t>> buckets;
 size_t first_new=0,lookups=0,rejections=0,comparisons=0,batches=0,level_nodes=0,parallel_levels=0,growths=0;
 explicit Workspace(bool use_levels=false,size_t workers=1):levels(use_levels),executor(new LevelExecutor(workers)){}
 void bind(const PackedRequest& request) {
  bool geometry=r.dimension!=request.dimension||r.bits!=request.bits||r.words!=request.words;
  bool rebuild=geometry||r.keys!=request.keys||r.slots!=request.slots||r.slot_count!=request.slot_count||r.count!=request.count||
   (r.maximum_integrals<=0xffffff)!=(request.maximum_integrals<=0xffffff);
  r=request;
  if(geometry) {
   word.resize(r.dimension);shift.resize(r.dimension);lowering.resize(r.dimension);key.resize(r.words);parent.resize(r.words);
   for(size_t j=0;j<r.dimension;j++){word[j]=(j*r.bits)/64;shift[j]=(j*r.bits)%64;lowering[j]=uint64_t(1)<<shift[j];}
  }
  size_t depth=r.maximum_depth;
  if(geometry||cursor.size()<depth) {
   growths++;stack.resize(depth*r.words);hashes.resize(depth);prefixes.resize(depth*(r.words+1));
   active.resize(depth*r.dimension);active_count.resize(depth);degree.resize(depth);
   cursor.resize(depth);axis.resize(depth);denominator.resize(depth);sums.resize(depth);pending.resize(depth);
   size_t previous=roots.size();roots.resize(depth);for(size_t n=previous;n<depth;n++)roots[n]=std::sqrt(double(n));
   if(levels)frame_edges.resize(depth);
  }
  if(rebuild) {
   index_bits=r.maximum_integrals<=0xffffff?24:56;index_mask=(uint64_t(1)<<index_bits)-1;
   require(r.maximum_integrals<=index_mask,"Packed index capacity exceeds addressable storage");
   if(index_bits==24){compact_index.assign(r.slot_count,0);wide_index.clear();}
   else{wide_index.assign(r.slot_count,0);compact_index.clear();}
   for(size_t node=0;node<r.count;node++) {
    uint64_t hash=key_hash(r.keys+node*r.words,r.words);size_t slot=hash&(r.slot_count-1);
    while(entry(slot))slot=(slot+1)&(r.slot_count-1);
    require(r.slots[slot]==int64_t(node),"Packed workspace requires valid checkpoint slots");put(slot,node,hash);
   }
  }

  batches++;first_new=r.count;edges.clear();offsets.clear();offsets.push_back(0);
  for(auto& bucket:buckets)bucket.clear();
 }
 template<unsigned Bits> uint64_t indexed(size_t slot)const {
  if constexpr(Bits==24)return compact_index[slot];else return wide_index[slot];
 }
 uint64_t entry(size_t slot)const{return index_bits==24?indexed<24>(slot):indexed<56>(slot);}
 size_t index_bytes()const{return compact_index.size()*sizeof(uint32_t)+wide_index.size()*sizeof(uint64_t);}
 static uint8_t tag(uint64_t hash){return uint8_t(hash>>56);}
 template<unsigned Bits> void put_index(size_t slot,size_t node,uint64_t hash) {
  uint64_t value=(uint64_t(tag(hash))<<Bits)|(node+1);
  if constexpr(Bits==24)compact_index[slot]=uint32_t(value);else wide_index[slot]=value;
 }
 void put(size_t slot,size_t node,uint64_t hash) {
  if(index_bits==24)put_index<24>(slot,node,hash);else put_index<56>(slot,node,hash);
 }
 template<unsigned Bits> std::pair<int64_t,size_t> lookup(const uint64_t* current,uint64_t hash) {
  lookups++;size_t slot=hash&(r.slot_count-1);uint8_t wanted=tag(hash);uint64_t packed;
  while((packed=indexed<Bits>(slot))) {
   if(uint8_t(packed>>Bits)==wanted) {
    comparisons++;int64_t node=int64_t((packed&((uint64_t(1)<<Bits)-1))-1);const uint64_t* candidate=r.keys+node*r.words;
    if(candidate[r.words-1]==current[r.words-1]&&std::equal(current,current+r.words-1,candidate))return {node,slot};
   }else rejections++;
   slot=(slot+1)&(r.slot_count-1);
  }return {-1,slot};
 }
 template<bool LevelMode> void edge(int64_t depth,double factor,int64_t node) {
  if constexpr(LevelMode)frame_edges[depth].push_back({node,factor});else sums[depth]+=factor*r.values[node];
 }
 void evaluate_levels() {
  for(auto& bucket:buckets)if(!bucket.empty()) {
   auto evaluate=[&](size_t worker,size_t workers) {
    size_t begin=bucket.size()*worker/workers,end=bucket.size()*(worker+1)/workers;
    for(size_t index=begin;index<end;index++) {
     size_t local=bucket[index];double sum=0.;
     for(size_t e=offsets[local];e<offsets[local+1];e++)sum+=edges[e].factor*r.values[edges[e].node];
     r.values[first_new+local]=sum;
    }
   };
   if(executor->size()>1&&bucket.size()>=256){parallel_levels++;executor->run(evaluate);}else evaluate(0,1);
  }
  level_nodes+=r.count-first_new;
 }
 uint64_t occupation(const uint64_t* key,size_t k)const{return (key[word[k]]>>shift[k])&r.mask;}
 double root(uint64_t n)const{return n<roots.size()?roots[n]:std::sqrt(double(n));}
 template<bool LevelMode,unsigned Bits> int64_t ensure(const uint64_t* key) {
  uint64_t hash=key_hash(key,r.words);auto existing=lookup<Bits>(key,hash);if(existing.first>=0)return existing.first;
  int64_t depth=0;std::copy(key,key+r.words,stack.begin());hashes[0]=hash;cursor[0]=-2;
  while(depth>=0) {
   uint64_t* current=stack.data()+depth*r.words;
   if(cursor[depth]==-2) {
    // A dependency lowers at most two words. Its unchanged prefix has the
    // same hash as this frame; recompute only the suffix, using the unchanged
    // checkpoint hash function rather than a second key representation.
    uint64_t* prefix=prefixes.data()+depth*(r.words+1);prefix[0]=1469598103934665603ull;
    for(size_t w=0;w<r.words;w++)prefix[w+1]=append_hash(prefix[w],current[w]);
    uint64_t minimum=r.mask+1;size_t selected=0,count=0,total_degree=0;
    if constexpr(LevelMode)frame_edges[depth].clear();
    size_t* occupied=active.data()+depth*r.dimension;
    for(size_t k=0;k<r.dimension;k++) {
     uint64_t value=occupation(current,k);
     if constexpr(LevelMode)total_degree+=value;
     if(value){occupied[count++]=k;if(value<=minimum){minimum=value;selected=k;}}
    }
    if constexpr(LevelMode)degree[depth]=total_degree;
    require(minimum<=r.mask,"Invalid packed recurrence vacuum");axis[depth]=selected;
    denominator[depth]=root(minimum);sums[depth]=0.;cursor[depth]=-1;
    // Compact the occupied dependencies, skipping exact zeros with ascending
    // summation semantics; no unoccupied coordinate needs a dependency visit.
    size_t kept=0;
    for(size_t index=0;index<count;index++) {
     size_t k=occupied[index];
     if((k!=selected||minimum>1)&&r.matrix[selected*r.dimension+k]!=0.)occupied[kept++]=k;
    }active_count[depth]=kept;
   }
   size_t j=axis[depth];
   if(cursor[depth]>=int64_t(active_count[depth])) {
    if(r.count>=r.maximum_integrals)return -1;
    size_t node=r.count++;std::copy(current,current+r.words,r.keys+node*r.words);
    r.values[node]=sums[depth];auto slot=lookup<Bits>(current,hashes[depth]).second;r.slots[slot]=node;put_index<Bits>(slot,node,hashes[depth]);
    if constexpr(LevelMode) {
     edges.insert(edges.end(),frame_edges[depth].begin(),frame_edges[depth].end());offsets.push_back(edges.size());
     if(buckets.size()<=degree[depth])buckets.resize(degree[depth]+1);
     buckets[degree[depth]].push_back(node-first_new);
    }
    depth--;if(depth<0)return node;edge<LevelMode>(depth,pending[depth],node);continue;
   }
   int64_t index=cursor[depth]++;double factor;size_t k=0;
   if(index==-1){factor=r.linear[j]/denominator[depth];if(factor==0)continue;}
   else{k=active[depth*r.dimension+size_t(index)];uint64_t count=occupation(current,k)-(k==j?1:0);if(!count)continue;factor=r.matrix[j*r.dimension+k]*root(count)/denominator[depth];}
   // Construct a dependency key only after establishing a nonzero edge.
   std::copy(current,current+r.words,parent.begin());parent[word[j]]-=lowering[j];
   if(index>=0)parent[word[k]]-=lowering[k];
   r.edge_count++;size_t first=index<0?word[j]:std::min(word[j],word[k]);
   uint64_t parent_hash=prefixes[depth*(r.words+1)+first];
   for(size_t w=first;w<r.words;w++)parent_hash=append_hash(parent_hash,parent[w]);
   int64_t known=lookup<Bits>(parent.data(),parent_hash).first;
   if(known>=0)edge<LevelMode>(depth,factor,known);
   else{require(size_t(depth+1)<r.maximum_depth,"Packed recurrence depth exceeds occupation bounds");pending[depth]=factor;depth++;std::copy(parent.begin(),parent.end(),stack.begin()+depth*r.words);hashes[depth]=parent_hash;cursor[depth]=-2;}
  }return -1;
 }
 template<bool LevelMode,unsigned Bits> void evaluate_batch() {
  for(size_t row=0;row<r.rows&&!r.exhausted;row++) {
   std::fill(key.begin(),key.end(),0);
   for(size_t k=0;k<r.dimension;k++) {
    uint64_t value=r.states[row*r.dimension+k];require(value<=r.mask,"Packed occupation exceeds exact key width");key[word[k]]|=value<<shift[k];
   }
   int64_t node=ensure<LevelMode,Bits>(key.data());if(node<0){r.exhausted=1;break;}r.ids[row*(r.dimension+1)]=node;
   for(size_t k=0;k<r.dimension;k++)if(r.states[row*r.dimension+k]&&r.ht[k]) {
    key[word[k]]-=lowering[k];node=ensure<LevelMode,Bits>(key.data());key[word[k]]+=lowering[k];
    if(node<0){r.exhausted=1;break;}r.ids[row*(r.dimension+1)+k+1]=node;
   }
   if(!r.exhausted)r.completed++;
  }
  if constexpr(LevelMode)evaluate_levels();
 }
 void evaluate(PackedRequest& request) {
  bind(request);std::fill(r.ids,r.ids+r.rows*(r.dimension+1),-1);r.completed=0;r.exhausted=0;
  // Select a fully general kernel once per batch. The direct inner loop has
  // no level-scheduling branches, and index shifts are compile-time constants.
  if(levels) {
   if(index_bits==24)evaluate_batch<true,24>();else evaluate_batch<true,56>();
  }else {
   if(index_bits==24)evaluate_batch<false,24>();else evaluate_batch<false,56>();
  }
  request=r;
 }

};
struct OperatorRequest {
 size_t rows,dimension,channels,forms,count;const int64_t *states,*ids;const double* values;
 const std::complex<double> *constant,*linear,*weights;const double* q;
 double *strengths,*norms;
};
std::array<double,4> inverse2(const double* matrix,double& determinant) {
 determinant=matrix[0]*matrix[3]-matrix[1]*matrix[2];require(determinant>0,"Nonpositive proposal covariance");
 return {matrix[3]/determinant,-matrix[1]/determinant,-matrix[2]/determinant,matrix[0]/determinant};
}
void marginal1(double vx,double vp,double displacement,size_t maximum,double* output) {
 double determinant=(vx+.5)*(vp+.5),squeeze=(vx-vp)/(2*determinant),thermal=std::max(0.,(vx*vp-.25)/determinant),source=displacement/(std::sqrt(2.)*(vx+.5));size_t length=maximum+1;
 std::vector<double> table(length*length,0.);table[0]=std::exp(-displacement*displacement/(2*(vx+.5)))/std::sqrt(determinant);
 for(size_t j=1;j<length;j++)table[j]=(source*table[j-1]+(j>1?squeeze*std::sqrt(double(j-1))*table[j-2]:0.))/std::sqrt(double(j));
 for(size_t i=1;i<length;i++)for(size_t j=0;j<length;j++){double value=source*table[(i-1)*length+j];if(i>1)value+=squeeze*std::sqrt(double(i-1))*table[(i-2)*length+j];if(j)value+=thermal*std::sqrt(double(j))*table[(i-1)*length+j-1];table[i*length+j]=value/std::sqrt(double(i));}
 for(size_t i=0;i<length;i++)output[i]=std::max(0.,table[i*length+i]);
}
void marginal2(const double* vx,const double* vp,const double* mean,size_t c0,size_t c1,size_t n0,size_t n1,double* output) {
 double dx[4]{vx[0]+.5,vx[1],vx[2],vx[3]+.5},dp[4]{vp[0]+.5,vp[1],vp[2],vp[3]+.5},detx,detp;
 auto ix=inverse2(dx,detx),ip=inverse2(dp,detp);double projected[2]{ix[0]*mean[0]+ix[1]*mean[1],ix[2]*mean[0]+ix[3]*mean[1]};
 double vacuum=std::exp(-.5*(mean[0]*projected[0]+mean[1]*projected[1]))/std::sqrt(detx*detp),purity=0.;
 for(int i=0;i<2;i++)for(int j=0;j<2;j++){double value=0.;for(int k=0;k<2;k++)value+=vx[i*2+k]*vp[k*2+j];purity=std::max(purity,std::abs(value-(i==j?.25:0.)));}
 size_t a=c0+1,b=c1+1;
 if(purity<1e-12) {
  std::vector<double> table(a*b,0.);table[0]=std::sqrt(vacuum);double matrix[4]{1-ix[0],-ix[1],-ix[2],1-ix[3]},source[2]{projected[0]/std::sqrt(2.),projected[1]/std::sqrt(2.)};
  for(size_t i=0;i<a;i++)for(size_t j=0;j<b;j++)if(i){double value=source[0]*table[(i-1)*b+j];if(i>1)value+=matrix[0]*std::sqrt(double(i-1))*table[(i-2)*b+j];if(j)value+=matrix[1]*std::sqrt(double(j))*table[(i-1)*b+j-1];table[i*b+j]=value/std::sqrt(double(i));}else if(j){double value=source[1]*table[j-1];if(j>1)value+=matrix[3]*std::sqrt(double(j-1))*table[j-2];table[j]=value/std::sqrt(double(j));}
  for(size_t i=0;i<a*b;i++)output[i]=table[i]*table[i];return;
 }
 if(a*a*b*b>100000) {
  using Complex=std::complex<double>;require(n0>=a&&n1>=b&&n0<=16777216/n1,"Proposal FFT workspace budget exceeded");size_t length=n0*n1;
  double logs[2]{c0?std::log(1e-12)/n0:0.,c1?std::log(1e-12)/n1:0.},pi=std::acos(-1.);
  std::vector<Complex> signal(length),coefficients(length);double px[4]{1-ix[0],-ix[1],-ix[2],1-ix[3]},pp[4]{1-ip[0],-ip[1],-ip[2],1-ip[3]};
  auto logdet=[](Complex z0,Complex z1,const double* matrix){Complex trace=z0*matrix[0]+z1*matrix[3],det=z0*z1*(matrix[0]*matrix[3]-matrix[1]*matrix[2]),disc=std::sqrt(trace*trace-4.*det);return std::log(1.-.5*(trace+disc))+std::log(1.-.5*(trace-disc));};
  for(size_t i=0;i<n0;i++)for(size_t j=0;j<n1;j++) {
   Complex z0=c0?std::exp(Complex(logs[0],-2*pi*i/n0)):Complex{},z1=c1?std::exp(Complex(logs[1],-2*pi*j/n1)):Complex{};
   Complex s00=1.+z0+2.*(1.-z0)*vx[0],s01=2.*(1.-z0)*vx[1],s10=2.*(1.-z1)*vx[2],s11=1.+z1+2.*(1.-z1)*vx[3];
   Complex r0=(1.-z0)*mean[0],r1=(1.-z1)*mean[1],det=s00*s11-s01*s10,sol0=(s11*r0-s01*r1)/det,sol1=(-s10*r0+s00*r1)/det;
   signal[i*n1+j]=std::exp(-.5*(std::log(detx*detp)+logdet(z0,z1,px)+logdet(z0,z1,pp))-sol0*mean[0]-sol1*mean[1]);
  }
  pocketfft::c2c<double>({n0,n1},{std::ptrdiff_t(n1*sizeof(Complex)),std::ptrdiff_t(sizeof(Complex))},{std::ptrdiff_t(n1*sizeof(Complex)),std::ptrdiff_t(sizeof(Complex))},{0,1},false,signal.data(),coefficients.data(),1./length,1);
  for(size_t i=0;i<a;i++)for(size_t j=0;j<b;j++)output[i*b+j]=std::max(0.,coefficients[i*n1+j].real()*std::exp(-double(i)*logs[0]-double(j)*logs[1]));return;
 }
 double matrix[16],source[4]{projected[0]/std::sqrt(2.),projected[1]/std::sqrt(2.),projected[0]/std::sqrt(2.),projected[1]/std::sqrt(2.)};
 for(int i=0;i<4;i++)for(int j=0;j<4;j++)matrix[i*4+j]=(i/2==j/2)?.5*(ip[(i%2)*2+j%2]-ix[(i%2)*2+j%2]):(i%2==j%2?1.:0.)-.5*(ip[(i%2)*2+j%2]+ix[(i%2)*2+j%2]);
 size_t strides[4]{b*a*b,a*b,b,1},dims[4]{a,b,a,b},length=a*a*b*b;std::vector<double> table(length,0.);table[0]=vacuum;
 for(size_t index=1;index<length;index++){size_t counts[4],remaining=index;for(int k=0;k<4;k++){counts[k]=remaining/strides[k];remaining%=strides[k];}int axis=0;while(!counts[axis])axis++;size_t previous=index-strides[axis];counts[axis]--;double value=source[axis]*table[previous];for(int k=0;k<4;k++)if(counts[k])value+=matrix[axis*4+k]*std::sqrt(double(counts[k]))*table[previous-strides[k]];table[index]=value/std::sqrt(double(counts[axis]+1));}
 for(size_t i=0;i<a;i++)for(size_t j=0;j<b;j++)output[i*b+j]=std::max(0.,table[i*(strides[0]+strides[2])+j*(strides[1]+strides[3])]);
}
}

extern "C" {
void matrix_selection_abi_sizes(size_t* sizes){sizes[0]=sizeof(matrix_native_selection::TreeRequest);sizes[1]=sizeof(matrix_native_selection::PackedRequest);sizes[2]=sizeof(matrix_native_selection::OperatorRequest);sizes[3]=sizeof(std::complex<double>);}
const char* matrix_selection_error(){return matrix_native_selection::error.c_str();}
int matrix_gaussian_marginal_1(double vx,double vp,double displacement,size_t maximum,double* output){try{matrix_native_selection::marginal1(vx,vp,displacement,maximum,output);return 0;}catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}}
int matrix_gaussian_marginal_2(const double* vx,const double* vp,const double* mean,size_t first,size_t second,size_t fft_first,size_t fft_second,double* output){try{matrix_native_selection::marginal2(vx,vp,mean,first,second,fft_first,fft_second,output);return 0;}catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}}
void* matrix_tree_create(const matrix_native_selection::TreeRequest* request){try{return new matrix_native_selection::Tree(*request);}catch(const std::exception& e){matrix_native_selection::error=e.what();return nullptr;}}
void matrix_tree_destroy(void* tree){delete static_cast<matrix_native_selection::Tree*>(tree);}
size_t matrix_tree_heap_size(void* tree){return static_cast<matrix_native_selection::Tree*>(tree)->heap.size();}
int64_t matrix_tree_batch(void* handle,size_t count,int64_t* output,size_t* allocated,size_t* visited){try{auto& tree=*static_cast<matrix_native_selection::Tree*>(handle);size_t n=tree.batch(count,output);*allocated=tree.r.allocated;*visited=tree.r.visited;return n;}catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}}
int matrix_tree_export_heap(void* handle,double* costs,int64_t* nodes,int64_t* axes){try{const auto& heap=static_cast<matrix_native_selection::Tree*>(handle)->heap;for(size_t k=0;k<heap.size();k++){costs[k]=std::get<0>(heap[k]);nodes[k]=std::get<1>(heap[k]);axes[k]=std::get<2>(heap[k]);}return 0;}catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}}
void* matrix_packed_context_create(int levels,size_t workers) {
 try{return new matrix_native_selection::Workspace(levels!=0,workers);}
 catch(const std::exception& e){matrix_native_selection::error=e.what();return nullptr;}
}
void matrix_packed_context_destroy(void* context){delete static_cast<matrix_native_selection::Workspace*>(context);}
int matrix_packed_context_evaluate(void* context,matrix_native_selection::PackedRequest* request) {
 try {
  auto& r=*request;using namespace matrix_native_selection;
  require(r.dimension&&r.words&&r.slot_count&&(r.slot_count&(r.slot_count-1))==0&&r.count>0&&r.count<=r.maximum_integrals,"Invalid packed cache dimensions");
  require(r.bits==1||r.bits==2||r.bits==4||r.bits==8||r.bits==16||r.bits==32,"Invalid packed occupation width");
  static_cast<Workspace*>(context)->evaluate(r);return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
int matrix_packed_context_stats(void* context,size_t* output) {
 auto& w=*static_cast<matrix_native_selection::Workspace*>(context);
 size_t values[]{w.batches,w.lookups,w.rejections,w.comparisons,w.growths,w.index_bytes(),w.level_nodes,w.parallel_levels,w.executor->size()};
 std::copy(values,values+9,output);return 0;
}
int matrix_packed_evaluate(matrix_native_selection::PackedRequest* request) {
 matrix_native_selection::Workspace workspace;return matrix_packed_context_evaluate(&workspace,request);
}
int matrix_packed_rehash(const uint64_t* keys,size_t words,size_t count,int64_t* slots,size_t capacity) {
 try{std::fill(slots,slots+capacity,-1);for(size_t node=0;node<count;node++){auto found=matrix_native_selection::find(keys+node*words,keys,words,slots,capacity);matrix_native_selection::require(found.first<0,"Duplicate overlap keys in checkpoint");slots[found.second]=node;}return 0;}catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
int matrix_operator_strengths(matrix_native_selection::OperatorRequest* request) {
 try {
  auto& r=*request;using namespace matrix_native_selection;using Complex=std::complex<double>;std::vector<Complex> amplitude(r.channels);double vacuum=1.;for(size_t k=0;k<r.dimension/2;k++)vacuum*=1-r.q[k];
  for(size_t row=0;row<r.rows;row++) {
   int64_t overlap=r.ids[row*(r.dimension+1)];require(overlap>=0&&size_t(overlap)<r.count,"Invalid operator overlap index");
   for(size_t a=0;a<r.channels;a++)amplitude[a]=r.constant[a]*r.values[overlap];
   for(size_t k=0;k<r.dimension;k++){int64_t lower=r.ids[row*(r.dimension+1)+k+1];if(lower>=0){require(size_t(lower)<r.count,"Invalid operator recurrence index");double factor=std::sqrt(double(r.states[row*r.dimension+k]))*r.values[lower];for(size_t a=0;a<r.channels;a++)amplitude[a]+=r.linear[a*r.dimension+k]*factor;}}
   double pop=vacuum,norm=0.;for(size_t k=0;k<r.dimension/2;k++)pop*=std::pow(r.q[k],r.states[row*r.dimension+k]);for(Complex value:amplitude)norm+=std::norm(value);r.norms[row]=pop*norm;
   for(size_t f=0;f<r.forms;f++){Complex value=0.;for(size_t a=0;a<r.channels;a++)for(size_t b=0;b<r.channels;b++)value+=amplitude[a]*r.weights[(f*r.channels+a)*r.channels+b]*std::conj(amplitude[b]);r.strengths[f*r.rows+row]=pop*value.real();}
  }return 0;
 }catch(const std::exception& e){matrix_native_selection::error=e.what();return -1;}
}
}
