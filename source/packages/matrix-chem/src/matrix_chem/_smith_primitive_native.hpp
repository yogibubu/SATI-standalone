#pragma once

// Compiled sparse evaluator for the complete frozen SMITH primitive atlas.
// Derivatives live only on the atoms touched by one primitive: fragment
// coordinates therefore scale with their physical support, while local and
// ring coordinates retain bounded support.

struct SmithDual {
    double v = 0.0;
    std::vector<double> d;
    SmithDual() = default;
    SmithDual(double value, std::size_t n) : v(value), d(n, 0.0) {}
};
using SmithVec = std::array<SmithDual, 3>;
using PlainVec = std::array<double, 3>;

SmithDual sd_add(const SmithDual& a, const SmithDual& b) {
    SmithDual r(a.v + b.v, a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=a.d[i]+b.d[i];
    return r;
}
SmithDual sd_sub(const SmithDual& a, const SmithDual& b) {
    SmithDual r(a.v - b.v, a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=a.d[i]-b.d[i];
    return r;
}
SmithDual sd_mul(const SmithDual& a, const SmithDual& b) {
    SmithDual r(a.v*b.v, a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=a.d[i]*b.v+a.v*b.d[i];
    return r;
}
SmithDual sd_scale(const SmithDual& a, double b) {
    SmithDual r(a.v*b, a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=a.d[i]*b;
    return r;
}
SmithDual sd_div(const SmithDual& a, const SmithDual& b) {
    if (std::abs(b.v) <= 1.e-15) throw std::runtime_error("singular SMITH division");
    SmithDual r(a.v/b.v, a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=(a.d[i]-r.v*b.d[i])/b.v;
    return r;
}
SmithDual sd_sqrt(const SmithDual& a) {
    if (a.v <= 1.e-24) throw std::runtime_error("singular SMITH square root");
    SmithDual r(std::sqrt(a.v), a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=a.d[i]/(2.*r.v);
    return r;
}
SmithDual sd_exp(const SmithDual& a) {
    SmithDual r(std::exp(a.v), a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=r.v*a.d[i];
    return r;
}
SmithDual sd_acos(const SmithDual& a) {
    const double x=std::max(-1.,std::min(1.,a.v));
    const double den=std::sqrt(std::max(0.,1.-x*x));
    if (den <= 1.e-12) throw std::runtime_error("singular SMITH acos");
    SmithDual r(std::acos(x), a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=-a.d[i]/den;
    return r;
}
SmithDual sd_asin(const SmithDual& a) {
    const double x=std::max(-1.,std::min(1.,a.v));
    const double den=std::sqrt(std::max(0.,1.-x*x));
    if (den <= 1.e-12) throw std::runtime_error("singular SMITH asin");
    SmithDual r(std::asin(x), a.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=a.d[i]/den;
    return r;
}
SmithDual sd_atan2(const SmithDual& y,const SmithDual& x) {
    const double den=x.v*x.v+y.v*y.v;
    if (den <= 1.e-12) throw std::runtime_error("singular SMITH atan2");
    SmithDual r(std::atan2(y.v,x.v), y.d.size());
    for (std::size_t i=0;i<r.d.size();++i) r.d[i]=(x.v*y.d[i]-y.v*x.d[i])/den;
    return r;
}
SmithVec sv_add(const SmithVec&a,const SmithVec&b){return{sd_add(a[0],b[0]),sd_add(a[1],b[1]),sd_add(a[2],b[2])};}
SmithVec sv_sub(const SmithVec&a,const SmithVec&b){return{sd_sub(a[0],b[0]),sd_sub(a[1],b[1]),sd_sub(a[2],b[2])};}
SmithVec sv_scale(const SmithVec&a,double b){return{sd_scale(a[0],b),sd_scale(a[1],b),sd_scale(a[2],b)};}
SmithVec sv_scale_dual(const SmithVec&a,const SmithDual&b){return{sd_mul(a[0],b),sd_mul(a[1],b),sd_mul(a[2],b)};}
SmithDual sv_dot(const SmithVec&a,const SmithVec&b){return sd_add(sd_add(sd_mul(a[0],b[0]),sd_mul(a[1],b[1])),sd_mul(a[2],b[2]));}
SmithVec sv_cross(const SmithVec&a,const SmithVec&b){return{sd_sub(sd_mul(a[1],b[2]),sd_mul(a[2],b[1])),sd_sub(sd_mul(a[2],b[0]),sd_mul(a[0],b[2])),sd_sub(sd_mul(a[0],b[1]),sd_mul(a[1],b[0]))};}
SmithVec sv_unit(const SmithVec&a){auto n=sd_sqrt(sv_dot(a,a));return{sd_div(a[0],n),sd_div(a[1],n),sd_div(a[2],n)};}

PlainVec pv_sub(const PlainVec&a,const PlainVec&b){return{a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
PlainVec pv_cross(const PlainVec&a,const PlainVec&b){return{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double pv_dot(const PlainVec&a,const PlainVec&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
PlainVec pv_unit(const PlainVec&a){double n=std::sqrt(pv_dot(a,a));if(n<=1.e-12)throw std::runtime_error("singular SMITH vector");return{a[0]/n,a[1]/n,a[2]/n};}

struct SmithTape {
    const double* xyz;
    const double* refxyz;
    npy_intp natoms;
    std::vector<npy_intp> support;
    std::map<npy_intp,std::size_t> local;
    std::size_t dim() const { return 3*support.size(); }
    SmithDual c(double v) const { return SmithDual(v,dim()); }
    SmithVec point(npy_intp atom) const {
        auto it=local.find(atom); if(it==local.end())throw std::runtime_error("SMITH support mismatch");
        SmithVec r{c(xyz[3*atom]),c(xyz[3*atom+1]),c(xyz[3*atom+2])};
        for(int k=0;k<3;++k) {
            r[k].d[3*it->second+k]=1.;
        }
        return r;
    }
    SmithVec center(const npy_intp* atoms,npy_intp n) const {
        if(n<=0)throw std::runtime_error("empty SMITH fragment");
        SmithVec r{c(0),c(0),c(0)};
        for(npy_intp q=0;q<n;++q){auto it=local.find(atoms[q]);if(it==local.end())throw std::runtime_error("SMITH support mismatch");for(int k=0;k<3;++k){r[k].v+=xyz[3*atoms[q]+k]/n;r[k].d[3*it->second+k]+=1./n;}}
        return r;
    }
    PlainVec plain_point(npy_intp atom,bool reference) const {const double*p=reference?refxyz:xyz;return{p[3*atom],p[3*atom+1],p[3*atom+2]};}
    PlainVec plain_center(const npy_intp* atoms,npy_intp n,bool reference) const {if(n<=0)throw std::runtime_error("empty SMITH fragment");PlainVec r{0,0,0};const double*p=reference?refxyz:xyz;for(npy_intp q=0;q<n;++q)for(int k=0;k<3;++k)r[k]+=p[3*atoms[q]+k]/n;return r;}
};

SmithDual smith_angle(const SmithTape&t,npy_intp a,npy_intp b,npy_intp c){return sd_acos(sv_dot(sv_unit(sv_sub(t.point(a),t.point(b))),sv_unit(sv_sub(t.point(c),t.point(b)))));}
SmithDual smith_dihedral(const SmithTape&t,npy_intp a,npy_intp b,npy_intp c,npy_intp d){auto p0=t.point(a),p1=t.point(b),p2=t.point(c),p3=t.point(d);auto b0=sv_scale(sv_sub(p1,p0),-1.);auto b1=sv_unit(sv_sub(p2,p1));auto b2=sv_sub(p3,p2);auto v=sv_sub(b0,sv_scale_dual(b1,sv_dot(b0,b1)));auto w=sv_sub(b2,sv_scale_dual(b1,sv_dot(b2,b1)));/* The projected-vector formula has both atan2 arguments opposite to
the canonical ORACLE convention based on the two plane normals.  Negate
both arguments so values and derivatives share exactly the same chart. */auto y=sv_dot(sv_cross(b1,v),w);auto x=sv_dot(v,w);return sd_atan2(sd_scale(y,-1.),sd_scale(x,-1.));}
SmithDual smith_u(const SmithTape&t,npy_intp c,npy_intp a,npy_intp b,npy_intp o){auto r1=sv_sub(t.point(a),t.point(c));auto r2=sv_sub(t.point(b),t.point(c));auto ro=sv_sub(t.point(o),t.point(c));return sd_scale(sd_asin(sv_dot(sv_unit(ro),sv_unit(sv_cross(r1,r2)))),-1.);}
SmithDual smith_h(const SmithTape&t,npy_intp c,npy_intp a,npy_intp b,npy_intp o){return sv_dot(sv_sub(t.point(o),t.point(c)),sv_unit(sv_cross(sv_sub(t.point(a),t.point(c)),sv_sub(t.point(b),t.point(c)))));}

std::array<SmithVec,3> smith_frame3(const SmithTape&t,const npy_intp* atoms,npy_intp n,const npy_intp* anchors){auto center=t.center(atoms,n);auto p=sv_unit(sv_sub(t.point(anchors[0]),center));auto q=sv_unit(sv_cross(p,sv_sub(t.point(anchors[1]),center)));auto s=sv_unit(sv_cross(p,q));return{p,q,s};}
std::array<PlainVec,3> plain_frame3(const SmithTape&t,const npy_intp* atoms,npy_intp n,const npy_intp* anchors,bool reference){auto center=t.plain_center(atoms,n,reference);auto p=pv_unit(pv_sub(t.plain_point(anchors[0],reference),center));auto q=pv_unit(pv_cross(p,pv_sub(t.plain_point(anchors[1],reference),center)));auto s=pv_unit(pv_cross(p,q));return{p,q,s};}

std::array<SmithDual,3> smith_rotation_vector(const std::array<std::array<SmithDual,3>,3>&r){auto trace=sd_add(sd_add(r[0][0],r[1][1]),r[2][2]);if(trace.v<=-1.+1.e-12)throw std::runtime_error("singular FROT chart");auto kw=sd_scale(sd_sqrt(sd_add(trace,SmithDual(1.,trace.d.size()))),.5);auto den=sd_scale(kw,4.);auto kx=sd_div(sd_sub(r[1][2],r[2][1]),den);auto ky=sd_div(sd_sub(r[2][0],r[0][2]),den);auto kz=sd_div(sd_sub(r[0][1],r[1][0]),den);auto kn2=sd_add(sd_add(sd_mul(kx,kx),sd_mul(ky,ky)),sd_mul(kz,kz));if(kn2.v<=1.e-24)return{sd_scale(kx,2),sd_scale(ky,2),sd_scale(kz,2)};auto kn=sd_sqrt(kn2);auto factor=sd_div(sd_scale(sd_atan2(kn,kw),2.),kn);return{sd_mul(factor,kx),sd_mul(factor,ky),sd_mul(factor,kz)};}

// Input arrays are a compact, immutable plan produced by matrix-smith.
PyObject* smith_primitive_values_sparse_gradients(PyObject*, PyObject* args) {
    PyObject *co,*ro,*ko,*mo,*ad_o,*ap_o,*rd_o,*rp_o,*fd_o,*fp_o,*gd_o,*gp_o,*tk_o,*ta_o,*tc_o,*tp_o;
    if(!PyArg_ParseTuple(args,"OOOOOOOOOOOOOOOO",&co,&ro,&ko,&mo,&ad_o,&ap_o,&rd_o,&rp_o,&fd_o,&fp_o,&gd_o,&gp_o,&tk_o,&ta_o,&tc_o,&tp_o))return nullptr;
    std::vector<PyArrayObject*> owned;
    auto arr=[&](PyObject*o,int typ){auto*a=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(o,typ,NPY_ARRAY_IN_ARRAY));if(a)owned.push_back(a);return a;};
    auto*c=arr(co,NPY_DOUBLE),*r=arr(ro,NPY_DOUBLE),*k=arr(ko,NPY_INTP),*m=arr(mo,NPY_INTP),*ad=arr(ad_o,NPY_INTP),*ap=arr(ap_o,NPY_INTP),*rd=arr(rd_o,NPY_INTP),*rp=arr(rp_o,NPY_INTP),*fd=arr(fd_o,NPY_INTP),*fp=arr(fp_o,NPY_INTP),*gd=arr(gd_o,NPY_INTP),*gp=arr(gp_o,NPY_INTP),*tk=arr(tk_o,NPY_INTP),*ta=arr(ta_o,NPY_INTP),*tc=arr(tc_o,NPY_DOUBLE),*tp=arr(tp_o,NPY_INTP);
    if(owned.size()!=16){for(auto*x:owned)Py_DECREF(x);return nullptr;}
    const npy_intp n=PyArray_DIM(c,0), nr=PyArray_DIM(k,0); if(PyArray_NDIM(c)!=2||PyArray_DIM(c,1)!=3||PyArray_NDIM(r)!=2||PyArray_DIM(r,0)!=n||PyArray_DIM(r,1)!=3){for(auto*x:owned)Py_DECREF(x);PyErr_SetString(PyExc_ValueError,"invalid SMITH coordinate arrays");return nullptr;}
    const double*xyz=(double*)PyArray_DATA(c),*refxyz=(double*)PyArray_DATA(r),*tcoef=(double*)PyArray_DATA(tc);const npy_intp*kinds=(npy_intp*)PyArray_DATA(k),*modes=(npy_intp*)PyArray_DATA(m),*atoms=(npy_intp*)PyArray_DATA(ad),*atp=(npy_intp*)PyArray_DATA(ap),*refs=(npy_intp*)PyArray_DATA(rd),*refp=(npy_intp*)PyArray_DATA(rp),*frames=(npy_intp*)PyArray_DATA(fd),*framep=(npy_intp*)PyArray_DATA(fp),*rframes=(npy_intp*)PyArray_DATA(gd),*rframep=(npy_intp*)PyArray_DATA(gp),*termk=(npy_intp*)PyArray_DATA(tk),*terma=(npy_intp*)PyArray_DATA(ta),*termp=(npy_intp*)PyArray_DATA(tp);
    npy_intp vd=nr;auto*vals=(PyArrayObject*)PyArray_SimpleNew(1,&vd,NPY_DOUBLE);if(!vals){for(auto*x:owned)Py_DECREF(x);return nullptr;}auto*vout=(double*)PyArray_DATA(vals);std::vector<double> data;std::vector<npy_intp> indices,indptr{0};
    struct SmithPrimitiveRowResult {double value=0.;std::vector<double> data;std::vector<npy_intp> indices;};
    std::vector<SmithPrimitiveRowResult> row_results(static_cast<std::size_t>(nr));
    auto evaluate_row=[&](npy_intp row){std::set<npy_intp> ss;auto add=[&](const npy_intp*x,npy_intp p,npy_intp q){for(npy_intp z=p;z<q;++z){if(x[z]<0||x[z]>=n)throw std::runtime_error("SMITH atom out of range");ss.insert(x[z]);}};add(atoms,atp[row],atp[row+1]);add(refs,refp[row],refp[row+1]);add(frames,framep[row],framep[row+1]);add(rframes,rframep[row],rframep[row+1]);for(npy_intp z=termp[row];z<termp[row+1];++z)add(terma,4*z,4*z+4);SmithTape t{xyz,refxyz,n,{ss.begin(),ss.end()},{}};for(std::size_t z=0;z<t.support.size();++z)t.local[t.support[z]]=z;const npy_intp*A=atoms+atp[row],na=atp[row+1]-atp[row],*R=refs+refp[row],nb=refp[row+1]-refp[row],*F=frames+framep[row],nf=framep[row+1]-framep[row],*G=rframes+rframep[row],ng=rframep[row+1]-rframep[row];SmithDual out(0.,t.dim());int kind=(int)kinds[row];
        if(kind==1||kind==23||kind==24||kind==25){
          auto d=sv_sub(t.point(A[0]),t.point(A[1]));
          auto radius=sd_sqrt(sv_dot(d,d));
          if(kind==1)out=radius;
          else if(kind==23)out=sd_div(t.c(1.),radius);
          else if(kind==24){auto r2=sd_mul(radius,radius);out=sd_div(t.c(1.),sd_mul(r2,radius));}
          else out=sd_exp(sd_scale(radius,-1.));
        }
        else if(kind==2)out=smith_angle(t,A[0],A[1],A[2]);
        else if(kind==3||kind==7)out=smith_dihedral(t,A[0],A[1],A[2],A[3]);
        else if(kind==4)out=smith_u(t,A[0],A[1],A[2],A[3]); else if(kind==5)out=smith_h(t,A[0],A[1],A[2],A[3]);
        else if(kind==6){if(nb==1&&modes[row]==-1)out=sd_add(smith_angle(t,A[0],A[1],R[0]),smith_angle(t,R[0],A[1],A[2]));else if(nb==1){auto x=smith_dihedral(t,A[0],A[1],R[0],A[2]),y=smith_dihedral(t,A[0],R[0],A[1],A[2]);out=sd_scale(sd_sub(x,y),.5);double w=std::fmod(x.v-y.v+MATRIX_PI,2*MATRIX_PI);if(w<0)w+=2*MATRIX_PI;out.v=MATRIX_PI+.5*(w-MATRIX_PI);}else{auto u=sv_unit(sv_sub(t.point(A[0]),t.point(A[1]))),v=sv_unit(sv_sub(t.point(A[2]),t.point(A[1])));SmithVec axis{t.c(1),t.c(0),t.c(0)};if(std::abs(u[0].v)>.9)axis={t.c(0),t.c(1),t.c(0)};auto e1=sv_unit(sv_cross(u,axis)),e2=sv_unit(sv_cross(u,e1));out=sv_dot(sv_add(u,v),modes[row]==-1?e1:e2);}}
        else if(kind==8||kind==9){auto delta=sv_sub(t.center(A,na),kind==8?t.center(R,nb):t.point(R[0]));out=sd_sqrt(sv_dot(delta,delta));}
        else if(kind==10&&ng==0){auto delta=sv_sub(t.center(A,na),t.center(R,nb));out=delta[modes[row]];}
        else if(kind>=12&&kind<=14){for(npy_intp z=termp[row];z<termp[row+1];++z){const npy_intp*q=terma+4*z;SmithDual x=termk[z]==2?smith_angle(t,q[0],q[1],q[2]):termk[z]==3?smith_dihedral(t,q[0],q[1],q[2],q[3]):smith_u(t,q[0],q[1],q[2],q[3]);out=sd_add(out,sd_scale(x,tcoef[z]));}}
        else if(kind>=18 && kind<=22){
          auto normal=[&](const npy_intp* face,npy_intp count){
            if(count<3)throw std::runtime_error("face requires three ordered atoms");
            auto center=t.center(face,count); SmithVec area{t.c(0),t.c(0),t.c(0)};
            for(npy_intp j=0;j<count;++j){auto term=sv_cross(sv_sub(t.point(face[j]),center),sv_sub(t.point(face[(j+1)%count]),center));for(int z=0;z<3;++z)area[z]=sd_add(area[z],term[z]);}
            return sv_unit(area);
          };
          auto delta=sv_sub(t.center(A,na),t.center(R,nb));
          if(kind==18)out=sv_dot(delta,normal(R,nb));
          else if(kind==19)out=sd_sqrt(sv_dot(delta,delta));
          else if(kind==20)out=sd_acos(sv_dot(normal(R,nb),sv_unit(delta)));
          else if(kind==21){if(nf!=2)throw std::runtime_error("hinge needs two edge atoms");auto n1=normal(A,na),n2=normal(R,nb);out=sd_atan2(sv_dot(sv_unit(sv_sub(t.point(F[1]),t.point(F[0]))),sv_cross(n1,n2)),sv_dot(n1,n2));}
          else {if(nf<1)throw std::runtime_error("contact needs vertex center");auto origin=t.center(F,nf);out=sd_acos(sv_dot(sv_unit(sv_sub(t.center(A,na),origin)),sv_unit(sv_sub(t.center(R,nb),origin))));}
        }
        else if(kind==17) out=t.c(0.0);
        else {auto delta=sv_sub(t.center(A,na),t.center(R,nb));if((kind==10&&ng==2)||kind==16){if(ng!=2)throw std::runtime_error("fragment reference frame needs two anchors");auto rf=smith_frame3(t,R,nb,G);if(kind==10)out=sv_dot(delta,rf[modes[row]]);else if(nf==1){auto dir=sv_unit(sv_sub(t.point(F[0]),t.center(A,na)));std::array<double,3> comp{};auto prf=plain_frame3(t,R,nb,G,true);auto pc=pv_unit(pv_sub(t.plain_point(F[0],true),t.plain_center(A,na,true)));for(int z=0;z<3;++z)comp[z]=pv_dot(pc,prf[z]);int pole=0;for(int z=1;z<3;++z)if(std::abs(comp[z])>std::abs(comp[pole]))pole=z;double sign=comp[pole]>=0?1:-1;std::vector<int> tr;for(int z=0;z<3;++z)if(z!=pole)tr.push_back(z);auto den=sd_add(t.c(1),sd_scale(sv_dot(dir,rf[pole]),sign));out=sd_div(sv_dot(dir,rf[tr[modes[row]]]),den);auto pdir=pv_unit(pv_sub(t.plain_point(F[0],true),t.plain_center(A,na,true)));double rv=pv_dot(pdir,prf[tr[modes[row]]])/(1+sign*pv_dot(pdir,prf[pole]));out.v-=rv;}else{if(nf!=2)throw std::runtime_error("FROT needs one or two moving anchors");auto mf=smith_frame3(t,A,na,F);std::array<std::array<SmithDual,3>,3> rot;for(int i=0;i<3;++i)for(int j=0;j<3;++j)rot[i][j]=sv_dot(mf[i],rf[j]);auto pm=plain_frame3(t,A,na,F,true),pr=plain_frame3(t,R,nb,G,true);double rr[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)rr[i][j]=pv_dot(pm[i],pr[j]);std::array<std::array<SmithDual,3>,3> dr;for(int i=0;i<3;++i)for(int j=0;j<3;++j){dr[i][j]=t.c(0);for(int z=0;z<3;++z)dr[i][j]=sd_add(dr[i][j],sd_scale(rot[i][z],rr[j][z]));}out=smith_rotation_vector(dr)[modes[row]];}}
          else {if(kind==11||kind==15){if(ng!=1)throw std::runtime_error("axial primitive needs one reference anchor");auto primary=sv_unit(sv_sub(t.point(G[0]),t.center(R,nb)));auto para=sv_dot(delta,primary);auto secondary=sv_unit(sv_sub(delta,sv_scale_dual(primary,para)));auto normal=sv_unit(sv_cross(primary,secondary));if(kind==11)out=modes[row]==0?para:sv_dot(delta,secondary);else{if(nf!=1)throw std::runtime_error("FAXIS needs one moving anchor");auto dir=sv_unit(sv_sub(t.point(F[0]),t.center(A,na)));std::array<SmithVec,3> axes{primary,secondary,normal};auto pc=pv_unit(pv_sub(t.plain_point(F[0],true),t.plain_center(A,na,true)));auto pdelta=pv_sub(t.plain_center(A,na,true),t.plain_center(R,nb,true));auto pp=pv_unit(pv_sub(t.plain_point(G[0],true),t.plain_center(R,nb,true)));auto ps=pv_unit(pv_sub(pdelta,PlainVec{pv_dot(pdelta,pp)*pp[0],pv_dot(pdelta,pp)*pp[1],pv_dot(pdelta,pp)*pp[2]}));auto pn=pv_unit(pv_cross(pp,ps));std::array<PlainVec,3> pax{pp,ps,pn};double cc[3];for(int z=0;z<3;++z)cc[z]=pv_dot(pc,pax[z]);int pole=0;for(int z=1;z<3;++z)if(std::abs(cc[z])>std::abs(cc[pole]))pole=z;double sign=cc[pole]>=0?1:-1;std::vector<int>tr;for(int z=0;z<3;++z)if(z!=pole)tr.push_back(z);out=sd_div(sv_dot(dir,axes[tr[modes[row]]]),sd_add(t.c(1),sd_scale(sv_dot(dir,axes[pole]),sign)));}}else throw std::runtime_error("unsupported SMITH primitive code");}}
        auto&stored=row_results[static_cast<std::size_t>(row)];stored.value=out.v;stored.data.reserve(out.d.size());stored.indices.reserve(out.d.size());for(std::size_t z=0;z<t.support.size();++z)for(int axis=0;axis<3;++axis){double x=out.d[3*z+axis];if(std::abs(x)>1.e-15){stored.indices.push_back(3*t.support[z]+axis);stored.data.push_back(x);}}};
    try {
      int worker_count=1;const char*requested=std::getenv("MATRIX_COORDINATE_THREADS");if(requested==nullptr)requested=std::getenv("OMP_NUM_THREADS");if(requested!=nullptr&&nr>=512)worker_count=std::max(1,std::min(8,std::atoi(requested)));worker_count=std::min(worker_count,static_cast<int>(nr));
      std::atomic<bool>failed(false);std::exception_ptr failure;std::mutex failure_mutex;
      auto run_row=[&](npy_intp row){if(failed.load(std::memory_order_relaxed))return;try{evaluate_row(row);}catch(...){std::lock_guard<std::mutex>guard(failure_mutex);if(!failure)failure=std::current_exception();failed.store(true,std::memory_order_relaxed);}};
      Py_BEGIN_ALLOW_THREADS
      if(worker_count<=1){for(npy_intp row=0;row<nr;++row)run_row(row);}else{std::vector<std::thread>workers;workers.reserve(static_cast<std::size_t>(worker_count));for(int worker=0;worker<worker_count;++worker)workers.emplace_back([&,worker](){for(npy_intp row=worker;row<nr;row+=worker_count)run_row(row);});for(auto&worker:workers)worker.join();}
      Py_END_ALLOW_THREADS
      if(failure)std::rethrow_exception(failure);
      for(npy_intp row=0;row<nr;++row){const auto&stored=row_results[static_cast<std::size_t>(row)];vout[row]=stored.value;indices.insert(indices.end(),stored.indices.begin(),stored.indices.end());data.insert(data.end(),stored.data.begin(),stored.data.end());indptr.push_back(static_cast<npy_intp>(data.size()));}
    } catch(const std::exception&e){for(auto*x:owned)Py_DECREF(x);Py_DECREF(vals);PyErr_SetString(PyExc_FloatingPointError,e.what());return nullptr;}
    for(auto*x:owned)Py_DECREF(x);PyObject*do_=numpy_vector(data,NPY_DOUBLE),*io=numpy_vector(indices,NPY_INTP),*po=numpy_vector(indptr,NPY_INTP);if(!do_||!io||!po){Py_XDECREF(do_);Py_XDECREF(io);Py_XDECREF(po);Py_DECREF(vals);return nullptr;}return Py_BuildValue("NNNN",vals,do_,io,po);
}
