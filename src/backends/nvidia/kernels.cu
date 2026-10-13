#include "native.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace {
void check(at::Tensor x, at::ScalarType type) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==type,"Expected contiguous CUDA tensor");
}
template<int Group> auto sf_layout(int r,int c) {
  return cutlass::detail::Sm1xxBlockScaledConfig<Group>::tile_atom_to_shape_SFA(cute::make_shape(r,r,c,1));
}
__device__ uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
__device__ uint8_t fp4(float x,bool stochastic,uint32_t seed) {
  const float v[8]={0,.5f,1,1.5f,2,3,4,6};
  float a=fabsf(x); int hi=1;
  while(hi<7 && a>v[hi]) ++hi;
  int lo=hi-1, code;
  if(a>=6) code=7;
  else if(stochastic) code=((hash32(seed)>>8)*0x1p-24f<(a-v[lo])/(v[hi]-v[lo])) ? hi : lo;
  else { float mid=(v[lo]+v[hi])*.5f; code=a<mid ? lo : a>mid ? hi : (lo%2==0 ? lo : hi); }
  return code | (signbit(x) ? 8 : 0);
}
template<int P, class Layout> __global__ void pack_kernel(const float* x,uint8_t* q,uint8_t* s,
    const float* global,int r,int c,int rp,int cp,bool tile,bool stochastic,uint32_t seed,Layout layout) {
  constexpr int group=P==2 ? 32 : P==3 ? 16 : 1;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rp*(cp/group)) return;
  int row=i/(cp/group), start=(i%(cp/group))*group;
  if constexpr(P<2) {
    float value=(row<r && start<c) ? x[row*c+start] : 0;
    if constexpr(P==0) reinterpret_cast<float*>(q)[i]=value;
    else reinterpret_cast<cutlass::bfloat16_t*>(q)[i]=cutlass::bfloat16_t(value);
  } else {
    float maximum=0, g=*global;
    int begin=tile ? row/16*16 : row, end=tile ? begin+16 : row+1;
    for(int rr=begin;rr<end && rr<r;++rr)
      for(int kk=start;kk<start+group && kk<c;++kk) maximum=fmaxf(maximum,fabsf(x[rr*c+kk]));
    float scale=1; uint8_t scale_code;
    if constexpr(P==2) {
      int exponent=maximum>0 ? int(ceilf(log2f(maximum/448.f))) : 0;
      exponent=max(-127,min(127,exponent)); scale=ldexpf(1.f,exponent); scale_code=uint8_t(exponent+127);
    } else {
      float target=maximum>0 ? maximum/(6.f*g) : 1.f;
      cutlass::float_e4m3_t value(fminf(448.f,fmaxf(0x1p-9f,target)));
      scale=float(value); scale_code=value.raw();
    }
    s[layout(cute::make_coord(row,start,0))]=scale_code;
    for(int t=0;t<group;++t) {
      int col=start+t, idx=row*cp+col;
      float value=(row<r && col<c) ? x[row*c+col]/(scale*g) : 0;
      if constexpr(P==2) q[idx]=cutlass::float_e4m3_t(fminf(448.f,fmaxf(-448.f,value))).raw();
      else {
        uint8_t code=fp4(value,stochastic,seed^uint32_t(idx));
        if(t%2==0) q[idx/2]=code; else q[idx/2] |= code<<4;
      }
    }
  }
}
template<int P,class Layout> __device__ float value(const uint8_t* q,const uint8_t* s,const float* g,
    int row,int col,int cp,Layout layout) {
  int idx=row*cp+col;
  if constexpr(P==0) return reinterpret_cast<const float*>(q)[idx];
  else if constexpr(P==1) return float(reinterpret_cast<const cutlass::bfloat16_t*>(q)[idx]);
  else {
    uint8_t code=s[layout(cute::make_coord(row,col,0))]; float scale;
    if constexpr(P==2) {
      scale=ldexpf(1.f,int(code)-127); cutlass::float_e4m3_t v; v.raw()=q[idx]; return float(v)*scale*(*g);
    } else {
      cutlass::float_e4m3_t v; v.raw()=code; scale=float(v);
      cutlass::float_e2m1_t f; f.raw()=(q[idx/2]>>((idx%2)*4))&15; return float(f)*scale*(*g);
    }
  }
}
template<int P,class Layout> __global__ void decode_kernel(const uint8_t* q,const uint8_t* s,const float* g,
    float* out,int r,int c,int cp,Layout layout) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<r*c) out[idx]=value<P>(q,s,g,idx/c,idx%c,cp,layout);
}
// Four input reductions followed by checksum propagation. Never forms A@B.
template<int P,class Layout> __global__ void input_checks(const uint8_t* a,const uint8_t* sa,const float* ga,
    const uint8_t* b,const uint8_t* sb,const float* gb,float* tmp,int m,int n,int k,int kp,Layout la,Layout lb) {
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=k) return;
  float a0=0,a1=0,b0=0,b1=0;
  for(int i=0;i<m;++i) { float x=value<P>(a,sa,ga,i,t,kp,la); a0+=x; a1=fmaf(float(i+1),x,a1); }
  for(int j=0;j<n;++j) { float x=value<P>(b,sb,gb,j,t,kp,lb); b0+=x; b1=fmaf(float(j+1),x,b1); }
  tmp[t]=a0; tmp[k+t]=a1; tmp[2*k+t]=b0; tmp[3*k+t]=b1;
}
template<int P,class Layout> __global__ void propagate(const uint8_t* a,const uint8_t* sa,const float* ga,
    const uint8_t* b,const uint8_t* sb,const float* gb,const float* tmp,float* out,
    int m,int n,int k,int kp,Layout la,Layout lb) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=m+n) return;
  float s0=0,s1=0;
  if(idx<n) {
    for(int t=0;t<k;++t) { float x=value<P>(b,sb,gb,idx,t,kp,lb); s0=fmaf(tmp[t],x,s0); s1=fmaf(tmp[k+t],x,s1); }
    out[idx]=s0; out[n+idx]=s1;
  } else {
    int row=idx-n;
    for(int t=0;t<k;++t) { float x=value<P>(a,sa,ga,row,t,kp,la); s0=fmaf(x,tmp[2*k+t],s0); s1=fmaf(x,tmp[3*k+t],s1); }
    out[2*n+row]=s0; out[2*n+m+row]=s1;
  }
}
__global__ void observed(const float* c,const float* ref,float* residual,float* threshold,
    int m,int n,int stride,float atol,float rtol) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=m+n) return;
  float s0=0,s1=0; int p,q;
  if(idx<n) {
    for(int i=0;i<m;++i) {float x=c[i*stride+idx]; s0+=x; s1=fmaf(float(i+1),x,s1);}
    p=idx; q=n+idx;
  } else {
    int row=idx-n;
    for(int j=0;j<n;++j) {float x=c[row*stride+j]; s0+=x; s1=fmaf(float(j+1),x,s1);}
    p=2*n+row; q=2*n+m+row;
  }
  residual[p]=s0-ref[p]; residual[q]=s1-ref[q];
  threshold[p]=atol+rtol*fmaxf(fabsf(s0),fabsf(ref[p]));
  threshold[q]=atol+rtol*fmaxf(fabsf(s1),fabsf(ref[q]));
}
__device__ bool bad(float r,float t) { return !isfinite(r) || fabsf(r)>t; }
__global__ void decide(float* c,const float* r,const float* t,int64_t* status,int m,int n,int stride,bool correct) {
  if(threadIdx.x || blockIdx.x) return;
  int nc=0,nr=0,ci=-1,ri=-1; bool finite=true,detected=false;
  for(int z=0;z<2*(m+n);++z) {finite &= isfinite(r[z]); detected |= bad(r[z],t[z]);}
  status[0]=detected; status[1]=0; status[2]=-1; status[3]=-1; status[4]=0; status[5]=detected;
  if(!detected || !finite) return;
  for(int j=0;j<n;++j) if(bad(r[j],t[j]) || bad(r[n+j],t[n+j])) {++nc;ci=j;}
  for(int i=0;i<m;++i) if(bad(r[2*n+i],t[2*n+i]) || bad(r[2*n+m+i],t[2*n+m+i])) {++nr;ri=i;}
  if(nc!=1 || nr!=1) return;
  float ec=r[ci],er=r[2*n+ri];
  if(fabsf(ec)<=t[ci] || fabsf(er)<=t[2*n+ri] || fabsf(ec-er)>t[ci]+t[2*n+ri]) return;
  if(fabsf(r[n+ci]-(ri+1)*ec)>t[n+ci]+(ri+1)*t[ci]) return;
  if(fabsf(r[2*n+m+ri]-(ci+1)*er)>t[2*n+m+ri]+(ci+1)*t[2*n+ri]) return;
  status[1]=1; status[2]=ri; status[3]=ci;
  if(correct) {c[ri*stride+ci]-=(er+ec)*.5f;status[4]=1;}
}
__global__ void postcheck(const float* r,const float* t,int64_t* status,int count) {
  if(threadIdx.x || blockIdx.x || !status[4]) return;
  bool failed=false; for(int z=0;z<count;++z) failed |= bad(r[z],t[z]); status[5]=failed;
}
__global__ void flip_kernel(uint8_t* q,int64_t idx,int bit,int width,bool enabled,int64_t* meta) {
  uint32_t old=0;
  if(width==4) old=(q[idx/2]>>((idx%2)*4))&15;
  else for(int j=0;j<width/8;++j) old |= uint32_t(q[idx*(width/8)+j])<<(8*j);
  uint32_t next=old^(enabled ? 1U<<bit : 0U); meta[0]=old;meta[1]=next;
  if(width==4) {int shift=(idx%2)*4;q[idx/2]=(q[idx/2]&~(15U<<shift))|(next<<shift);}
  else for(int j=0;j<width/8;++j) q[idx*(width/8)+j]=uint8_t(next>>(8*j));
}
__global__ void set_mask(float* c,int idx,int bit,bool enabled) { c[idx]=__uint_as_float(enabled ? 1U<<bit : 0U); }

template<int P> std::vector<at::Tensor> pack_impl(at::Tensor x,bool tile,bool stochastic,int64_t seed) {
  int r=x.size(0),c=x.size(1),rp=pad128(r),cp=pad128(c);
  constexpr int bits=P==0 ? 32 : P==1 ? 16 : P==2 ? 8 : 4;
  auto opts=x.options().dtype(at::kByte);
  auto q=at::empty({int64_t(rp)*cp*bits/8},opts);
  auto layout=sf_layout<P==2 ? 32 : 16>(rp,cp);
  auto s=at::empty({P<2 ? 0 : int64_t(cute::cosize(layout))},opts);
  auto g=P==3 ? at::clamp_min(at::amax(at::abs(x))/(448.f*6.f),1e-30) : at::ones({},x.options());
  int groups=rp*cp/(P==2 ? 32 : P==3 ? 16 : 1);
  pack_kernel<P><<<(groups+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    x.data_ptr<float>(),q.data_ptr<uint8_t>(),s.data_ptr<uint8_t>(),g.data_ptr<float>(),
    r,c,rp,cp,tile,stochastic,uint32_t(seed),layout);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {q,s,g};
}
template<int P> at::Tensor decode_impl(at::Tensor q,at::Tensor s,at::Tensor g,int r,int c) {
  auto out=at::empty({r,c},q.options().dtype(at::kFloat));
  auto layout=sf_layout<P==2 ? 32 : 16>(pad128(r),pad128(c));
  decode_kernel<P><<<(r*c+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<uint8_t>(),s.data_ptr<uint8_t>(),
    g.data_ptr<float>(),out.data_ptr<float>(),r,c,pad128(c),layout);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
template<int P> at::Tensor reference_impl(at::Tensor a,at::Tensor sa,at::Tensor ga,at::Tensor b,at::Tensor sb,at::Tensor gb,int m,int n,int k) {
  auto tmp=at::empty({4*k},a.options().dtype(at::kFloat)),out=at::empty({2*(m+n)},tmp.options());
  auto la=sf_layout<P==2 ? 32 : 16>(pad128(m),pad128(k));
  auto lb=sf_layout<P==2 ? 32 : 16>(pad128(n),pad128(k));
  input_checks<P><<<(k+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    a.data_ptr<uint8_t>(),sa.data_ptr<uint8_t>(),ga.data_ptr<float>(),b.data_ptr<uint8_t>(),sb.data_ptr<uint8_t>(),gb.data_ptr<float>(),
    tmp.data_ptr<float>(),m,n,k,pad128(k),la,lb);
  propagate<P><<<(m+n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    a.data_ptr<uint8_t>(),sa.data_ptr<uint8_t>(),ga.data_ptr<float>(),b.data_ptr<uint8_t>(),sb.data_ptr<uint8_t>(),gb.data_ptr<float>(),
    tmp.data_ptr<float>(),out.data_ptr<float>(),m,n,k,pad128(k),la,lb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
} // namespace

std::vector<at::Tensor> pack_cuda(at::Tensor x,int64_t p,bool tile,bool stochastic,int64_t seed) {
  check(x,at::kFloat);TORCH_CHECK(x.dim()==2 && x.size(0)>0 && x.size(1)>0,"Expected nonempty matrix");
  check_shape(x.size(0),x.size(1));
  c10::cuda::CUDAGuard guard(x.device());
  switch(p) {case 0:return pack_impl<0>(x,tile,stochastic,seed);case 1:return pack_impl<1>(x,tile,stochastic,seed);
    case 2:return pack_impl<2>(x,tile,stochastic,seed);case 3:return pack_impl<3>(x,tile,stochastic,seed);}
  TORCH_CHECK(false,"Unknown precision");
}
at::Tensor decode_cuda(at::Tensor q,at::Tensor s,at::Tensor g,int64_t r,int64_t c,int64_t p) {
  check_operand(q,s,g,r,c,p);c10::cuda::CUDAGuard guard(q.device());
  switch(p) {case 0:return decode_impl<0>(q,s,g,r,c);case 1:return decode_impl<1>(q,s,g,r,c);
    case 2:return decode_impl<2>(q,s,g,r,c);case 3:return decode_impl<3>(q,s,g,r,c);}
  TORCH_CHECK(false,"Unknown precision");
}
at::Tensor reference_cuda(at::Tensor a,at::Tensor sa,at::Tensor ga,at::Tensor b,at::Tensor sb,at::Tensor gb,int64_t m,int64_t n,int64_t k,int64_t p) {
  check_operand(a,sa,ga,m,k,p);check_operand(b,sb,gb,n,k,p);TORCH_CHECK(a.device()==b.device(),"Device mismatch");
  c10::cuda::CUDAGuard guard(a.device());
  switch(p) {case 0:return reference_impl<0>(a,sa,ga,b,sb,gb,m,n,k);case 1:return reference_impl<1>(a,sa,ga,b,sb,gb,m,n,k);
    case 2:return reference_impl<2>(a,sa,ga,b,sb,gb,m,n,k);case 3:return reference_impl<3>(a,sa,ga,b,sb,gb,m,n,k);}
  TORCH_CHECK(false,"Unknown precision");
}
std::vector<at::Tensor> verify_cuda(at::Tensor c,at::Tensor ref,double atol,double rtol,bool correct) {
  TORCH_CHECK(c.is_cuda() && c.scalar_type()==at::kFloat && c.dim()==2 && c.stride(1)==1,"Expected FP32 output");
  check(ref,at::kFloat);check_shape(c.size(0),c.size(1));
  TORCH_CHECK(ref.device()==c.device() && c.stride(0)>=c.size(1),"Invalid checksum device or output stride");
  TORCH_CHECK(std::isfinite(atol) && std::isfinite(rtol) && atol>=0 && rtol>=0,"Invalid tolerance");
  c10::cuda::CUDAGuard guard(c.device());
  int m=c.size(0),n=c.size(1);TORCH_CHECK(ref.numel()==2*(m+n),"Checksum length mismatch");
  auto r=at::empty_like(ref),t=at::empty_like(ref),status=at::empty({6},c.options().dtype(at::kLong));
  auto stream=at::cuda::getCurrentCUDAStream();
  observed<<<(m+n+255)/256,256,0,stream>>>(c.data_ptr<float>(),ref.data_ptr<float>(),r.data_ptr<float>(),t.data_ptr<float>(),m,n,c.stride(0),atol,rtol);
  decide<<<1,1,0,stream>>>(c.data_ptr<float>(),r.data_ptr<float>(),t.data_ptr<float>(),status.data_ptr<int64_t>(),m,n,c.stride(0),correct);
  if(correct) {
    auto post=at::empty_like(ref),pt=at::empty_like(ref);
    observed<<<(m+n+255)/256,256,0,stream>>>(c.data_ptr<float>(),ref.data_ptr<float>(),post.data_ptr<float>(),pt.data_ptr<float>(),m,n,c.stride(0),atol,rtol);
    postcheck<<<1,1,0,stream>>>(post.data_ptr<float>(),pt.data_ptr<float>(),status.data_ptr<int64_t>(),ref.numel());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {status,r,t};
}
at::Tensor flip_cuda(at::Tensor q,int64_t index,int64_t bit,int64_t width,bool enabled) {
  check(q,at::kByte);TORCH_CHECK((width==4 || width==8 || width==16 || width==32) && index>=0 && (index+1)*width<=q.numel()*8 && bit>=0 && bit<width,"Invalid bit event");
  c10::cuda::CUDAGuard guard(q.device());auto meta=at::empty({2},q.options().dtype(at::kLong));
  flip_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<uint8_t>(),index,bit,width,enabled,meta.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return meta;
}
at::Tensor mask_cuda(int64_t m,int64_t n,int64_t row,int64_t col,int64_t bit,bool enabled,at::Tensor like) {
  check_shape(m,n);TORCH_CHECK(like.is_cuda(),"Expected CUDA tensor");
  TORCH_CHECK(row>=0 && row<m && col>=0 && col<n && bit>=0 && bit<32,"Invalid accumulator event");
  c10::cuda::CUDAGuard guard(like.device());auto out=at::zeros({pad128(m),pad128(n)},like.options().dtype(at::kFloat));
  set_mask<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(),row*pad128(n)+col,bit,enabled);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
