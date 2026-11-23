#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
static CUgreenCtx green;
extern "C" int make_green(int requested, unsigned long long *stream, int *info){
 CUdevResource input{},part{},rem{},check{};unsigned n=1;CUdevResourceDesc desc;CUstream s;
 #define CHECK(x) do {CUresult e=(x);if(e){const char* txt;cuGetErrorString(e,&txt);fprintf(stderr,"%s: %s\n",#x,txt);return e;}}while(0)
 CHECK(cuDeviceGetDevResource(0,&input,CU_DEV_RESOURCE_TYPE_SM));
 info[0]=input.sm.smCount;info[1]=input.sm.minSmPartitionSize;info[2]=input.sm.smCoscheduledAlignment;
 CHECK(cuDevSmResourceSplitByCount(&part,&n,&input,&rem,0,requested));
 CHECK(cuDevResourceGenerateDesc(&desc,&part,1));
 CHECK(cuGreenCtxCreate(&green,desc,0,CU_GREEN_CTX_DEFAULT_STREAM));
 CHECK(cuGreenCtxGetDevResource(green,&check,CU_DEV_RESOURCE_TYPE_SM));info[3]=check.sm.smCount;
 CHECK(cuGreenCtxStreamCreate(&s,green,CU_STREAM_NON_BLOCKING,0));*stream=(unsigned long long)s;return 0;
}
extern "C" int stream_sms(unsigned long long st){CUcontext c;CUgreenCtx g;CUdevResource res{};if(cuStreamGetCtx_v2((CUstream)st,&c,&g))return -1;CUresult e=g?cuGreenCtxGetDevResource(g,&res,CU_DEV_RESOURCE_TYPE_SM):cuCtxGetDevResource(c,&res,CU_DEV_RESOURCE_TYPE_SM);return e?-(int)e:(int)res.sm.smCount;}
__global__ void probe(int *seen){unsigned id;asm("mov.u32 %0, %%smid;":"=r"(id));if(threadIdx.x==0 && id<1024)atomicExch(seen+id,1);unsigned long long t=clock64();while(clock64()-t<200000){} }
extern "C" int probe_sms(unsigned long long stream,void *seen){probe<<<4096,128,0,(cudaStream_t)stream>>>((int*)seen);return (int)cudaGetLastError();}
