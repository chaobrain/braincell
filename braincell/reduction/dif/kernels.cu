// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <mutex>
#include "brainevent/common.h"

#ifndef REDUCTION_BLOCK_THREADS
#define REDUCTION_BLOCK_THREADS 256
#endif
#if REDUCTION_INDEX_BITS == 16
using EventIndex = int16_t;
#else
using EventIndex = int32_t;
#endif

// A read-only kernel argument can be referenced without a per-thread copy.
#if (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 7)) \
    && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
#define REDUCTION_GRID_CONSTANT __grid_constant__
#else
#define REDUCTION_GRID_CONSTANT
#endif

namespace response {
namespace cg = cooperative_groups;
using I = int64_t;
constexpr int block_samples = 128;
// Every event retains independent queries for the main and monitor trajectories.
constexpr int event_branches = 2;
constexpr unsigned warp_mask = 0xffffffffu;
// Exact binary64 zero predicate, including both signs of zero.
// NaN, infinity and subnormals remain nonzero, matching an IEEE comparison against zero.
__device__ __forceinline__ bool nonzero(double value) {
    return ((static_cast<uint32_t>(__double2hiint(value)) & 0x7fffffffu)
            | static_cast<uint32_t>(__double2loint(value))) != 0u;
}
__device__ int lane() { return threadIdx.x & 31; }
template<class T> __device__ T broadcast(T v) { return __shfl_sync(warp_mask, v, 0); }
__device__ I broadcast(I v) { return static_cast<I>(__shfl_sync(warp_mask, static_cast<long long>(v), 0)); }
template<class T> __device__ T low(T a, T b) { return a < b ? a : b; }
template<class T> __device__ T high(T a, T b) { return a > b ? a : b; }

// Curve banks share one allocation, preserving their calibrated row layout.
// Every warp reads contiguous samples; only bank base addresses differ.
enum Index { RS_PTR, RP_PTR, PS_PTR, PP_PTR, BS_PTR, BP_PTR,
    RS_SUPPORT, PS_SUPPORT, BS_SUPPORT, HISTORY_SUPPORT, TAU_LO, TAU_HI,
    RELEASE_LO, RELEASE_HI, TAU, EVENT, RELEASE,
    PAIR_SOURCE_PTR, PAIR_SOURCES, PAIR_TARGETS, PAIR_REVERSE, BS_INIT_PTR, BP_INIT_PTR, QUERY_SUPPORT, INDEX_COUNT };
enum Real { REST_GRID, POST_GRID, TAU_RATIO, RELEASE_RATIO, TAU_INVERSE, EVENT_INVERSE, RELEASE_INVERSE, BS_INITIAL, BP_INITIAL, MA_REVERSAL };
enum Header { LOCATIONS, RESPONSE_STEPS, TAUS, EVENTS, RELEASES, REST_STATES, POST_STATES,
              ETA_LENGTH, REBASE_AGE, REFRACTORY_AGE, TROUGH_AGE, PAIRS, AXIS_LUT_SIZE, HEADER_SIZE };

using Value = double4;
using Future = double2; // Only recovered G/I are persistent.
struct Table {
    const I* indices;
    const double* real;
    const double *rs,*rp,*ps,*pp,*bs,*bp;
    const double* eta;
    static constexpr int metadata_size=HEADER_SIZE+INDEX_COUNT+MA_REVERSAL+1;
    I metadata[metadata_size];
    double scalars[7];
    __device__ const I* ix(Index field) const { return indices + metadata[HEADER_SIZE+field]; }
    __device__ const double* fp(Real field) const { return real + metadata[HEADER_SIZE+INDEX_COUNT+field]; }
    // Per-location axes; all stored rows keep the original rectangular stride.
    __device__ const I* ix(Index field,int loc) const {
        int width=(field==TAU?taus():(field==EVENT?events():(field==RELEASE?releases():int(metadata[AXIS_LUT_SIZE]))));
        return ix(field)+loc*width;
    }
    __device__ const double* fp(Real field,int loc) const {
        int width=(field==TAU_INVERSE?taus()-1:(field==EVENT_INVERSE?events()-1:(field==RELEASE_INVERSE?releases()-1:int(metadata[AXIS_LUT_SIZE]))));
        return fp(field)+loc*width;
    }
    __device__ int locations() const { return metadata[LOCATIONS]; }
    __device__ int pairs() const { return metadata[PAIRS]; }
    __device__ int horizon() const { return metadata[RESPONSE_STEPS]; }
    __device__ int taus() const { return metadata[TAUS]; }
    __device__ int events() const { return metadata[EVENTS]; }
    __device__ int releases() const { return metadata[RELEASES]; }
};

struct EventAttributes {
    double ratio[event_branches];
    EventIndex state[event_branches];
};
struct Event {
    I birth, prefix_count, prefix_birth;
    EventAttributes attributes;
};

enum JobField { ORDINARY_COUNT, ORDINARY_CURSOR, REBASE_COUNT, REBASE_CURSOR, JOB_HEADER };

struct Jobs {
    I* data;
    int count;
    __device__ I& operator[](JobField field) const { return data[field]; }
    __host__ __device__ I* ordinary() const { return data + JOB_HEADER; }
    __host__ __device__ I* rebase() const { return ordinary() + count; }
};
// A shared arena avoids the device heap's general-purpose allocation search.
// Only warp leaders enter these short per-size-class critical sections.
struct Arena {
    static constexpr int minimum_shift = 3;
    static constexpr int minimum_bytes = 1 << minimum_shift;
    static constexpr int bins = sizeof(I) * CHAR_BIT - 1 - minimum_shift;
    static constexpr int words = 2 + 4 * bins;
    unsigned char* data;
    I* control;
    I capacity;
    __device__ void request(I bytes) const {
        atomicAdd(reinterpret_cast<unsigned long long*>(control + 1 + 3 * bins + bin(bytes)), 1ull);
    }
    __device__ int bin(I bytes) const { return high(0, 64 - __clzll(bytes - 1) - minimum_shift); }
    __device__ void lock(int k) const {
        auto* p = reinterpret_cast<unsigned long long*>(control + 1 + bins + k);
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> flag(*p);
        unsigned long long expected = 0;
        while (!flag.compare_exchange_weak(expected, 1ull, cuda::memory_order_acquire,
                                           cuda::memory_order_relaxed))
            expected = 0;
    }
    __device__ void unlock(int k) const {
        auto* p = reinterpret_cast<unsigned long long*>(control + 1 + bins + k);
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> flag(*p);
        flag.store(0ull, cuda::memory_order_release);
    }
    __device__ I allocate(I bytes) const {
        int k = bin(bytes);
        lock(k);
        volatile I* heads = control + 1;
        I offset = heads[k];
        if (offset >= 0) {
            heads[k] = *reinterpret_cast<volatile I*>(data + offset);
            --control[1 + 2 * bins + k];
        }
        unlock(k);
        I span = I(1) << (k + minimum_shift);
        if (offset < 0)
            offset = atomicAdd(reinterpret_cast<unsigned long long*>(control),
                               static_cast<unsigned long long>(span));
        assert(offset + span <= capacity && "DIF arena exhausted; increase gpu_heap_bytes before running");
        return offset;
    }
    __device__ void release(I offset, I bytes) const {
        int k = bin(bytes);
        lock(k);
        volatile I* heads = control + 1;
        *reinterpret_cast<volatile I*>(data + offset) = heads[k];
        heads[k] = offset;
        ++control[1 + 2 * bins + k];
        unlock(k);
    }
};
struct History {
    I offset, before_count, before_birth;
    int start, size, capacity;
    bool has_wide;
    unsigned char* base;
    // Each column has capacity entries; branch attributes use one column per branch.
    // Capacity is a power of two >= 4, so every I/double column is 8-byte aligned.
    __host__ __device__ static I storage_bytes(int capacity) {
        return I(capacity) * (3 * sizeof(I) + event_branches *
                             (sizeof(double) + sizeof(EventIndex)));
    }
    __device__ int physical(int i) const { return (start + i) & (capacity - 1); }
    __device__ I* coordinates() const { return reinterpret_cast<I*>(base + offset); }
    __device__ double* ratios() const {
        return reinterpret_cast<double*>(coordinates() + I(3) * capacity);
    }
    __device__ EventIndex* states() const {
        return reinterpret_cast<EventIndex*>(ratios() + I(event_branches) * capacity);
    }
    __device__ I& birth_at(int i) const { return coordinates()[physical(i)]; }
    __device__ I& prefix_count_at(int i) const { return coordinates()[I(capacity) + physical(i)]; }
    __device__ I& prefix_birth_at(int i) const { return coordinates()[I(2) * capacity + physical(i)]; }
    __device__ double& ratio(int i, int branch) const {
        return ratios()[I(branch) * capacity + physical(i)];
    }
    __device__ EventIndex& state(int i, int branch) const {
        return states()[I(branch) * capacity + physical(i)];
    }
    // Complete values are needed only for resize and isolated verification.
    // Concurrent runtime readers use the field accessors above, never whole-Event loads.
    __device__ Event at(int i) const {
        Event result{birth_at(i), prefix_count_at(i), prefix_birth_at(i), {}};
        for (int branch = 0; branch < event_branches; ++branch) {
            result.attributes.ratio[branch] = ratio(i, branch);
            result.attributes.state[branch] = state(i, branch);
        }
        return result;
    }
    __device__ void assign(int i, const Event& value) const {
        birth_at(i) = value.birth;
        prefix_count_at(i) = value.prefix_count;
        prefix_birth_at(i) = value.prefix_birth;
        for (int branch = 0; branch < event_branches; ++branch) {
            ratio(i, branch) = value.attributes.ratio[branch];
            state(i, branch) = value.attributes.state[branch];
        }
    }
    __device__ I count(int i) const {
        return prefix_count_at(i) - (i ? prefix_count_at(i - 1) : before_count);
    }
    __device__ void prune(I oldest, const Arena& arena) {
        while (size && birth_at(0) < oldest) {
            before_count = prefix_count_at(0);
            before_birth = prefix_birth_at(0);
            start = (start + 1) & (capacity - 1);
            --size;
        }
        if (!size) {
            start = 0;
            has_wide = false;
            if (capacity) arena.release(offset, storage_bytes(capacity));
            capacity = 0;
        }
    }
    __device__ void request_append(const Arena& arena) const {
        if (size == capacity) arena.request(storage_bytes(high(4, 2 * capacity)));
    }
    __device__ void append(I birth, I count, const Arena& arena) {
        I n = size ? prefix_count_at(size - 1) : before_count;
        I s = size ? prefix_birth_at(size - 1) : before_birth;
        if (size == capacity) {
            int next = high(4, 2 * capacity);
            I next_offset = arena.allocate(storage_bytes(next));
            History grown = *this;
            grown.offset = next_offset;
            grown.capacity = next;
            grown.start = 0;
            for (int i = 0; i < size; ++i) grown.assign(i, at(i));
            if (capacity) arena.release(offset, storage_bytes(capacity));
            offset = next_offset;
            capacity = next;
            start = 0;
        }
        birth_at(size) = birth;
        prefix_count_at(size) = n + count;
        prefix_birth_at(size) = s + count * birth;
        has_wide |= count > UINT16_MAX;
        for (int branch = 0; branch < event_branches; ++branch) {
            ratio(size, branch) = NAN;
            state(size, branch) = 0;
        }
        ++size;
    }
    __device__ void release(const Arena& arena) {
        if (capacity) arena.release(offset, storage_bytes(capacity));
        *this = {};
    }
    __device__ int rank(I birth, int first, int last) const {
        while (first < last) {
            int middle = (first + last) / 2;
            if (birth_at(middle) < birth) first = middle + 1;
            else last = middle;
        }
        return first;
    }
    __device__ int rank(I birth) const { return rank(birth, 0, size); }
    __device__ int lower(I birth, int first, int last) const {
        return broadcast(lane() == 0 ? rank(birth, first, last) : 0);
    }
    __device__ int lower(I birth) const {
        return broadcast(lane() == 0 ? rank(birth) : 0);
    }
    __device__ int rank_before(I birth, int first, int last) const {
        int result = last;
        if (first < last && birth_at(last - 1) >= birth) {
            int distance = 1;
            while (last - distance > first && birth_at(last - distance - 1) >= birth)
                distance *= 2;
            result = rank(birth, high(first, last - distance), last - distance / 2);
        }
        return result;
    }
    __device__ void read_statistics(int first, int last, I& nc, I& ns) const {
        nc = last ? prefix_count_at(last - 1) : before_count;
        ns = last ? prefix_birth_at(last - 1) : before_birth;
        nc -= first ? prefix_count_at(first - 1) : before_count;
        ns -= first ? prefix_birth_at(first - 1) : before_birth;
    }
};
struct Branch {
    I last_spike, rebase_at, rest_since, ordinary_since, future;
    double previous,u;
    bool post;
    int query_count;
};
struct Neuron {
    Branch branch[2];
    I next_step, pending_start;
    int main;
    bool pending, confirmed;
};
__device__ void near(const double* grid, int size, double value, int& a, int& b, double& ratio) {
    if (lane() == 0) {
        int first = 0, last = size;
        while (first < last) {
            int mid = (first + last) / 2;
            if (grid[mid] < value) first = mid + 1;
            else last = mid;
        }
        if (first < size && grid[first] == value) { a = b = first; ratio = 0.; }
        else {
            b = low(size - 1, high(1, first)); a = b - 1;
            ratio = (value - grid[a]) / (grid[b] - grid[a]);
        }
    }
    a = broadcast(a); b = broadcast(b); ratio = broadcast(ratio);
}
__device__ bool needs_rebase(const Table& d, const Branch& state, I step) {
    return (state.last_spike >= 0 && step == state.rebase_at)
        || (step % block_samples == 0 && state.rebase_at >= 0 && step > state.rebase_at
            && step - state.rebase_at < d.horizon());
}

// Joint count-error criterion in the DIF affine information space.
// q comes only from existing joint G/I sums; no xi/third/additional table input.
// Ordinary finite magnitudes need one sqrt and one exp.
// Extreme/subnormal magnitudes retain the scaled hypot and exp-log paths.
__device__ __forceinline__ double2 recover_joint_conductance(
    double G1,double I1,double G2,double I2,double beta,double sigma){
    if(G2==0.0&&I2==0.0)return make_double2(G1,I1);
    double x0=sigma*G1,x1=fma(-beta,G1,I1);
    double y0=sigma*G2,y1=fma(-beta,G2,I2);
    double xx=fma(x0,x0,x1*x1),yy=fma(y0,y0,y1*y1),q;
    if(xx>=2.2250738585072014e-308&&yy>=2.2250738585072014e-308&&isfinite(xx)&&isfinite(yy)){
        q=sqrt(yy/xx);
    }else{
        double nx=hypot(x0,x1);if(nx==0.0)return make_double2(0.0,0.0);
        q=hypot(y0,y1)/nx;
    }
    if(isinf(q))return make_double2(0.0,0.0);
    double a=exp(-q),b=q<700.0?(1.0+q)*a:exp(log1p(q)-q);
    return make_double2(fma(b,G1,a*G2),fma(b,I1,a*I2));
}

__device__ __forceinline__ double2 conditional_recover(const Table& d,Value v){
#if !REDUCTION_HIGH_ORDER
    return make_double2(v.x+v.z,v.y+v.w);
#else
    return recover_joint_conductance(v.x,v.y,v.z,v.w,d.scalars[5],d.scalars[6]);
#endif
}

__device__ double voltage(const Table& d,Branch& state,const Future* future,I step){
    I age=step-state.last_spike;
    double eta=state.last_spike>=0&&age>=0&&age<d.metadata[ETA_LENGTH]?d.eta[age]:0.;
    if(state.last_spike>=0&&step<state.rebase_at){state.u=0.;return d.scalars[0]+eta;}
    Future value=future?future[step%block_samples]:Future{};
    double G=value.x,Irate=value.y,before=state.u,rate=d.scalars[3]+G,dt=d.scalars[4],z=rate*dt;
    double decay,weight;
    if(fabs(z)<=0.69314718055994530942){
        double delta=expm1(-z);decay=1.+delta;
        weight=fabs(z)<1e-10?dt*(1.-.5*z):-delta/rate;
    }else{decay=exp(-z);weight=(1.-decay)/rate;}
    state.u=before*decay+(Irate-G*eta)*weight;
    return d.scalars[0]+eta+before;
}

__device__ void ma_first_moment(const History& h,int first,int last,I center,double& n,double& delta){
    I ni,bi;h.read_statistics(first,last,ni,bi);n=double(ni);
    delta=double(I(uint64_t(center)*uint64_t(ni)-uint64_t(bi)));
#if REDUCTION_INDEX_BITS == 16
    if(!h.has_wide)return;
#endif
    if(first<last){
        double far=high(fabs(double(center-h.birth_at(first))),fabs(double(center-h.birth_at(last-1))));
        if(n*far>=9223372036854775808.){
            delta=0.;for(int j=first;j<last;++j)delta=fma(double(h.count(j)),double(center-h.birth_at(j)),delta);
        }
    }
}

__device__ double ma_initial(const Table& d,bool pair,int row) {
    const I* p=d.ix(pair?BP_INIT_PTR:BS_INIT_PTR);
    return p[row+1]>p[row]?d.fp(pair?BP_INITIAL:BS_INITIAL)[p[row]]:0.;
}
__device__ int ma_event_interval(const Table& d,int loc,int age) {
    // Event ages and input intervals share each location's time grid.
    int upper=d.events()-1;
    if(age<=0)return 0;
    if(age>=__ldg(d.ix(EVENT,loc)+upper))return upper-1;
    return low<int>(int(__ldg(d.ix(TAU_LO,loc)+age)),upper-1);
}

struct Coefficients { double a, b, c; };
struct Span {
    I a, b, c, na, nb, nc;
    double wa, wb, wc;
    int begin, end;
};
struct RowWriter {
    double* output;
    I birth;
    const double* bank;
    Span* spans;
    double* nodes;
    int begin, end;
    cg::thread_block_tile<32> warp;

    __device__ void append(Span span, bool valid);

    __device__ double contract(int node, I length, int first, int last, double weight) {
        if (first >= length) weight = 0.;
        bool shared = nonzero(weight) && first == begin && last >= low<I>(end, length);
        if(!__any_sync(warp_mask,shared))return weight;
        auto group = cg::labeled_partition(warp, shared ? node : -1);
        double sum = shared ? cg::reduce(group, weight, cg::plus<double>()) : 0.;
        if (shared && group.thread_rank() == 0) nodes[node] += sum;
        __syncwarp();
        return shared ? 0. : weight;
    }
};

__device__ void panel_moments(const History& h,int first,int last,I origin,double inv,double& n,double& r){
    ma_first_moment(h,first,last,origin,n,r);r*=inv;
}
__device__ double accumulate_curve(double value, const double* bank, I offset, I length, int k, double weight) {
    return nonzero(weight) && k < length ? fma(weight, __ldg(bank + offset + k), value) : value;
}
template<int mask>
__device__ void expand_active(double* output, I birth, int begin, int end, const double* bank,
                              I a, double wa, I b, double wb, I c, double wc) {
    int cursor = (birth + begin) & 31;
    int first = begin + ((lane() - cursor) & 31);
    int slot = (birth + first) & (block_samples - 1);
    for (int k = first; k < end; k += 32, slot = (slot + 32) & (block_samples - 1)) {
        double value = output[slot];
        if (mask & 1) value = fma(wa, __ldg(bank + a + k), value);
        if (mask & 2) value = fma(wb, __ldg(bank + b + k), value);
        if (mask & 4) value = fma(wc, __ldg(bank + c + k), value);
        output[slot] = value;
    }
}

__device__ void expand(double* output, I birth, int begin, int end, const double* bank,
                       I a, I na, double wa, I b = 0, I nb = 0, double wb = 0.,
                       I c = 0, I nc = 0, double wc = 0.) {
    // Support and nonzero weights are constant within each interval. Resolve
    // them once, then run the same FP64 sum without per-sample curve checks.
    while (begin < end) {
        int mask = 0, stop = end;
        if (nonzero(wa) && begin < na) { mask |= 1; stop = low<I>(stop, na); }
        if (nonzero(wb) && begin < nb) { mask |= 2; stop = low<I>(stop, nb); }
        if (nonzero(wc) && begin < nc) { mask |= 4; stop = low<I>(stop, nc); }
        switch (mask) {
            case 1: expand_active<1>(output, birth, begin, stop, bank, a, wa, b, wb, c, wc); break;
            case 2: expand_active<2>(output, birth, begin, stop, bank, a, wa, b, wb, c, wc); break;
            case 3: expand_active<3>(output, birth, begin, stop, bank, a, wa, b, wb, c, wc); break;
            case 4: expand_active<4>(output, birth, begin, stop, bank, a, wa, b, wb, c, wc); break;
            case 5: expand_active<5>(output, birth, begin, stop, bank, a, wa, b, wb, c, wc); break;
            case 6: expand_active<6>(output, birth, begin, stop, bank, a, wa, b, wb, c, wc); break;
            case 7: expand_active<7>(output, birth, begin, stop, bank, a, wa, b, wb, c, wc); break;
            default: return;
        }
        begin = stop;
    }
}

__device__ void expand_four(double* output, I birth, const double* bank, const Span* spans) {
    int cursor = (birth + spans[0].begin) & 31;
    int first = spans[0].begin + ((lane() - cursor) & 31);
    int slot = (birth + first) & (block_samples - 1);
    for (int k = first; k < spans[0].end; k += 32, slot = (slot + 32) & (block_samples - 1)) {
        double value = output[slot];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const Span& s = spans[i];
            value = accumulate_curve(value, bank, s.a, s.na, k, s.wa);
            value = accumulate_curve(value, bank, s.b, s.nb, k, s.wb);
            value = accumulate_curve(value, bank, s.c, s.nc, k, s.wc);
        }
        output[slot] = value;
    }
}

__device__ void RowWriter::append(Span span, bool valid) {
    unsigned mask = __ballot_sync(warp_mask, valid);
    int count = __popc(mask);
    if (!count) return;
    if (valid) spans[__popc(mask & ((1u << lane()) - 1))] = span;
    __syncwarp();
    for (int i = 0; i < count;) {
        const Span& s = spans[i];
        bool four = i + 3 < count;
        for (int j = 1; j < 4 && four; ++j)
            four = spans[i + j].begin == s.begin && spans[i + j].end == s.end;
        if (four) {
            expand_four(output, birth, bank, spans + i);
            i += 4;
        } else {
            expand(output, birth, s.begin, s.end, bank,
                   s.a, s.na, s.wa, s.b, s.nb, s.wb, s.c, s.nc, s.wc);
            ++i;
        }
    }
    __syncwarp();
}

struct OrdinaryWeights {
    double* values;
    int begin, end;
    __device__ double collect(int node, I length, int first, int last, double weight) const {
        if (values && first == begin && last >= low<I>(end, length)) {
            if (lane() == 0) values[node] += weight;
            return 0.;
        }
        return weight;
    }
};

__device__ void ordinary_panel(const Table& d, const History& h, double* single, double* pair,
                              I birth, int location, int old, int pair_row, I count, bool post,
                              int a, double vr, int begin, int end, int first, int last,
                              int* boundaries, double* coefficients, Span* spans) {
    const I* ptr = d.ix(post ? PP_PTR : RP_PTR);
    const I* support = d.ix(post ? PS_SUPPORT : RS_SUPPORT);
    const double* bank = post ? d.pp : d.rp;
    I stride = ptr[d.pairs() * d.taus()];
    int row = pair_row * d.taus();
    int b = !nonzero(vr) ? a : a + 1;
    double scale = double(count), rest_gain = scale * (1. - vr), voltage_gain = scale * vr;
    if (old == location) {
        const I* sp = d.ix(post ? PS_PTR : RS_PTR);
        I length = support[location] + 1;
        // `single` now points directly to the existing joint cohort sums.
        const double* sb=post?d.ps:d.rs;
        I sa=sp[location]+a*length,sb_offset=sp[location]+b*length;
        double eps=d.fp(MA_REVERSAL)[location]-d.scalars[0];
        int first_sample=begin+((lane()-int(birth+begin))&31);
        for(int k=first_sample;k<low<I>(end,length);k+=32){
            double value=0.;
            if(nonzero(rest_gain))value=fma(rest_gain,__ldg(sb+sa+k),value);
            if(nonzero(voltage_gain))value=fma(voltage_gain,__ldg(sb+sb_offset+k),value);
            int slot=(birth+k)&(block_samples-1);
            single[slot]+=value;
            single[block_samples+slot]+=eps*value;
        }
        if (count > 1) {
            I p = ptr[row], n = ptr[row + 1] - p;
            double q = scale * double(count - 1) * .5;
            int stop = low<I>(end, low(n, support[location] + 1));
            OrdinaryWeights weights{coefficients, begin, end};
            expand(pair, birth, begin, stop, bank,
                   a * stride + p, n, weights.collect(0, n, begin, stop, q * (1. - vr)),
                   b * stride + p, n, weights.collect(b == a ? 0 : d.taus(), n, begin, stop, q * vr));
        }
    }
    first=h.rank(birth-d.ix(QUERY_SUPPORT)[old],first,last);
    // Every interpolation boundary is located once, with independent lanes.
    for (int node = lane(); node < d.taus(); node += 32)
        boundaries[node] = h.rank_before(birth - d.ix(TAU,old)[node] + 1, first, low(last, boundaries[node]));
    __syncwarp();
    RowWriter writer{pair, birth, bank, spans, nullptr, begin, end,
                     cg::tiled_partition<32>(cg::this_thread_block())};
    double previous_upper = 0.;
    for (int batch = 0; batch < d.taus(); batch += 32) {
        int node = batch + lane(), lo = 0, hi = 0;
        int bulk = begin, stop = begin;
        I p = 0, q = 0, np = 0, nq = 0;
        double inverse_width = 0.;
        Coefficients w{};
        if (node + 1 < d.taus()) {
            lo = boundaries[node + 1]; hi = boundaries[node];
            if (lo < hi) {
                p = ptr[row + node]; q = ptr[row + node + 1];
                np = q - p; nq = ptr[row + node + 2] - q;
                stop = low<I>(end, low<I>(support[location] + 1,
                    low<I>(support[old] - birth + h.birth_at(hi - 1) + 1, high(np, nq))));
                bulk = low<I>(stop, support[old] - birth + h.birth_at(lo) + 1);
                inverse_width = d.fp(TAU_INVERSE,old)[node];
                double cn, ratio;
                panel_moments(h,lo, hi, birth - d.ix(TAU,old)[node], inverse_width, cn, ratio);
                w = {fma(cn, rest_gain, -scale * ratio), cn * voltage_gain, scale * ratio};
            }
        }
        bool lower_full = bulk > begin && bulk >= low<I>(end, np);
        bool upper_full = bulk > begin && bulk >= low<I>(end, nq);
        double lower = lower_full ? w.a : 0., upper = upper_full ? w.c : 0.;
        double carried = __shfl_up_sync(warp_mask, upper, 1);
        if (lane() == 0) carried = previous_upper;
        previous_upper = __shfl_sync(warp_mask, upper, 31);
        if (node < d.taus()) {
            coefficients[node] += lower + carried;
            coefficients[node + d.taus()] += lower_full ? w.b : 0.;
        }
        Span span{a * stride + p, b * stride + p, a * stride + q, np, np, nq,
                  lower_full ? 0. : w.a, lower_full ? 0. : w.b, upper_full ? 0. : w.c,
                  begin, bulk};
        writer.append(span, bulk > begin && (nonzero(span.wa) || nonzero(span.wb) || nonzero(span.wc)));
        int k = high(begin, bulk), cut = lo;
        if (k < stop) cut = h.rank(birth + k - support[old], lo, hi);
        while (__any_sync(warp_mask, k < stop)) {
            Span tail{};
            if (k < stop) {
                int next = low<I>(stop, support[old] - birth + h.birth_at(cut) + 1);
                double cn, ratio;
                panel_moments(h,cut, hi, birth - d.ix(TAU,old)[node], inverse_width, cn, ratio);
                double wc = scale * ratio;
                tail = {a * stride + p, b * stride + p, a * stride + q, np, np, nq,
                        fma(cn, rest_gain, -wc), cn * voltage_gain, wc, k, next};
                k = next; ++cut;
            }
            writer.append(tail, tail.begin < tail.end);
        }
    }
    __syncwarp();
}

struct Curve { I offset; int length; double weight; };

template<int count>
__device__ void sum_curves(double (&values)[block_samples / 32], int first, int end,
                           const double* bank, const Curve (&curves)[4]) {
    int common_end = end;
    #pragma unroll
    for (int c = 0; c < count; ++c) common_end = low(common_end, curves[c].length);
    const double* source[count];
    #pragma unroll
    for (int c = 0; c < count; ++c) source[c] = bank + curves[c].offset + first;
    if (common_end - first > block_samples - 32) {
        #pragma unroll
        for (int c = 0; c < count; ++c) {
            double read[block_samples / 32];
            #pragma unroll
            for (int k = 0; k < block_samples / 32; ++k) read[k] = __ldg(source[c] + 32 * k);
            #pragma unroll
            for (int k = 0; k < block_samples / 32; ++k) values[k] = fma(curves[c].weight, read[k], values[k]);
        }
        return;
    }
    #pragma unroll
    for (int k = 0; k < block_samples / 32; ++k) {
        int sample = first + 32 * k;
        if (sample < common_end) {
            double read[count];
            #pragma unroll
            for (int c = 0; c < count; ++c) read[c] = __ldg(source[c] + 32 * k);
            #pragma unroll
            for (int c = 0; c < count; ++c) values[k] = fma(curves[c].weight, read[c], values[k]);
        } else if (sample < end) {
            #pragma unroll
            for (int c = 0; c < count; ++c)
                if (sample < curves[c].length)
                    values[k] = fma(curves[c].weight, __ldg(source[c] + 32 * k), values[k]);
        }
    }
}

__device__ void add_curve_batch(double (&values)[block_samples / 32], int first, int end,
                                const double* bank, Curve own) {
    unsigned active = __ballot_sync(warp_mask, nonzero(own.weight));
    while (active) {
        Curve curves[4];
        int count = low(4, __popc(active));
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int source = active ? __ffs(active) - 1 : 0;
            curves[c] = {
                static_cast<I>(__shfl_sync(warp_mask, static_cast<long long>(own.offset), source)),
                __shfl_sync(warp_mask, own.length, source),
                __shfl_sync(warp_mask, own.weight, source),
            };
            active &= active - 1;
        }
        switch (count) {
            case 1: sum_curves<1>(values, first, end, bank, curves); break;
            case 2: sum_curves<2>(values, first, end, bank, curves); break;
            case 3: sum_curves<3>(values, first, end, bank, curves); break;
            case 4: sum_curves<4>(values, first, end, bank, curves); break;
        }
    }
}

__device__ __noinline__ void render_ordinary(const Table& d, double* output, I birth, int pair_row,
                                bool post, int state, int begin, int end, const double* coefficients,double* cohort,double eps) {
    const I* ptr = d.ix(post ? PP_PTR : RP_PTR);
    const double* bank = post ? d.pp : d.rp;
    I stride = ptr[d.pairs() * d.taus()];
    int row = pair_row * d.taus();
    int first = begin + ((lane() - int(birth + begin)) & 31);
    int slot = (birth + first) & (block_samples - 1);
    double values[block_samples / 32];
    #pragma unroll
    for (int k = 0; k < block_samples / 32; ++k)
        values[k] = output[(slot + 32 * k) & (block_samples - 1)];
    for (int batch = 0; batch < 2 * d.taus(); batch += 32) {
        int node = batch + lane();
        Curve curve{};
        if (node < 2 * d.taus()) {
            curve.weight = coefficients[node];
            if (nonzero(curve.weight)) {
                int voltage = int(node >= d.taus());
                int index = row + node - voltage * d.taus();
                I offset = ptr[index];
                curve.offset = offset + (state + voltage) * stride;
                curve.length = low<I>(end, ptr[index + 1] - offset);
                if (curve.length <= begin) curve.weight = 0.;
            }
        }
        add_curve_batch(values, first, end, bank, curve);
    }
    #pragma unroll
    for (int k = 0; k < block_samples / 32; ++k){
        int at=(slot+32*k)&(block_samples-1);
        cohort[2*block_samples+at]+=values[k];
        cohort[3*block_samples+at]+=eps*values[k];
    }
}

// Workspace for the linear response renderer, reused for each Ma row.
// Cohort and running sums never leave shared memory before their final write.
__host__ __device__ I panel_bytes(I taus){
    I prefix=block_samples*(sizeof(Value)+sizeof(double))+2*taus*sizeof(double)+taus*sizeof(int);
    return ((prefix+15)&~I(15))+(taus<32?taus:32)*sizeof(Span);
}

__device__ __forceinline__ void accumulate_query_panel(const Table& d,const History* histories,
        I step,I birth,int loc,I count,bool post,int va,double vr,double* linear,double* cohort,int own_first,int own_last,int own_equal){
    int begin=step-birth,end=low<I>(d.horizon(),begin+block_samples-step%block_samples);
    const I* support=d.ix(post?PS_SUPPORT:RS_SUPPORT);
    if(begin>support[loc]||begin>=end)return;
    double* pair=linear;
    double* coefficients=pair+block_samples;
    int* boundaries=reinterpret_cast<int*>(coefficients+2*d.taus());
    auto* spans=reinterpret_cast<Span*>((reinterpret_cast<uintptr_t>(boundaries+d.taus())+15)&~uintptr_t(15));
    {
        for(int partner=d.ix(PAIR_SOURCE_PTR)[loc];partner<d.ix(PAIR_SOURCE_PTR)[loc+1];++partner){
            int old=d.ix(PAIR_TARGETS)[partner],row=d.ix(PAIR_REVERSE)[partner];
            const History& h=histories[old];
            int first=__shfl_sync(warp_mask,own_first,old);
            int last=__shfl_sync(warp_mask,own_last,old)-int(old>=loc)*__shfl_sync(warp_mask,own_equal,old);
            if(first==last&&old!=loc)continue;
            for(int k=lane();k<block_samples;k+=32)pair[k]=0.;
            for(int k=lane();k<2*d.taus();k+=32)coefficients[k]=0.;
            for(int k=lane();k<d.taus();k+=32)boundaries[k]=last;
            __syncwarp();
            ordinary_panel(d,h,cohort,pair,birth,loc,old,row,count,post,va,vr,
                           begin,end,first,last,boundaries,coefficients,spans);
            double eps=high(__ldg(d.fp(MA_REVERSAL)+old),__ldg(d.fp(MA_REVERSAL)+loc))-d.scalars[0];
            render_ordinary(d,pair,birth,row,post,va,begin,end,coefficients,cohort,eps);
            __syncwarp();
        }
    }
}
__device__ void render_rebase(RowWriter& writer, const I* ptr, int size) {
    int first = writer.begin + ((lane() - int(writer.birth + writer.begin)) & 31);
    int slot = (writer.birth + first) & (block_samples - 1);
    double values[block_samples / 32];
    #pragma unroll
    for (int k = 0; k < block_samples / 32; ++k)
        values[k] = writer.output[(slot + 32 * k) & (block_samples - 1)];
    for (int batch = 0; batch < size; batch += 32) {
        int node = batch + lane();
        double weight = node < size ? writer.nodes[node] : 0.;
        I offset = 0, length = 0;
        if (nonzero(weight)) {
            offset = ptr[node];
            length = ptr[node + 1] - offset;
            writer.nodes[node] = 0.;
        }
        Curve own{offset,int(low<I>(writer.end,length)),weight};
        add_curve_batch(values,first,writer.end,writer.bank,own);
    }
    #pragma unroll
    for (int k = 0; k < block_samples / 32; ++k)
        writer.output[(slot + 32 * k) & (block_samples - 1)] = values[k];
}
__device__ void prepare_rebase_panel(const Table& d, History* histories, I birth, int row,
                                   int begin, int end, RowWriter& writer, int partition, int partitions,double& initial) {
    int old = d.ix(PAIR_SOURCES)[row], next = d.ix(PAIR_TARGETS)[row];
    const History& oh = histories[old];
    const History& nh = histories[next];
    int os = d.ix(BS_SUPPORT)[old], ns = d.ix(BS_SUPPORT)[next];
    int of = oh.lower(birth - low<I>(os, d.ix(HISTORY_SUPPORT)[old]) + 1);
    int oe = oh.lower(birth + 1);
    int nf = nh.lower(birth - low<I>(ns, d.ix(HISTORY_SUPPORT)[next]) + 1);
    int ne = nh.lower(birth + 1);
    of=oh.lower(birth-d.ix(QUERY_SUPPORT)[old],of,oe);
    nf=nh.lower(birth-d.ix(QUERY_SUPPORT)[next],nf,ne);
    if (of == oe || nf == ne) return;
    // A newer event can expire first only when the older slot has the
    // longer support. Otherwise its entire anchor pass is empty.
    int anchors = oe - of + (os > ns ? ne - nf : 0);
    int first = anchors * partition / partitions;
    int last = anchors * (partition + 1) / partitions;
    const I* ptr = d.ix(BP_PTR) + row * d.taus() * d.releases();
    for (int batch = first; batch < last; batch += 32) {
        int anchor = batch + lane();
        bool visit_old = anchor < oe - of;
        int index = visit_old ? of + anchor : nf + anchor - (oe - of);
        const History& h = visit_old ? oh : nh;
        const History& ph = visit_old ? nh : oh;
        int age = 0, expiry = -1, pfirst = 0, pend = 0;
        double scale = 0.;
        if (anchor < last) {
            age = birth - h.birth_at(index);
            scale = h.count(index);
            expiry = (visit_old ? os : ns) - age;
            if (expiry >= begin) {
                // Each pair belongs to the anchor with the earlier expiry;
                // old wins equal-expiry ties. Prefix bounds replace the merge.
                if (visit_old) {
                    pfirst = nh.rank(birth - age + os - ns, nf, ne);
                    pend = ne;
                } else {
                    pfirst = oh.rank(birth - age + ns - os + 1, of, oe);
                    pend = oh.rank(birth - age + 1, pfirst, oe);
                }
            }
        }
        while (__any_sync(warp_mask, pend > pfirst)) {
            Span span{};
            span.begin = begin;
            span.end = low(end, expiry + 1);
            int slots[3] = {};
            if (pend > pfirst) {
                int pa = birth - ph.birth_at(pend - 1);
                int oa = visit_old ? age : pa, na = visit_old ? pa : age;
                if (oa < na) {
                    pend = pfirst;
                } else {
                    int tau = oa - na;
                    int tl = d.ix(TAU_LO,old)[tau], th = d.ix(TAU_HI,old)[tau];
                    int rl = d.ix(RELEASE_LO,old)[oa], rh = d.ix(RELEASE_HI,old)[oa];
                    int last_age = visit_old
                        ? (tl == th ? pa : oa - int(d.ix(TAU,old)[tl]) - 1)
                        : low(tl == th ? oa : na + int(d.ix(TAU,old)[th]) - 1,
                              rl == rh ? oa : int(d.ix(RELEASE,old)[rh]) - 1);
                    int group = ph.rank(birth - last_age, pfirst, pend);
                    double cw = 0., tw = 0., rw = 0.;
                    if (pend - group == 1) {
                        if (oa != na || old <= next) {
                            double pc = ph.count(group);
                            cw = oa == na && old == next ? scale * (scale - 1.) * .5 : scale * pc;
                            tw = cw * d.fp(TAU_RATIO,old)[tau];
                            rw = cw * d.fp(RELEASE_RATIO,old)[oa];
                        }
                    } else {
                        double cn = 0., ages = 0.; ma_first_moment(ph,group,pend,birth,cn,ages);
                        cw = scale * cn;
                        if (visit_old) {
                            tw = scale * ((oa - d.ix(TAU,old)[tl]) * cn - ages) * d.fp(TAU_INVERSE,old)[tl];
                            rw = cw * d.fp(RELEASE_RATIO,old)[oa];
                        } else {
                            tw = scale * (ages - (na + d.ix(TAU,old)[tl]) * cn) * d.fp(TAU_INVERSE,old)[tl];
                            rw = scale * (ages - d.ix(RELEASE,old)[rl] * cn) * d.fp(RELEASE_INVERSE,old)[rl];
                        }
                    }
                    pend = group;
                    bool triangle = tl != th && rl != rh && d.ix(RELEASE,old)[rl] < d.ix(TAU,old)[th];
                    slots[0] = tl * d.releases() + rl;
                    slots[1] = tl * d.releases() + rh;
                    slots[2] = th * d.releases() + (triangle ? rh : rl);
                    span = {ptr[slots[0]], ptr[slots[1]], ptr[slots[2]],
                            ptr[slots[0] + 1] - ptr[slots[0]], ptr[slots[1] + 1] - ptr[slots[1]],
                            ptr[slots[2] + 1] - ptr[slots[2]],
                            triangle ? cw - rw : cw - tw - rw, triangle ? rw - tw : rw, tw,
                            begin, low(end, expiry + 1)};
                }
            }
            if(begin==0){
                double own=span.end>0?span.wa*ma_initial(d,true,row*d.taus()*d.releases()+slots[0])
                    +span.wb*ma_initial(d,true,row*d.taus()*d.releases()+slots[1])
                    +span.wc*ma_initial(d,true,row*d.taus()*d.releases()+slots[2]):0.;
                double sum=cg::reduce(writer.warp,own,cg::plus<double>());
                if(lane()==0)initial+=sum;
            }
            span.wa = writer.contract(slots[0], span.na, span.begin, span.end, span.wa);
            span.wb = writer.contract(slots[1], span.nb, span.begin, span.end, span.wb);
            span.wc = writer.contract(slots[2], span.nc, span.begin, span.end, span.wc);
            span.end = low<I>(span.end, high(nonzero(span.wa) ? span.na : 0,
                                  high(nonzero(span.wb) ? span.nb : 0, nonzero(span.wc) ? span.nc : 0)));
            writer.append(span, span.begin < span.end);
        }
    }
    Span boundary{};
    if (lane() == 0 && partition == 0 && begin == 0 && old < next && os < ns && os < d.horizon()
        && os < d.ix(HISTORY_SUPPORT)[old] && os < d.ix(HISTORY_SUPPORT)[next]) {
        int op = oh.rank(birth - os), np = nh.rank(birth - os);
        if (op < oh.size && np < nh.size && oh.birth_at(op) == birth - os && nh.birth_at(np) == birth - os) {
            double count = double(oh.count(op)) * double(nh.count(np));
            int lo = d.ix(RELEASE_LO,old)[os], hi = d.ix(RELEASE_HI,old)[os];
            double ratio = d.fp(RELEASE_RATIO,old)[os];
            initial+=count*((1.-ratio)*ma_initial(d,true,row*d.taus()*d.releases()+lo)
                +ratio*ma_initial(d,true,row*d.taus()*d.releases()+hi));
            boundary = {ptr[lo], ptr[hi], 0, ptr[lo + 1] - ptr[lo], ptr[hi + 1] - ptr[hi], 0,
                        count * (1. - ratio), count * ratio, 0., 0, 1};
        }
    }
    writer.append(boundary, boundary.begin < boundary.end);
    render_rebase(writer, ptr, d.taus() * d.releases());
}
__device__ void rebase_single_panel(const Table& d, const History* histories, double* single,
                              I birth, int loc, int begin, int end,double& initial) {
    {
        const History& h = histories[loc];
        I support = d.ix(BS_SUPPORT)[loc];
        int first = h.lower(high<I>(high(birth - support + 1, birth + begin - support), birth-d.ix(QUERY_SUPPORT)[loc]));
        int last = h.lower(birth + 1);
        // The event-age coordinate may differ from the ordinary tau coordinate.
        while (last > first) {
            I age = birth - h.birth_at(last - 1);
            int lo = ma_event_interval(d,loc,age);
            int group = h.lower(birth - d.ix(EVENT,loc)[lo + 1] + 1, first, last);
            double n=0.,r=0.;double inverse_width=d.fp(EVENT_INVERSE,loc)[lo];
            if(lane()==0)ma_first_moment(h,group,last,birth-d.ix(EVENT,loc)[lo],n,r);
            n=broadcast(n);r=broadcast(r*inverse_width);
            const I* ptr = d.ix(BS_PTR);
            int row = loc * d.events() + lo;
            if(begin==0&&lane()==0)
                initial+=(n-r)*ma_initial(d,false,row)+r*ma_initial(d,false,row+1);
            I a = ptr[row], b = ptr[row + 1], na = b - a, nb = ptr[row + 2] - b;
            int stop = low<I>(end, low<I>(high(na, nb), support - birth + h.birth_at(last - 1) + 1));
            int bulk = low<I>(stop, support - birth + h.birth_at(group) + 1);
            expand(single, birth, begin, bulk, d.bs, a, na, n - r, b, nb, r);
            int k = high(begin, bulk);
            if (k < stop) {
                int cut = h.lower(birth + k - support, group, last);
                while (k < stop) {
                    double cn=0.,cr=0.;
                    if(lane()==0)ma_first_moment(h,cut,last,birth-d.ix(EVENT,loc)[lo],cn,cr);
                    cn=broadcast(cn);cr=broadcast(cr*inverse_width);
                    int next = low<I>(stop, support - birth + h.birth_at(cut) + 1);
                    expand(single, birth, k, next, d.bs, a, na, cn - cr, b, nb, cr);
                    k = next;
                    ++cut;
                }
            }
            last = group;
        }
    }

}
__host__ __device__ I rebase_bytes(I nodes){
    I capacity=nodes>2*block_samples?nodes:2*block_samples;
    I bytes=(capacity+3*block_samples)*sizeof(double);
    return ((bytes+15)&~I(15))+32*sizeof(Span);
}
__global__ void rebase_parallel_kernel(const REDUCTION_GRID_CONSTANT Table d,Arena arena,
        Neuron* states,History* history,Jobs q,const I* clock){
    extern __shared__ double rebase_workspace[];
    I size=I(d.taus())*d.releases(),capacity=high<I>(size,2*block_samples);
    unsigned char* base=reinterpret_cast<unsigned char*>(rebase_workspace)+(threadIdx.x/32)*rebase_bytes(size);
    double* nodes=reinterpret_cast<double*>(base);
    double* output=nodes+capacity;double* sumG=output+block_samples;double* sumI=sumG+block_samples;
    auto* spans=reinterpret_cast<Span*>((reinterpret_cast<uintptr_t>(sumI+block_samples)+15)&~uintptr_t(15));
    double* initials=reinterpret_cast<double*>(reinterpret_cast<unsigned char*>(rebase_workspace)+4*rebase_bytes(size));
    __shared__ I selected_work;
    __shared__ int row_cursor;
    for(int k=lane();k<capacity;k+=32)nodes[k]=0.;
    __syncwarp();
    for(;;){
        if(threadIdx.x==0){selected_work=atomicAdd(reinterpret_cast<unsigned long long*>(q.data+REBASE_CURSOR),1ull);row_cursor=0;}
        __syncthreads();
        I work=selected_work;if(work>=q[REBASE_COUNT])return;
        int id=q.rebase()[work];Branch& state=states[id/2].branch[id%2];
        I step=*clock,birth=state.rebase_at,begin=step-birth;
        int end=low<I>(d.horizon(),begin+block_samples-step%block_samples);
        if(begin>=end){__syncthreads();continue;}
        for(int k=lane();k<block_samples;k+=32){sumG[k]=0.;sumI[k]=0.;}
        double initial=0.;

        double* singleG=nodes;double* singleI=nodes+block_samples;
        bool singles_started=false;
        int rows=d.pairs(),partitions=high(1,(4+high(1,rows)-1)/high(1,rows));
        int pair_tasks=rows*partitions,total_tasks=pair_tasks+d.locations();
        for(;;){
            int task=0;if(lane()==0)task=atomicAdd(&row_cursor,1);
            task=broadcast(task);if(task>=total_tasks)break;
            for(int k=lane();k<block_samples;k+=32)output[k]=0.;
            __syncwarp();
            if(task<pair_tasks){
                int row=task/partitions;
                RowWriter writer{output,birth,d.bp,spans,nodes,int(begin),end,cg::tiled_partition<32>(cg::this_thread_block())};
                prepare_rebase_panel(d,history+(id/2)*d.locations(),birth,row,begin,end,writer,task%partitions,partitions,initial);
                __syncwarp();
                int old=d.ix(PAIR_SOURCES)[row],next=d.ix(PAIR_TARGETS)[row];
                double eps=high(d.fp(MA_REVERSAL)[old],d.fp(MA_REVERSAL)[next])-d.scalars[0];
                for(int k=lane();k<block_samples;k+=32){sumG[k]+=output[k];sumI[k]+=eps*output[k];}
            }else{
                // The cursor is monotone: this warp will never receive another
                // pair task after its first single, so node storage can be reused.
                if(!singles_started){
                    for(int k=lane();k<block_samples;k+=32){singleG[k]=0.;singleI[k]=0.;}
                    __syncwarp();singles_started=true;
                }
                int loc=task-pair_tasks;
                rebase_single_panel(d,history+(id/2)*d.locations(),output,birth,loc,begin,end,initial);
                __syncwarp();
                double eps=d.fp(MA_REVERSAL)[loc]-d.scalars[0];
                for(int k=lane();k<block_samples;k+=32){singleG[k]+=output[k];singleI[k]+=eps*output[k];}
            }
            __syncwarp();
        }
        if(!singles_started){
            for(int k=lane();k<block_samples;k+=32){singleG[k]=0.;singleI[k]=0.;}
            __syncwarp();
        }
        if(lane()==0)initials[threadIdx.x/32]=initial;
        __syncthreads();
        int offset=threadIdx.x;
        if(offset<block_samples-step%block_samples){
            I now=step+offset;int slot=now%block_samples;
            double G1=0.,I1=0.,G2=0.,I2=0.;
            #pragma unroll
            for(int w=0;w<4;++w){
                auto* values=reinterpret_cast<double*>(reinterpret_cast<unsigned char*>(rebase_workspace)+w*rebase_bytes(size));
                G1+=values[slot];I1+=values[block_samples+slot];
                G2+=values[capacity+block_samples+slot];I2+=values[capacity+2*block_samples+slot];
            }
            initial=0.;if(offset==0)for(int w=0;w<4;++w)initial+=initials[w];
            Future* future=reinterpret_cast<Future*>(arena.data+state.future);
            double2 recovered=conditional_recover(d,{G1,I1,G2,I2});
            if(nonzero(recovered.x))atomicAdd(&future[slot].x,recovered.x);
            if(nonzero(recovered.y))atomicAdd(&future[slot].y,recovered.y);
            if(now==birth)state.u+=initial;
        }
        __syncthreads();
        for(int k=lane();k<capacity;k+=32)nodes[k]=0.;
        __syncwarp();
    }
}
__global__ void initialize_kernel(Neuron* states, int count, double initial, double rest) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= count) return;
    Neuron& s = states[n]; s = {};
    for (int b = 0; b < 2; ++b) s.branch[b] = {-1, -1, -1, 0, -1, initial, initial-rest, false};
    s.pending_start = -1;
}

// Assemble each arrival cohort in shared memory under its retained query.
__global__ void cohort_kernel(const REDUCTION_GRID_CONSTANT Table d,Arena arena,
        Neuron* states,History* history,Jobs q,const I* clock){
    {
        extern __shared__ double cohort_workspace[];
        auto* cohort=reinterpret_cast<double*>(reinterpret_cast<unsigned char*>(cohort_workspace)
                     +(threadIdx.x/32)*panel_bytes(d.taus()));
        double* linear=cohort+4*block_samples;
        I step=*clock,end=step+block_samples-step%block_samples;
        for(;;){
            I work=0;if(lane()==0)work=atomicAdd(reinterpret_cast<unsigned long long*>(q.data+ORDINARY_CURSOR),1ull);
            work=broadcast(work);if(work>=q[ORDINARY_COUNT])return;
            int n=q.ordinary()[work];Neuron& neuron=states[n];
            History* histories=history+n*d.locations();
            for(int pass=0;pass<1+int(neuron.pending&&!neuron.confirmed);++pass){
                int branch=pass?1-neuron.main:neuron.main;Branch& state=neuron.branch[branch];
                if(state.future<0||!state.query_count)continue;
                int channel=branch%event_branches,entry=-1,first_entry=0,upper_cache=0;
                if(lane()<d.locations()){
                    const History& h=histories[lane()];upper_cache=h.size;
                    first_entry=step%block_samples==0?h.rank(state.ordinary_since):h.size-1;
                    entry=h.size-1;
                    if(step%block_samples&&(entry<0||h.birth_at(entry)!=step))entry=-1;
                }
                for(;;){
                    I own_birth=-1;
                    if(lane()<d.locations()){
                        const History& h=histories[lane()];
                        while(entry>=first_entry&&entry>=0&&isnan(h.ratio(entry,channel)))--entry;
                        if(entry>=first_entry&&entry>=0)own_birth=h.birth_at(entry);
                    }
                    I birth=own_birth;
                    for(int offset=16;offset;offset/=2)
                        birth=high(birth,static_cast<I>(__shfl_xor_sync(warp_mask,static_cast<long long>(birth),offset)));
                    if(birth<0)break;
                    bool belongs=own_birth==birth,post=birth<state.rest_since;
                    unsigned active=__ballot_sync(warp_mask,belongs);
                    int own_first=0,own_last=0,own_equal=0;
                    if(lane()<d.locations()){
                        const History& h=histories[lane()];
                        I support=d.ix(post?PS_SUPPORT:RS_SUPPORT)[lane()];
                        own_first=h.rank(high<I>(step-support,birth-support+1));
                        own_last=h.rank_before(birth+1,own_first,high(own_first,upper_cache));
                        own_equal=own_last>own_first&&h.birth_at(own_last-1)==birth;
                        upper_cache=own_last;
                    }
                    for(int k=lane();k<4*block_samples;k+=32)cohort[k]=0.;__syncwarp();
                    while(active){
                        int loc=__ffs(active)-1,at=__shfl_sync(warp_mask,entry,loc);
                        const History& h=histories[loc];
                        accumulate_query_panel(d,histories,step,birth,loc,h.count(at),post,
                            h.state(at,channel),h.ratio(at,channel),linear,cohort,
                            own_first,own_last,own_equal);
                        active&=active-1;
                    }
                    Future* future=reinterpret_cast<Future*>(arena.data+state.future);
                    #pragma unroll
                    for(int k=0;k<4;++k)if(step+lane()+32*k<end){
                        int slot=(step+lane()+32*k)%block_samples;
                        Value sums=make_double4(cohort[slot],cohort[block_samples+slot],cohort[2*block_samples+slot],cohort[3*block_samples+slot]);
                        double2 v=conditional_recover(d,sums);
                        if(nonzero(v.x))atomicAdd(&future[slot].x,v.x);
                        if(nonzero(v.y))atomicAdd(&future[slot].y,v.y);
                    }
                    if(belongs)--entry;
                    __syncwarp();
                }
            }
        }
        return;
    }
}

__global__ void history_kernel(const REDUCTION_GRID_CONSTANT Table d, Arena arena, Neuron* states, History* history,
                                const I* inbox, Jobs q, const I* clock, int count) {
    int n = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (n >= count) return;
    I step = *clock;
    Neuron& s = states[n];
    if ((!s.pending || s.confirmed) && lane() == 0) {
        Branch& inactive = s.branch[1 - s.main];
        if (inactive.future >= 0) {
            arena.release(inactive.future, block_samples * sizeof(Future));
            inactive.future = -1;
        }
    }
    History* histories = history + n * d.locations();
    const I* arrivals = inbox + n * d.locations();
    bool input = false;
    for (int i = lane(); i < d.locations(); i += 32) input |= arrivals[i] != 0;
    input = __any_sync(warp_mask, input);
    __syncwarp();
    bool work = input || step % block_samples == 0 || step == s.branch[s.main].rebase_at
        || (s.pending && !s.confirmed && step == s.branch[1 - s.main].rebase_at);
    if (work && lane() == 0) {
        assert(s.next_step == step && "DIF and Network clocks differ; reset before replay");
        bool history_present = input;
        for (int loc = 0; loc < d.locations(); ++loc) {
            histories[loc].base = arena.data;
            I support = low<I>(d.horizon() - 1, __ldg(d.ix(HISTORY_SUPPORT)+(loc)));
            histories[loc].prune(step - support, arena);
            history_present |= histories[loc].size != 0;
            if (arrivals[loc]) histories[loc].request_append(arena);
        }
        for (int pass = 0; pass < 1 + int(s.pending && !s.confirmed); ++pass) {
            Branch& state = s.branch[pass ? 1 - s.main : s.main];
            state.query_count=0;
            if(state.last_spike<0||step>state.rebase_at){
                if(step%block_samples)state.query_count=int(input);
                else for(int loc=0;loc<d.locations();++loc)
                    state.query_count+=histories[loc].size-histories[loc].rank(state.ordinary_since)+int(arrivals[loc]!=0);

            }
            if (state.future < 0 && (needs_rebase(d, state, step)
                || (history_present && (state.last_spike < 0 || step > state.rebase_at))))
                arena.request(block_samples * sizeof(Future));
        }
        I index = atomicAdd(reinterpret_cast<unsigned long long*>(q.data + ORDINARY_COUNT), 1ull);
        q.ordinary()[index] = n;
    }
}

__global__ void prepare_kernel(const REDUCTION_GRID_CONSTANT Table d, Arena arena, Neuron* states, History* history,
                                const I* inbox, I* jobs,
                                const I* clock, int count) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    Jobs q{jobs, count};
    if (index >= q[ORDINARY_COUNT]) return;
    int n = q.ordinary()[index];
    I step = *clock;
    Neuron& s = states[n];
    History* histories = history + n * d.locations();
    const I* arrivals = inbox + n * d.locations();
    bool input = false;
    for (int i = lane(); i < d.locations(); i += 32) input |= arrivals[i] != 0;
    input = __any_sync(warp_mask, input);
    if (lane() == 0)
        for (int loc = 0; loc < d.locations(); ++loc)
            if (arrivals[loc]) histories[loc].append(step, arrivals[loc], arena);
    __syncwarp();
    bool history_present = false;
    for (int loc = lane(); loc < d.locations(); loc += 32) history_present |= histories[loc].size != 0;
    history_present = __any_sync(warp_mask, history_present);
    for (int pass = 0; pass < 1 + int(s.pending && !s.confirmed); ++pass) {
        int branch = pass ? 1 - s.main : s.main;
        Branch& state = s.branch[branch];
        bool fresh = state.future < 0 && (needs_rebase(d, state, step)
            || (history_present && (state.last_spike < 0 || step > state.rebase_at)));
        if (fresh && lane() == 0) state.future = arena.allocate(block_samples * sizeof(Future));
        __syncwarp();
        if (state.future>=0 && (fresh || step%block_samples==0 || (state.last_spike >= 0 && step == state.rebase_at))) {
            Future* values = reinterpret_cast<Future*>(arena.data + state.future);
            for (int k = lane(); k < block_samples; k += 32) values[k] = {};
        }
        if (state.last_spike >= 0 && step == state.rebase_at && lane() == 0)
            { state.ordinary_since = step + 1; state.u=0.; }
        if (needs_rebase(d, state, step) && lane() == 0) {
            I index = atomicAdd(reinterpret_cast<unsigned long long*>(jobs + REBASE_COUNT), 1ull);
            q.rebase()[index] = 2 * n + branch;
        }
        if (input && (state.last_spike < 0 || step > state.rebase_at)) {
            int a = 0, b = 0;
            double ratio = 0.;
            near(d.fp(state.post ? POST_GRID : REST_GRID), d.indices[state.post ? POST_STATES : REST_STATES],
                 state.previous, a, b, ratio);
            if (lane() == 0) {
                for (int loc = 0; loc < d.locations(); ++loc) {
                    if (!arrivals[loc]) continue;
                    History& h = histories[loc];
                    h.state(h.size - 1, branch % event_branches) = a;
                    h.ratio(h.size - 1, branch % event_branches) = ratio;
                }
            }
        }
    }
}

__global__ void finish_kernel(const REDUCTION_GRID_CONSTANT Table d, Arena arena, Neuron* states, History* history, I* inbox,
                               const I* clock, int count, double abort_voltage,
                               const I* columns, const I* record_meta, double* records, double* monitors,
                               double* out, bool* spikes, I* next_clock) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= count) return;
    I step = *clock;
    I record_start = record_meta[0];
    int record_count = record_meta[1];
    Neuron& s = states[n];
    Future* f[2];
    for (int branch = 0; branch < 2; ++branch)
        f[branch] = s.branch[branch].future < 0 ? nullptr
            : reinterpret_cast<Future*>(arena.data + s.branch[branch].future);
    I* arrivals = inbox + n * d.locations();
    Branch& before = s.branch[s.main];
    const History* histories = history + n * d.locations();
    double main_voltage = voltage(d, before, f[s.main], step);
    bool monitoring = s.pending && !s.confirmed;
    double monitor_voltage = monitoring ? voltage(d, s.branch[1 - s.main], f[1 - s.main], step)
        : s.pending ? s.branch[1 - s.main].previous : main_voltage;
    bool refractory = before.last_spike >= 0 && step - before.last_spike < d.metadata[REFRACTORY_AGE];
    bool candidate = !s.pending && !refractory && before.previous < d.scalars[1] && main_voltage >= d.scalars[1];
    bool active = s.pending || candidate;
    if (candidate) main_voltage = d.scalars[0] + d.eta[0];
    // A confirmed monitor stops querying curves while eta advances to emission.
    bool confirmed = s.confirmed || (active && monitor_voltage >= d.scalars[2]);
    bool zero = s.pending && before.previous < 0. && main_voltage >= 0.;
    I start = candidate ? step : s.pending_start;
    bool aborted = active && !confirmed && monitor_voltage <= abort_voltage;
    bool committed = zero && !aborted;
    int column = columns[n];
    if (column >= 0 && active && !confirmed)
        monitors[(step - record_start) * record_count + column] = monitor_voltage;
    if (candidate) {
        int next = 1 - s.main;
        s.branch[next] = {step, step + d.metadata[REBASE_AGE], LLONG_MAX, step + 1, -1,
                          s.branch[s.main].previous, 0., true};
        s.main = next;
    }
    if (aborted) s.main = 1 - s.main;
    double selected = aborted ? monitor_voltage : main_voltage;
    if (column >= 0) {
        records[(step - record_start) * record_count + column] = selected;
        if (aborted)
            for (I t = high(start, record_start); t <= step; ++t)
                records[(t - record_start) * record_count + column] = monitors[(t - record_start) * record_count + column];
    }
    Branch& main = s.branch[s.main];
    if (main.post && !candidate && step - main.last_spike > d.metadata[TROUGH_AGE] && selected > main.previous) {
        main.post = false; main.rest_since = step + 1;
    }
    main.previous = selected;
    if (monitoring || candidate) {
        Branch& monitor = s.branch[1 - s.main];
        if (monitor.post && step - monitor.last_spike > d.metadata[TROUGH_AGE] && monitor_voltage > monitor.previous) {
            monitor.post = false; monitor.rest_since = step + 1;
        }
        monitor.previous = monitor_voltage;
    }
    s.pending = active && !aborted && !committed; s.pending_start = s.pending ? start : -1;
    s.confirmed = s.pending && confirmed;
    s.next_step = step + 1;
    int k = step % block_samples;
    for (int branch = 0; branch < 2; ++branch)
        if (f[branch] && (branch == s.main || monitoring || candidate)) {
            f[branch][k] = {};
        }
    out[n] = selected; spikes[n] = committed;
    if (n == 0) *next_clock = step + 1;
    for (int i = 0; i < d.locations(); ++i) if (arrivals[i]) arrivals[i] = 0;
}

__device__ void schedule_arrivals(const I* steps, const I* ptr, const void* groups, const I* targets,
                                   const I* target_ptr, I* inbox, const I* clock, int buckets,
                                   bool narrow_groups, int schedule_blocks) {
    __shared__ I first, last;
    I step = *clock;
    if (threadIdx.x == 0) {
        int lo = 0, hi = buckets;
        while (lo < hi) { int mid = (lo + hi) / 2; if (steps[mid] < step) lo = mid + 1; else hi = mid; }
        bool found = lo < buckets && steps[lo] == step;
        first = found ? ptr[lo] : 0; last = found ? ptr[lo + 1] : 0;
    }
    __syncthreads();
    for (I event = first + I(blockIdx.x) * blockDim.x + threadIdx.x; event < last; event += I(schedule_blocks) * blockDim.x) {
        I group = narrow_groups ? static_cast<const uint32_t*>(groups)[event]
                                : static_cast<const uint64_t*>(groups)[event];
        for (I p = target_ptr[group]; p < target_ptr[group + 1]; ++p)
            atomicAdd(reinterpret_cast<unsigned long long*>(inbox + targets[p]), 1ull);
    }
}
__global__ void deliver_kernel(const I* ptr, const I* delay, const I* counts,
                                I* source_counts, uint32_t* pending, const I* clock,
                                int sources, int groups, int delays) {
    int source = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (source >= sources) return;
    I count = counts[source], step = *clock - 1;
    if (count > 0 && ptr[source] < ptr[source + 1]) {
        if (lane() == 0) source_counts[(step % delays) * sources + source] = count;
        I words = (groups + 31) / 32;
        for (I group = ptr[source] + lane(); group < ptr[source + 1]; group += 32) {
            I due = step + delay[group];
            atomicOr(pending + (due % delays) * words + group / 32, uint32_t(1) << (group % 32));
        }
    }
}
__device__ void pending_arrivals(const I* ptr, const I* targets, const I* delay, const I* source,
                                  const I* source_counts, uint32_t* pending, I* inbox,
                                  const I* clock, int sources, int groups, int delays, I word) {
    int words = (groups + 31) / 32;
    if (word >= words) return;
    I step = *clock;
    uint32_t bits = 0;
    if (lane() == 0) {
        bits = pending[(step % delays) * words + word];
        pending[(step % delays) * words + word] = 0;
    }
    bits = broadcast(bits);
    while (bits) {
        int group = word * 32 + __ffs(bits) - 1;
        bits &= bits - 1;
        I birth = step - delay[group];
        I count = source_counts[(birth % delays) * sources + source[group]];
        for (I p = ptr[group] + lane(); p < ptr[group + 1]; p += 32)
            atomicAdd(reinterpret_cast<unsigned long long*>(inbox + targets[p]),
                      static_cast<unsigned long long>(count));
    }
}
// These operations touch disjoint metadata; only inbox writes overlap,
// and those retain the original integer atomicAdd. The next same-stream
// history launch is the global completion barrier for arrivals and resets.
__global__ void arrival_reset_kernel(
        const I* steps, const I* schedule_ptr, const void* schedule_groups, const I* targets,
        const I* target_ptr, const I* route_ptr, const I* route_targets,
        const I* delay, const I* source, const I* source_counts, const I* counts,
        uint32_t* pending, I* inbox, const I* clock, Arena arena, Jobs q,
        int schedule_buckets, bool narrow_groups, int schedule_blocks,
        int sources, int groups, int delays, int width) {
    I index = I(blockIdx.x) * blockDim.x + threadIdx.x;
    I stride = I(gridDim.x) * blockDim.x;
    for (I k = index; k < width; k += stride)
        if (counts[k]) atomicAdd(reinterpret_cast<unsigned long long*>(inbox + k),
                                 static_cast<unsigned long long>(counts[k]));
    for (I k = index; k < JOB_HEADER; k += stride) q.data[k] = 0;
    for (I k = index; k < Arena::bins; k += stride)
        arena.control[1 + 3 * Arena::bins + k] = 0;
    if (blockIdx.x < schedule_blocks) {
        // Uniform within the block, including the helper's __syncthreads.
        schedule_arrivals(steps, schedule_ptr, schedule_groups, targets, target_ptr,
                          inbox, clock, schedule_buckets, narrow_groups, schedule_blocks);
    } else if (groups) {
        I word = ((I(blockIdx.x) - schedule_blocks) * blockDim.x + threadIdx.x) / 32;
        pending_arrivals(route_ptr, route_targets, delay, source, source_counts,
                         pending, inbox, clock, sources, groups, delays, word);
    }
}

__global__ void release_kernel(Neuron* states, int count, History* history, int size, Arena arena) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (lane() != 0) return;
    if (i < size) {
        history[i].release(arena);
    }
    if (i < count)
        for (int branch = 0; branch < 2; ++branch) {
            I offset = states[i].branch[branch].future;
            if (offset >= 0) arena.release(offset, block_samples * sizeof(Future));
            states[i].branch[branch].future = -1;
        }
}
__global__ void inspect_kernel(const Neuron* states, I* pending, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) pending[i] = states[i].pending_start;
}
} // namespace response

// Keep a stable virtual address while committing only requested physical pages.
// The FFI owns this resource explicitly; it is never hidden in a device global.
struct DeviceRegion {
    CUdeviceptr base = 0;
    size_t capacity = 0, mapped = 0, granularity = 0;
    CUmemAllocationProp properties{};
    struct Chunk { CUmemGenericAllocationHandle handle; size_t bytes; };
    std::vector<Chunk> chunks;

    static void check(CUresult status) {
        if (status != CUDA_SUCCESS) {
            const char* message;
            cuGetErrorString(status, &message);
            throw std::runtime_error(message);
        }
    }
    void initialize(size_t budget) {
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        BE_CUDA_CHECK(cudaGetDevice(&properties.location.id));
        check(cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        capacity = (budget + granularity - 1) / granularity * granularity;
        check(cuMemAddressReserve(&base, capacity, 0, 0, 0));
    }
    void grow(size_t requested) {
        if (requested <= mapped) return;
        size_t total = (requested + granularity - 1) / granularity * granularity;
        if (total > capacity) throw std::runtime_error("DIF arena budget exhausted; increase gpu_heap_bytes");
        size_t bytes = total - mapped;
        CUmemGenericAllocationHandle handle;
        check(cuMemCreate(&handle, bytes, &properties, 0));
        CUresult status = cuMemMap(base + mapped, bytes, 0, handle, 0);
        if (status != CUDA_SUCCESS) { cuMemRelease(handle); check(status); }
        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        status = cuMemSetAccess(base + mapped, bytes, &access, 1);
        if (status != CUDA_SUCCESS) {
            cuMemUnmap(base + mapped, bytes);
            cuMemRelease(handle);
            check(status);
        }
        chunks.push_back({handle, bytes});
        mapped = total;
    }
    ~DeviceRegion() {
        size_t offset = 0;
        for (const auto& chunk : chunks) {
            cuMemUnmap(base + offset, chunk.bytes);
            cuMemRelease(chunk.handle);
            offset += chunk.bytes;
        }
        if (base) cuMemAddressFree(base, capacity);
    }
};

struct ArenaStorage {
    DeviceRegion history;
    response::I* counters = nullptr;

    void initialize(size_t budget) {
        history.initialize(budget);
        BE_CUDA_CHECK(cudaMallocHost(&counters, response::Arena::words * sizeof(response::I)));
    }
    void prepare(const response::I* control, cudaStream_t stream) {
        // history_kernel has released expired blocks and counted this step's
        // allocations. Reserve only the requests that cannot reuse a free block.
        BE_CUDA_CHECK(cudaMemcpyAsync(counters, control, response::Arena::words * sizeof(response::I),
                                      cudaMemcpyDeviceToHost, stream));
        BE_CUDA_CHECK(cudaStreamSynchronize(stream));
        size_t requested = counters[0];
        for (int k = 0; k < response::Arena::bins; ++k) {
            response::I extra = counters[1 + 3 * response::Arena::bins + k]
                              - counters[1 + 2 * response::Arena::bins + k];
            if (extra <= 0) continue;
            size_t unit = size_t(1) << (k + response::Arena::minimum_shift);
            if (size_t(extra) > history.capacity / unit || requested > history.capacity - size_t(extra) * unit)
                throw std::runtime_error("DIF event arena exceeds gpu_heap_bytes");
            requested += size_t(extra) * unit;
        }
        history.grow(requested);
    }
    ~ArenaStorage() { if (counters) cudaFreeHost(counters); }
};

// Runtime-owned auxiliary stream. The mutex protects host enqueue sequences
// and event reuse; each runtime has an independent context, even on one device.
struct StageStreams {
    response::I metadata[response::Table::metadata_size]{};
    double scalars[7]{};
    int device = -1, multiprocessors = 0;
    cudaStream_t auxiliary = nullptr, last_main = nullptr, current_main = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr, finished = nullptr;
    bool finish_recorded = false, main_seen = false;
    std::mutex mutex;

    void initialize(int target) {
        int previous;
        BE_CUDA_CHECK(cudaGetDevice(&previous));
        BE_CUDA_CHECK(cudaSetDevice(target));
        device = target;
        try {
            BE_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, target));
            BE_CUDA_CHECK(cudaStreamCreateWithFlags(&auxiliary, cudaStreamNonBlocking));
            BE_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            BE_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            BE_CUDA_CHECK(cudaEventCreateWithFlags(&finished, cudaEventDisableTiming));
        } catch (...) {
            cudaSetDevice(previous);
            throw;
        }
        BE_CUDA_CHECK(cudaSetDevice(previous));
    }

    // Called with mutex held and this runtime's CUDA device current.
    // Raw status-returning calls keep the original enqueue exception intact.
    void synchronize_pending() noexcept {
        if (main_seen) cudaStreamSynchronize(current_main);
        if (finish_recorded && finished) cudaEventSynchronize(finished);
        if (auxiliary) cudaStreamSynchronize(auxiliary);
    }

    ~StageStreams() {
        std::lock_guard<std::mutex> lock(mutex);
        if (device < 0) return;
        int previous;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
        // The main-stream finish event follows its wait for auxiliary work.
        // Auxiliary synchronization also covers a partially enqueued failure.
        synchronize_pending();
        if (ready) cudaEventDestroy(ready);
        if (done) cudaEventDestroy(done);
        if (finished) cudaEventDestroy(finished);
        if (auxiliary) cudaStreamDestroy(auxiliary);
        cudaSetDevice(previous);
    }
};

// Every failed enqueue leaves both streams drained before the FFI wrapper can
// return an error and permit buffer release/reset. Successful calls retain the
// asynchronous ready/done/finished event chain without a host synchronization.
struct StageEnqueueGuard {
    StageStreams& stages;
    bool committed = false;
    ~StageEnqueueGuard() noexcept {
        if (!committed) stages.synchronize_pending();
    }
};

extern "C" uint64_t reduction_stage_create(int device) {
    auto* stages = new StageStreams;
    try { stages->initialize(device); }
    catch (...) { delete stages; return 0; }
    return reinterpret_cast<uint64_t>(stages);
}
extern "C" int reduction_stage_metadata(uint64_t handle,const int64_t* header,int size,const double* scalars){
    if(size!=response::Table::metadata_size)return 1;
    auto& stages=*reinterpret_cast<StageStreams*>(handle);
    std::memcpy(stages.metadata,header,sizeof(stages.metadata));
    std::memcpy(stages.scalars,scalars,sizeof(stages.scalars));
    return 0;
}
extern "C" void reduction_stage_free(uint64_t handle) { delete reinterpret_cast<StageStreams*>(handle); }

extern "C" uint64_t reduction_arena_create(size_t capacity) {
    auto* memory = new ArenaStorage;
    try { memory->initialize(capacity); }
    catch (...) { delete memory; return 0; }
    return reinterpret_cast<uint64_t>(memory);
}
extern "C" void reduction_arena_free(uint64_t handle) { delete reinterpret_cast<ArenaStorage*>(handle); }
extern "C" size_t reduction_neuron_size() { return sizeof(response::Neuron); }
extern "C" size_t reduction_history_size() { return sizeof(response::History); }
extern "C" size_t reduction_arena_control_size() { return response::Arena::words; }
extern "C" size_t reduction_jobs_size(int64_t count) { return response::JOB_HEADER + 3 * count; }
extern "C" int64_t reduction_ma_rebase_configure(int64_t nodes){
    int device,limit;BE_CUDA_CHECK(cudaGetDevice(&device));
    BE_CUDA_CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    if(nodes>limit/(4*int64_t(sizeof(double))))return 0;
    int64_t bytes=4*response::rebase_bytes(nodes)+4*sizeof(double);
    if(bytes>limit)return 0;
    BE_CUDA_CHECK(cudaFuncSetAttribute(response::rebase_parallel_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    return bytes;
}

extern "C" int64_t reduction_panel_configure(int64_t taus){
    int device,limit;BE_CUDA_CHECK(cudaGetDevice(&device));
    BE_CUDA_CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    if(taus>limit/(8*int64_t(sizeof(double))))return 0;
    int64_t bytes=4*response::panel_bytes(taus);
    if(bytes>limit)return 0;
    BE_CUDA_CHECK(cudaFuncSetAttribute(response::cohort_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    return bytes;
}

static const void* active_kernel(bool rebase){
    return rebase?reinterpret_cast<const void*>(response::rebase_parallel_kernel)
                 :reinterpret_cast<const void*>(response::cohort_kernel);
}
extern "C" int reduction_configure(bool rebase,int64_t bytes){
    const void* kernel=active_kernel(rebase);
    cudaFuncAttributes attributes;BE_CUDA_CHECK(cudaFuncGetAttributes(&attributes,kernel));
    // Rebase's four-warp reduction is structural. Ordinary warps are independent.
    int chosen=128,best=-1;
    for(int threads=rebase?128:32;threads<=(rebase?128:std::min(REDUCTION_BLOCK_THREADS,attributes.maxThreadsPerBlock));threads+=32){
        int dynamic=rebase?int(bytes):int(bytes/4)*(threads/32),resident;
        if(dynamic>attributes.maxDynamicSharedSizeBytes)continue;
        BE_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,threads,dynamic));
        if(resident*threads>=best){chosen=threads;best=resident*threads;}
    }
    int dynamic=rebase?int(bytes):int(bytes/4)*(chosen/32),resident,device,major;
    BE_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,chosen,dynamic));
    if(!resident)throw std::runtime_error("DIF active kernel has no resident launch configuration");
    BE_CUDA_CHECK(cudaGetDevice(&device));
    BE_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
    if(major>=7){
        int shared,reserved=0;
        BE_CUDA_CHECK(cudaDeviceGetAttribute(&shared,cudaDevAttrMaxSharedMemoryPerMultiprocessor,device));
#if CUDART_VERSION >= 11030
        BE_CUDA_CHECK(cudaDeviceGetAttribute(&reserved,cudaDevAttrReservedSharedMemoryPerBlock,device));
#endif
        int64_t need=int64_t(resident)*(dynamic+attributes.sharedSizeBytes+reserved);
        int percent=std::min<int64_t>(100,(100*need+shared-1)/shared);
        BE_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,percent));
    }
    return chosen;
}
extern "C" int reduction_resident_blocks(bool rebase,int threads,int64_t bytes){
    int device,count,blocks;BE_CUDA_CHECK(cudaGetDevice(&device));
    BE_CUDA_CHECK(cudaDeviceGetAttribute(&count,cudaDevAttrMultiProcessorCount,device));
    BE_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,active_kernel(rebase),threads,int(bytes)));
    return count*blocks;
}

template<class T> T* data(BE::Tensor x) { return static_cast<T*>(x.data_ptr()); }

// @BE reduction_initialize
void reduction_initialize(BE::Tensor output, int64_t count, double initial, double rest, int64_t stream) {
    BE_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.numel(), (cudaStream_t)stream));
    response::initialize_kernel<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(data<response::Neuron>(output), count, initial, rest);
    BE_CUDA_CHECK(cudaGetLastError());
}

// @BE reduction_advance
void reduction_advance(const BE::Tensor curves,
                 const BE::Tensor indices, const BE::Tensor real,
                 const BE::Tensor control_in, const BE::Tensor clock,
                 const BE::Tensor core_in,
                 const BE::Tensor counts,
                 const BE::Tensor steps, const BE::Tensor schedule_ptr, const BE::Tensor schedule_groups,
                 const BE::Tensor targets, const BE::Tensor target_ptr,
                 const BE::Tensor columns,
                 const BE::Tensor records_in, const BE::Tensor monitors_in,
                 const BE::Tensor record_meta,
                 const BE::Tensor delivery_in,
                 const BE::Tensor source_ptr, const BE::Tensor group_ptr, const BE::Tensor route_targets,
                 const BE::Tensor route_delay, const BE::Tensor route_source,
                 BE::Tensor core,
                 BE::Tensor records, BE::Tensor monitors,
                 BE::Tensor delivery,
                 BE::Tensor control,
                 BE::Tensor voltage, BE::Tensor spikes, BE::Tensor next_clock,
                 int64_t count, double abort_voltage,
                 int64_t delays, int64_t pending_offset, int64_t width, int64_t schedule_blocks,
                 int64_t threads, int64_t rebase_threads, int64_t ordinary_threads,
                 int64_t blocks, int64_t ordinary_blocks,
                 int64_t arena_handle, int64_t stage_handle,
                 int64_t rp_offset, int64_t ps_offset, int64_t pp_offset,
                 int64_t bs_offset, int64_t bp_offset, int64_t eta_offset,
                 int64_t history_offset, int64_t inbox_offset,
                 int64_t jobs_offset, int64_t ma_shared_bytes, int64_t panel_shared_bytes, int64_t stream) {
    auto& stages = *reinterpret_cast<StageStreams*>(stage_handle);
    std::lock_guard<std::mutex> stage_lock(stages.mutex);
    auto main_stream = reinterpret_cast<cudaStream_t>(stream);
    StageEnqueueGuard enqueue_guard{stages};
    stages.current_main = main_stream;
    stages.main_seen = true;
    if (stages.finish_recorded && stages.last_main != main_stream)
        BE_CUDA_CHECK(cudaStreamWaitEvent(main_stream, stages.finished, 0));
    const double* bank = data<double>(curves);
    response::Table table{data<response::I>(indices), data<double>(real),
        bank, bank + rp_offset,
        bank + ps_offset, bank + pp_offset,
        bank + bs_offset, bank + bp_offset,
        bank + eta_offset};
    std::memcpy(table.metadata,stages.metadata,sizeof(table.metadata));
    std::memcpy(table.scalars,stages.scalars,sizeof(table.scalars));
    auto* base = data<unsigned char>(core);
    auto* neurons = reinterpret_cast<response::Neuron*>(base);
    auto* history = reinterpret_cast<response::History*>(base + history_offset);
    auto* inbox = reinterpret_cast<response::I*>(base + inbox_offset);
    auto* jobs = reinterpret_cast<response::I*>(base + jobs_offset);
    auto* delivery_base = delivery.numel() ? data<unsigned char>(delivery) : base;
    auto* source_counts = reinterpret_cast<response::I*>(delivery_base);
    auto* pending = reinterpret_cast<uint32_t*>(delivery_base + pending_offset);
    auto& storage = *reinterpret_cast<ArenaStorage*>(arena_handle);
    response::Arena memory{reinterpret_cast<unsigned char*>(storage.history.base), data<response::I>(control), response::I(storage.history.mapped)};
    response::Jobs q{jobs, int(count)};
    constexpr int reset_threads = 256;
    int64_t reset_items = std::max<int64_t>(width, std::max<int64_t>(response::JOB_HEADER, response::Arena::bins));
    int64_t pending_blocks = (((route_delay.numel() + 31) / 32) * 32 + reset_threads - 1) / reset_threads;
    int64_t reset_blocks = std::max<int64_t>((reset_items + reset_threads - 1) / reset_threads,
                                           schedule_blocks + pending_blocks);
    response::arrival_reset_kernel<<<reset_blocks, reset_threads, 0, main_stream>>>(
        data<response::I>(steps), data<response::I>(schedule_ptr), schedule_groups.data_ptr(),
        data<response::I>(targets), data<response::I>(target_ptr),
        data<response::I>(group_ptr), data<response::I>(route_targets), data<response::I>(route_delay),
        data<response::I>(route_source), source_counts, data<response::I>(counts), pending, inbox,
        data<response::I>(clock), memory, q, steps.numel(), target_ptr.numel() - 1 <= UINT32_MAX,
        schedule_blocks, source_ptr.numel() - 1, route_delay.numel(), delays, width);
    int neuron_blocks = (count * 32 + threads - 1) / threads;
    response::history_kernel<<<neuron_blocks, threads, 0, main_stream>>>(
        table, memory, neurons, history, inbox,
        q, data<response::I>(clock), count);
    storage.prepare(data<response::I>(control), main_stream);
    memory.capacity = storage.history.mapped;
    response::prepare_kernel<<<neuron_blocks, threads, 0, main_stream>>>(
        table, memory,
        neurons, history, inbox, jobs,
        data<response::I>(clock), count);
    BE_CUDA_CHECK(cudaEventRecord(stages.ready, main_stream));
    BE_CUDA_CHECK(cudaStreamWaitEvent(stages.auxiliary, stages.ready, 0));
    response::rebase_parallel_kernel<<<blocks, rebase_threads, ma_shared_bytes, stages.auxiliary>>>(
        table, memory, neurons, history, q, data<response::I>(clock));
    BE_CUDA_CHECK(cudaEventRecord(stages.done, stages.auxiliary));
    response::cohort_kernel<<<ordinary_blocks, ordinary_threads, panel_shared_bytes, main_stream>>>(
        table, memory, neurons, history, q, data<response::I>(clock));
    BE_CUDA_CHECK(cudaStreamWaitEvent(main_stream, stages.done, 0));
    // Spread independent neuron readout over the device before filling blocks.
    int64_t finish_warps = std::max<int64_t>(1,
        (count + int64_t(32) * stages.multiprocessors - 1) / (int64_t(32) * stages.multiprocessors));
    int64_t finish_threads = std::min<int64_t>(threads, 32 * finish_warps);
    response::finish_kernel<<<(count + finish_threads - 1) / finish_threads, finish_threads, 0, main_stream>>>(
        table, memory, neurons, history, inbox,
        data<response::I>(clock), count, abort_voltage, data<response::I>(columns), data<response::I>(record_meta),
        data<double>(records), data<double>(monitors),
        data<double>(voltage), data<bool>(spikes), data<response::I>(next_clock));
    BE_CUDA_CHECK(cudaEventRecord(stages.finished, main_stream));
    stages.finish_recorded = true;
    stages.last_main = main_stream;
    BE_CUDA_CHECK(cudaGetLastError());
    enqueue_guard.committed = true;
}

// @BE reduction_release
void reduction_release(const BE::Tensor core_in, const BE::Tensor control_in,
                 BE::Tensor core, BE::Tensor control,
                 int64_t count, int64_t history_offset, int64_t history_count, int64_t arena_handle, int64_t stream) {
    auto& storage = *reinterpret_cast<ArenaStorage*>(arena_handle);
    response::release_kernel<<<(history_count * 32 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        data<response::Neuron>(core), count,
        reinterpret_cast<response::History*>(data<unsigned char>(core) + history_offset), history_count,
        {reinterpret_cast<unsigned char*>(storage.history.base), data<response::I>(control), response::I(storage.history.mapped)});
    BE_CUDA_CHECK(cudaGetLastError());
}

// @BE reduction_inspect
void reduction_inspect(const BE::Tensor neurons, BE::Tensor pending, int64_t stream) {
    int count = pending.numel();
    response::inspect_kernel<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(data<response::Neuron>(neurons), data<response::I>(pending), count);
    BE_CUDA_CHECK(cudaGetLastError());
}

// @BE reduction_deliver
void reduction_deliver(const BE::Tensor ptr, const BE::Tensor delay, const BE::Tensor counts,
                       const BE::Tensor clock, const BE::Tensor delivery_in,
                       BE::Tensor delivery, int64_t sources, int64_t delays, int64_t threads,
                       int64_t pending_offset, int64_t stage_handle, int64_t stream) {
    auto& stages = *reinterpret_cast<StageStreams*>(stage_handle);
    std::lock_guard<std::mutex> stage_lock(stages.mutex);
    auto current = reinterpret_cast<cudaStream_t>(stream);
    StageEnqueueGuard enqueue_guard{stages};
    stages.current_main = current;
    stages.main_seen = true;
    if (stages.finish_recorded && stages.last_main != current)
        BE_CUDA_CHECK(cudaStreamWaitEvent(current, stages.finished, 0));
    response::deliver_kernel<<<(sources * 32 + threads - 1) / threads, threads, 0, current>>>(
        data<response::I>(ptr), data<response::I>(delay), data<response::I>(counts),
        data<response::I>(delivery), reinterpret_cast<uint32_t*>(data<unsigned char>(delivery) + pending_offset),
        data<response::I>(clock), sources, delay.numel(), delays);
    BE_CUDA_CHECK(cudaEventRecord(stages.finished, current));
    stages.finish_recorded = true;
    stages.last_main = current;
    BE_CUDA_CHECK(cudaGetLastError());
    enqueue_guard.committed = true;
}
