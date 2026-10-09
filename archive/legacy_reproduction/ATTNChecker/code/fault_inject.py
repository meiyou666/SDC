'''
template <typename T>
// __global__ void bitflip(T *dA, int64_t row, int64_t col, int64_t lda, int64_t batch){
__global__ void bitflip(T *dA, int64_t idx){
	// int stride = row * col;
	// int idx = batch * stride + row + col * lda;
	
	// T value = NAN;
	// T value = (T)1e10;
	// T value = INFINITY;
	// *(dA + idx) = value;

	int64_t flipBit = 0;
	float orgValue = (float)*(dA + idx);
	if(fabs(orgValue) >= 2){
		flipBit = 29;
	}
	else{
		flipBit = 30;
	}
	uint32_t* intValue = reinterpret_cast<uint32_t*>(&orgValue);
    *intValue ^= (1u << flipBit);
	*(dA + idx) = (T) *reinterpret_cast<float*>(intValue);
	// printf("%.6f\n", (float)*(dA + idx));
}
'''
import numpy as np

def bitflip(arr: np.ndarray):
    arr = arr.copy()
    assert arr.dtype == np.float32

    idx = tuple(np.random.randint(0, d) for d in arr.shape)
    org_value = arr[idx]

    flip_bit = 29 if abs(org_value) >= 2.0 else 30

    flat = arr.reshape(-1)
    flat_idx = np.ravel_multi_index(idx, arr.shape)
    int_value = flat.view(np.uint32)[flat_idx]
    int_value ^= (1 << flip_bit)
    flat.view(np.uint32)[flat_idx] = int_value

    return flat.reshape(arr.shape)