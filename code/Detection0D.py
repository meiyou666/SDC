'''
checksum源码:

template<class T, int64_t NROW, int64_t NCOL, int64_t C>
__global__ void encode_col_v5(int64_t num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					 T *dA_colchk, int64_t ldda_colchk, int64_t strideA_colchk) {

	SharedMemory<T> smem;
 	T* dA_sm = smem.getPointer();
	
	// extern __shared__ T dA_sm [];

	const int batch_id = blockIdx.x;
	const int tid = threadIdx.x;
	const int y_load = tid / NROW;
	const int x_load = tid % NROW;
	const int y_compute = tid / NCOL;
	const int x_compute = tid % NCOL;
	dA = dA + batch_id * strideA;
	dA_colchk = dA_colchk + batch_id * strideA_colchk;

	for (int i = 0; i < NCOL; i += C) {
		dA_sm[x_load+(NROW+1)*(i+y_load)] = dA[x_load+(NROW)*(i+y_load)];
	}	
	__syncthreads();

	if (x_compute < NCOL && y_compute < 2) {
		T res = 0.0;
		T * dA_col = &dA_sm[x_compute * (NROW+1)];
		if (y_compute == 0) {
			for (int i = 0; i < NROW; i++) {
				res += dA_col[i];
			}
		}
		if (y_compute == 1) {
			for (int i = 0; i < NROW; i++) {
				res += (T)(i+1) * dA_col[i];
			}
		}
		dA_colchk[y_compute + x_compute * ldda_colchk] = res;
	}
}

template<typename T, int64_t NROW, int64_t NCOL>
__global__ void encode_row_v5(int num_batches,
					T *dA, int64_t ldda, int64_t strideA, 
					 T *dA_rowchk, int64_t ldda_rowchk, int64_t strideA_rowchk) {

	const int batch_id = blockIdx.x;
	const int tid = threadIdx.x;
	const int y = tid / NROW;
	const int x = tid % NROW;
	dA = dA + batch_id * strideA;
	dA_rowchk = dA_rowchk + batch_id * strideA_rowchk;

	// printf("%d %d\n", x, y);

	if (x < NROW && y < 2) {
		T res = 0.0;
		T * dA_row = &dA[x];
		if (y == 0) {
			for (int i = 0; i < NCOL; i++) {
				res += dA_row[i * NROW];
			}
		}
		if (y == 1) {
			for (int i = 0; i < NCOL; i++) {
				res += (T)(i+1) * dA_row[i * NROW];
			}
		}
		dA_rowchk[y * NROW + x] = res;
	}
}
'''
import numpy as np
from fault_inject import bitflip

np.random.seed(42)

X = np.random.randn(8, 8).astype(np.float32)
Y = np.random.randn(8, 8).astype(np.float32)
Z = X @ Y
Z_faulty = bitflip(Z.copy())

chk1 = X.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (X * weights).sum(axis=0)
X_aug = np.vstack([X, chk1, chk2]).astype(np.float32)

chk1_y = Y.sum(axis=1, keepdims=True)
weights_y = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
chk2_y = (Y * weights_y).sum(axis=1, keepdims=True)
Y_aug = np.hstack([Y, chk1_y, chk2_y]).astype(np.float32)

Z_aug = X_aug @ Y_aug

chk1_z = Z_faulty.sum(axis=0)
weights_z = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2_z = (Z_faulty * weights_z).sum(axis=0)
Z_0 = np.vstack([Z_faulty, chk1_z, chk2_z]).astype(np.float32)

chk1_z2 = Z_0.sum(axis=1, keepdims=True)
weights_z2 = np.arange(1, 11, dtype=np.float32).reshape(1, 10)
chk2_z2 = (Z_0 * weights_z2).sum(axis=1, keepdims=True)
Z_faulty_aug = np.hstack([Z_0, chk1_z2, chk2_z2]).astype(np.float32)

def col_detect0D(Z_aug: np.ndarray, Z_faulty_aug: np.ndarray):
    for col in range(8):
        d1 = Z_aug[8, col] - Z_faulty_aug[8, col]
        abs_d1 = abs(float(d1))
        
        if abs_d1 < 1e-3:
            continue 
        
        if np.isinf(d1):
            col_data = Z_faulty[:, col]
            inf_rows = np.where(np.isinf(col_data))[0]
            large_rows = np.where(np.abs(col_data) > 1e10)[0]
            err_rows = np.union1d(inf_rows, large_rows)
            
            if len(err_rows) > 1:
                print(f"列 {col}: 多个错误, 无法纠正")
                continue
            if len(err_rows) == 1:
                row = err_rows[0]
                correct_sum = Z_aug[8, col] - np.nansum(np.delete(col_data, row))
                Z_faulty[row, col] = correct_sum
                print(f"列 {col}: INF 已纠正，位置 ({row}, {col})")
            continue
        
        if np.isnan(d1):
            col_data = Z_faulty[:, col]
            nan_rows = np.where(np.isnan(col_data))[0]
            if len(nan_rows) > 1:
                print(f"列 {col}: 多个 NaN, 无法纠正")
                continue
            if len(nan_rows) == 1:
                row = nan_rows[0]
                correct_sum = Z_aug[8, col] - np.nansum(np.delete(col_data, row))
                Z_faulty[row, col] = correct_sum
                print(f"列 {col}: NaN 已纠正，位置 ({row}, {col})")
            continue

        d2 = Z_aug[9, col] - Z_faulty_aug[9, col]
        
        if not np.isinf(d2):
            loc = int(round(float(d2) / float(d1))) - 1
            if 0 <= loc < 8:
                if abs_d1 > 1e3:
                    col_data = Z_faulty[:, col]
                    correct_sum = Z_aug[8, col] - np.sum(np.delete(col_data, loc))
                    Z_faulty[loc, col] = correct_sum
                else:
                    Z_faulty[loc, col] += float(d1)
                print(f"列 {col}: 错误已纠正，位置 ({loc}, {col}), d1={d1:.4f}")
            else:
                print(f"列 {col}: 定位越界 loc={loc}")
        else:
            print(f"列 {col}: d2 为 INF, 无法定位")
col_detect0D(Z_aug, Z_faulty_aug)
'''
detection_col源码:
template <typename T>
__global__ void
detect_correct_col(T * dA, int64_t ldda, float E, int64_t stridea,
						     T * dA_colchk, 	int64_t ldda_colchk,	int64_t stride_colchk,
						     T * dA_colchk_r, int64_t ldda_colchk_r,	int64_t stride_colchk_r){
    //printf("col_chk kernel func. \n");
	//determin the block to process
	// printf("determin the block to process. \n");
    dA = dA + blockIdx.x * stridea;
	dA_colchk = dA_colchk + blockIdx.x * stride_colchk;
	dA_colchk_r = dA_colchk_r + blockIdx.x * stride_colchk_r;
    
    //determine the specific colum to process
	// printf("determin the specific colum to process. \n");
    dA = dA + threadIdx.x * ldda;
    dA_colchk   = dA_colchk   + threadIdx.x * ldda_colchk;
    dA_colchk_r = dA_colchk_r + threadIdx.x * ldda_colchk_r;
	
    float d1 = (float)((*dA_colchk)       - (*dA_colchk_r));
    float d2 = (float)(*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
	float abs_d1 = fabs(d1);
	int loc = -1;
	int locT = -1;
	float MAX;

	// E < abs d1 < INF
	if(abs_d1 > E && !isinf(abs_d1) && !isnan(abs_d1)) {
		if(!isinf(d2)){
			// d2 != INF
			//locate the error
			
			int counter = 0;
			// if more than one large number in the col
			for(int i = 0; i < ldda; i++){
				if(fabs((float)*(dA+i)) > 1e10){
					counter++;
					if(counter > 1){
						printf("[col check]col chksum error, more than one large number. (d1 = %.6f, d2 = %.6f, iter = %d)\n",
									(float)d1, (float)d2, i);
						return;
					}
				}
			}

			loc = round(d2 / d1) - 1;
			printf("[col check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n",  (float)d1, (float)d2, loc);
			//correction
			// *(dA+loc) += d1;
			if(abs_d1 > (float)1e3){
				// printf("d1 > threshold.\n");
				T sum = 0.0;
				for(int i = 0; i < ldda; i++) {
					if (i != loc) {
						sum +=	*(dA + i); 
					}
				}
				//correct the error
				*(dA + loc) = *dA_colchk - sum;
			}
			else{
				// printf("d1 =< threshold.\n");
				*(dA + loc) += d1;
			} 
		}
		else{
			if(isinf(*(dA_colchk + 1))){
				// C1,j == INF
				printf("[col check]Error detected in INPUTS.\n");
				return;
			}
			else{
				// C1,j != INF
				MAX = 0;
				int counter = 0;
				for(int i = 0; i < ldda; i++) {
					if(fabs((float)*(dA+i)) > MAX){
						MAX = fabs((float)*(dA+i));
						loc = i;
					}
					// if((*(dA+i)) < MIN){
					// 	MIN = *(dA+i);
					// 	locT = i;
					// }
					if(fabs((float)*(dA+i)) > 1e10){
						counter++;
						if(counter > 1){
							printf("[col check]col chksum error, more than one large number. (d1 = %.6f, d2 = %.6f)\n",(float)d1, (float)d2);
							return;
						}
					}
				}
				// if(fabs(((float)MAX)) < fabs(((float)MIN))){
				// 	loc = locT;
				// }
				printf("[col check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
				//correction
				// *(dA+loc) += d1;
				if(abs_d1 > (float)1e3){
					// printf("d1 > threshold.\n");
					T sum = 0.0;
					for(int i = 0; i < ldda; i++) {
						if (i != loc) {
							sum +=	*(dA + i); 
						}
					}
					//correct the error
					*(dA + loc) = *dA_colchk - sum;
				}
				else{
					// printf("d1 =< threshold.\n");
					*(dA + loc) += d1;
				} 
			}
		}
		return;
	}
	// abs = inf
	if(isinf(abs_d1)){
		MAX = 0;
		int64_t counter = 0;
		for(int i = 0; i < ldda; i++) {
			if(fabs((float)*(dA+i)) > MAX){
				MAX = fabs((float)*(dA+i));
				loc = i;
			}
			// if(*(dA+i) < MIN){
			// 	MIN = *(dA+i);
			// }
			if(isinf(*(dA+i)) || fabs((float)*(dA+i)) > 1e10){
				counter++;
				if(counter > 1){
					printf("[col check]Multi INFs or Large Number detected in one column.(d1 = %.6f, d2 = %.6f, iter = %d)\n",
										(float)d1, (float)d2, i);
					return;
				}
			}
		}
		if(counter == 0){
			printf("[col chk]No INF or Large Number found.\n");
			return;
		}
		// if(fabs((T)MAX) < fabs((T)MIN)){
		// 	MAX = MIN;
		// }
		// for(int i = 0; i < ldda; i++) {
		// 	if (fabs((float)*(dA+i)) == MAX || isinf(*(dA+i))) {
		// 		loc = i;
		// 		break;
		// 	}
		// }
		printf("[col check]INF detected (d1 = %.6f, d2 = %.6f, loc = %d, %.6f, %.6f) \n", 
											(float)d1, (float)d2, loc,(float)*(dA+29), (float)*(dA+loc));
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
		return;
	}
	// abs == nan
	if(isnan(abs_d1)){
		int64_t counter = 0;
		for(int i = 0; i < ldda; i++) {
			if (isnan(*(dA+i))) {
				loc = i;
				counter++;
			}
			if(isinf(*(dA+i))){
				counter++;
			}
			if(fabs((float)*(dA+i)) > 1e10){
				counter++;
			}
			if(counter > 1){
				printf("[col check]Multi INF, NAN or Large Number detected in one column. (iter = %d)\n", i);
				return;
			}
		}
		// if(loc == -1){
			// printf("[col check]No found NAN for d1 = NAN (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d, chk = %.6f, chk_r = %.6f) \n",
			// 								blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc, (float)(*dA_colchk), (float)(*dA_colchk_r));
			// return;
		// }
		printf("[col check]NAN detected (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",  
											blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
		return;
	}
}
'''