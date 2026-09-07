import torch
import numpy as np
from fault_inject import bitflip
from Detection0D import col_detect0D

np.random.seed(42)

# 1R
X = np.random.randn(8, 8).astype(np.float32)
Y = np.random.randn(8, 8).astype(np.float32)
X_faulty = bitflip(X)
A = X @ Y
A_faulty = X_faulty @ Y

chk1 = A.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (A * weights).sum(axis=0)
A_aug = np.vstack([A, chk1, chk2]).astype(np.float32)

chk1 = A_faulty.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (A_faulty * weights).sum(axis=0)
A_faulty_aug = np.vstack([A_faulty, chk1, chk2]).astype(np.float32)

col_detect0D(A_aug, A_faulty_aug)

# 1C
B = Y @ X
B_faulty = Y @ X_faulty

chk1 = B.sum(axis=1, keepdims=True)
weights = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
chk2 = (B * weights).sum(axis=1, keepdims=True)
B_aug = np.hstack([B, chk1, chk2]).astype(np.float32)

chk1 = B_faulty.sum(axis=1, keepdims=True)
weights = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
chk2 = (B_faulty * weights).sum(axis=1, keepdims=True)
B_faulty_aug = np.hstack([B_faulty, chk1, chk2]).astype(np.float32)

def row_detect0D(B_aug: np.ndarray, B_faulty_aug: np.ndarray):
    for row in range(8):
            d1 = B_aug[row, 8] - B_faulty_aug[row, 8]
            abs_d1 = abs(float(d1))
            
            if abs_d1 < 1e-3:
                continue 
            
            if np.isinf(d1):
                row_data = B_faulty[row, :]
                inf_cols = np.where(np.isinf(row_data))[0]
                large_cols = np.where(np.abs(row_data) > 1e10)[0]
                err_cols = np.union1d(inf_cols, large_cols)
                
                if len(err_cols) > 1:
                    print(f"行 {row}: 多个错误, 无法纠正")
                    continue
                if len(err_cols) == 1:
                    col = err_cols[0]
                    correct_sum = B_aug[8, col] - np.nansum(np.delete(row_data, col))
                    B_faulty[row, col] = correct_sum
                    print(f"列 {col}: INF 已纠正，位置 ({row}, {col})")
                continue
            
            if np.isnan(d1):
                row_data = B_faulty[row, :]
                nan_cols = np.where(np.isnan(row_data))[0]
                if len(nan_cols) > 1:
                    print(f"列 {col}: 多个 NaN, 无法纠正")
                    continue
                if len(nan_cols) == 1:
                    col = nan_cols[0]
                    correct_sum = B_aug[row, 8] - np.nansum(np.delete(row_data, row))
                    B_faulty[row, col] = correct_sum
                    print(f"行 {row}: NaN 已纠正，位置 ({row}, {col})")
                continue

            d2 = B_aug[row, 9] - B_faulty_aug[row, 9]
            
            if not np.isinf(d2):
                loc = int(round(float(d2) / float(d1))) - 1
                if 0 <= loc < 8:
                    if abs_d1 > 1e3:
                        row_data = B_faulty[row, :]
                        correct_sum = B_aug[row, 8] - np.sum(np.delete(row_data, loc))
                        B_faulty[row, loc] = correct_sum
                    else:
                        B_faulty[row, loc] += float(d1)
                    print(f"行 {row}: 错误已纠正，位置 ({row}, {loc}), d1={d1:.4f}")
                else:
                    print(f"行 {row}: 定位越界 loc={loc}")
            else:
                print(f"列 {col}: d2 为 INF, 无法定位")
row_detect0D(B_aug, B_faulty_aug)

# Nondeterministic
X = np.random.randn(8, 8).astype(np.float32)
Y = np.random.randn(8, 8).astype(np.float32)
B = np.random.randn(8, 8).astype(np.float32)
X_faulty = bitflip(X)
A = Y @ X
A_faulty = Y @ X_faulty
B_faulty = bitflip(B)

chk1 = A.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (A * weights).sum(axis=0)
A_aug = np.vstack([A, chk1, chk2]).astype(np.float32)

chk1 = B.sum(axis=1, keepdims=True)
weights = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
chk2 = (B * weights).sum(axis=1, keepdims=True)
B_aug = np.hstack([B, chk1, chk2]).astype(np.float32)

chk1 = A_faulty.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (A_faulty * weights).sum(axis=0)
A_faulty_aug = np.vstack([A_faulty, chk1, chk2]).astype(np.float32)

chk1 = B_faulty.sum(axis=1, keepdims=True)
weights = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
chk2 = (B_faulty * weights).sum(axis=1, keepdims=True)
B_faulty_aug = np.hstack([B_faulty, chk1, chk2]).astype(np.float32)

C_aug = A_aug @ B_aug
C_faulty_aug = A_faulty_aug @ B_faulty_aug

'''
不知道错误是行还是列(模式未知), 若1列有多个错误则列chk被污染无法用于错误检测定位
换用行校验和
col_detect0D(C_aug, C_faulty_aug)
row_detect0D(C_aug, C_faulty_aug)
'''

'''
detection_row源码:
template<typename T>
__global__ void
detect_correct_row(T * dA, int64_t ldda, float E, int64_t stridea, int64_t col,
						    T * dA_rowchk, 	int64_t ldda_rowchk,	int64_t stride_rowchk,
						     T * dA_rowchk_r, int64_t ldda_rowchk_r,	int64_t stride_rowchk_r){
    // printf("row_chk kernel func. \n");
	//determin the block to process
	// printf("determin the block to process. \n");
    dA = dA + blockIdx.x * stridea;
    dA_rowchk = dA_rowchk + blockIdx.x * stride_rowchk;
    dA_rowchk_r = dA_rowchk_r + blockIdx.x * stride_rowchk_r;
        
    //determine the specific row to process
	// printf("determin the specific row to process. \n");
	dA = dA + threadIdx.x;
    dA_rowchk   = dA_rowchk   + threadIdx.x;
    dA_rowchk_r = dA_rowchk_r + threadIdx.x;
	
    float d1 = (float)(*dA_rowchk)                 - (*dA_rowchk_r);
    float d2 = (float)(*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
	float abs_d1 = fabs(d1);
	int loc = -1;
	int locT = -1;
	float MAX;

	if(abs_d1 > E && !isinf(abs_d1) && !isnan(abs_d1)) {
		if(!isinf(d2)){
			// d2 != INF
			//locate the error
			int counter = 0;
			// if more than one large number
			for(int i = 0; i < col; i++){
				if(fabs((float)*(dA + i*ldda)) > 1e10){
					counter++;
					if(counter > 1){
						printf("[row check]row chksum error. More than one Large Number detected. \n");
						return;
					}
				}
			}
			if(counter == 0){
				printf("[row chk]Recaculate row chk. No Large Number detected for d1 = INF (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d, %.6f) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc, (float)*(dA+21*ldda));
				T sum = 0.0;
				T sumW = 0.0;
				for(int i = 0; i < col; i++) {
					sum +=	*(dA + i * ldda); 
					sumW += (i+1)*(*(dA + i * ldda));
				}
				*(dA_rowchk) = sum;
				*(dA_rowchk + ldda_rowchk) = sumW;
				return;
			}
			loc = round(d2 / d1) - 1;
			printf("[row check]error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
			//correction
			// *(dA + loc * ldda) += d1;
			if(abs_d1 > (float)1e3){
				// printf("d1 > threshold.\n");
				T sum = 0.0;
				for (int i = 0; i < col; i++) {
					if (i != loc) {
						sum +=	*(dA + i * ldda); 
					}
				}
				*(dA + loc * ldda) = *dA_rowchk - sum;
			}
			else{
				// printf("d1 =< threshold.\n");
				*(dA + loc * ldda) += d1;
			} 	
		}
		else{
			if(isinf(*(dA_rowchk + ldda_rowchk))){
				// C1,j == INF
				// two cases: 1. inp matrxi has error; 2. inp matrix not error
				T sum = 0.0;
				T sumW = 0.0;
				for(int i = 0; i < col; i++) {
					if(fabs((float)*(dA+i*ldda)) > 1e10 || isinf(*(dA+i*ldda))){
						printf("[row check]Error detected in INPUTS.(loc = %d)\n", i);
						return;
					}
					else{
						sum +=	*(dA + i * ldda); 
						sumW += (i+1)*(*(dA + i * ldda));
					}
				}
				*(dA_rowchk) = sum;
				*(dA_rowchk + ldda_rowchk) = sumW;
				printf("[row check]Recaculate row chk. No found Large Number or INF for d1 = INF (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
				return;
			}
			else{
				// C1,j != INF
				MAX = 0;
				int counter = 0;
				for(int i = 0; i < col; i++) {
					if(fabs((float)*(dA + i * ldda)) > MAX){
						MAX = fabs((float)*(dA + i * ldda));
						loc = i;
					}
					// if(*(dA + i * ldda) < MIN){
					// 	MIN = *(dA+i*ldda);
					// 	locT = i;
					// }
					if(fabs((float)*(dA + i * ldda)) > 1e10){
						counter++;
						if(counter > 1){
							printf("[row check]row chksum error. More than one Large Number detected. \n");
							return;
						}
					}
				}
				// if(fabs((T)MAX) < fabs((T)MIN)){
				// 	loc = locT;
				// }
				printf("[row check]chk inf error detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
				//correction
				// *(dA + loc * ldda) += d1;
				// correction
				if(abs_d1 > (float)1e3){
					// printf("d1 > threshold.\n");
					T sum = 0.0;
					for (int i = 0; i < col; i++) {
						if (i != loc) {
							sum +=	*(dA + i * ldda); 
						}
					}
					*(dA + loc * ldda) = *dA_rowchk - sum;
				}
				else{
					// printf("d1 =< threshold.\n");
					*(dA + loc * ldda) += d1;
				} 		
			}
		}
		return;
	}
	// abs d1 = INF
	if(isinf(abs_d1)){
		// abs == inf
		int64_t counter = 0;
		MAX = 0;
		for(int i = 0; i < col; i++) {
			if(fabs((float)*(dA + i * ldda)) > MAX){
				MAX = fabs((float)*(dA + i * ldda));
				loc = i;
			}
			// if(*(dA + i * ldda) < MIN){
			// 	MIN = *(dA + i * ldda);
			// }
			if(isinf(*(dA + i * ldda)) || fabs((float)*(dA + i * ldda)) > 1e10){
				counter++;
				if(counter > 1){
					printf("[row check]Multi INFs or Large Number detected in one row. \n");
					return;
				}
			}
		}
		if(counter == 0){
			printf("[row check]Recaculate row chk. No found INF for d1 = INF (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, -1);
			// printf("(C0: %.6f, C1: %.6f, R1: %.6f, R2: %.6f) \n", (float)(*(dA_rowchk)), (float)(*(dA_rowchk + ldda_rowchk)),
			// 													(float)(*(dA_rowchk_r)), (float)(*(dA_rowchk_r + ldda_rowchk_r)));
			T sum = 0.0;
			T sumW = 0.0;
			for(int i = 0; i < col; i++) {
				sum +=	*(dA + i * ldda); 
				sumW += (i+1)*(*(dA + i * ldda));
			}
			*(dA_rowchk) = sum;
			*(dA_rowchk + ldda_rowchk) = sumW;
			return;
		}
		// for(int i = 0; i < col; i++) {
		// 	if (*(dA + i * ldda) == MAX || isinf(*(dA + i * ldda))) {
		// 		loc = i;
		// 		break;
		// 	}
		// }
		printf("[row check]INF detected (idx = (%d, %d), d1 = %.6f, d2 = %.6f, loc = %d) \n", 
								 (blockIdx.x),(threadIdx.x),  (float)d1, (float)d2, loc);
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for (int i = 0; i < col; i++) {
			if (i != loc) {
				sum +=	*(dA + i * ldda); 
			}
		}
		*(dA + loc * ldda) = *dA_rowchk - sum;
		return;
	}
	// abs d1 = NAN
	if(isnan(abs_d1)){
		int64_t counter = 0;
		// abs == nan
		for(int i = 0; i < col; i++) {
			if (isnan(*(dA + i * ldda))) {
				loc = i;
				counter++;
			}
			if (isinf(*(dA + i * ldda))){
				counter++;
			}
			if(counter > 1){
				printf("[row check]Multi INF or NAN detected in one row. \n");
			}
		}
		if(loc == -1){
			printf("[row check]Recaculate row chk. No found NAN for d1 = NAN (idx = (%d, %d) d1 = %.6f, d2 = %.6f, loc = %d) \n",
															blockIdx.x, threadIdx.x, (float)d1, (float)d2, loc);
			// printf("(C0: %.6f, C1: %.6f, R1: %.6f, R2: %.6f) \n", (float)(*(dA_rowchk)), (float)(*(dA_rowchk + ldda_rowchk)),
			// 													(float)(*(dA_rowchk_r)), (float)(*(dA_rowchk_r + ldda_rowchk_r)));
			T sum = 0.0;
			T sumW = 0.0;
			for(int i = 0; i < col; i++) {
				sum +=	*(dA + i * ldda); 
				sumW += (i+1)*(*(dA + i * ldda));
			}
			*(dA_rowchk) = sum;
			*(dA_rowchk + ldda_rowchk) = sumW;
			return;
		}
		printf("[row check]NAN detected (d1 = %.6f, d2 = %.6f, loc = %d) \n", (float)d1, (float)d2, loc);
		//the sum of the rest correct number except the error one
		T sum = 0.0;
		for(int i = 0; i < col; i++) {
			if (i != loc) {
				sum +=	*(dA + i * ldda); 
			}
		}
		//correct the error
		*(dA + loc * ldda) = *dA_rowchk - sum;
		return;
	}
}
'''