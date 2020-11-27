
HEADERS=test.cu # mlp.h # common.h cost_func.h layer.h linalg.h

CORE_IMPLEMENTATION=mlp_mem_management.cu mlp_execution.cu linalg.cu layer.cu cost_func.cu activation_func.cu csv.cu

test: test.cu $(CORE_IMPLEMENTATION)
	nvcc -rdc=true --run $^


test_csv: csv.cu
	nvcc -rdc=true --run $^

