
HEADERS=test.cu # mlp.h # common.h cost_func.h layer.h linalg.h

test:
	nvcc -rdc=true --run mlp_mem_management.cu mlp_execution.cu linalg.cu layer.cu cost_func.cu activation_func.cu