
HEADERS=test.cu # mlp.h # activation_func.h common.h cost_func.h layer.h linalg.h

test:
	nvcc -rdc=true --run mlp.cu linalg.cu layer.cu