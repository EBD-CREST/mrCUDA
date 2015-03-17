all:
	gcc -I/opt/cuda/tk5.0/include/ -shared -fPIC -g -Wall -o libcudart.so.5.0 cudart.c -ldl

clean:
	rm -rf *.o *.so *.so.*
