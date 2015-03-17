all:
	gcc -I/usr/local/cuda/include/ -shared -fPIC -g -Wall -o libcudart.so cudart.c -ldl

clean:
	rm -rf *.o *.so *.so.*
