
all: clops.dll

clops.dll: clops.c
	gcc -shared -fPIC clops.c `pkg-config --cflags --libs guile-2.0` \
		-I ./OpenCL-Headers /c/Windows/System32/OpenCL.dll \
		/usr/lib/libguile-2.0.dll.a  -o clops.dll
