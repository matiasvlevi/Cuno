cmd_Release/obj.target/hello/geni/kernel.o := LD_LIBRARY_PATH=/home/vlev/Documents/Github/cuno/build/Release/lib.host:/home/vlev/Documents/Github/cuno/build/Release/lib.target:$$LD_LIBRARY_PATH; export LD_LIBRARY_PATH; cd ../.; mkdir -p /home/vlev/Documents/Github/cuno/build/Release/obj.target/hello/geni; nvcc -Xcompiler -fpic -c "/home/vlev/Documents/Github/cuno/src/kernel.cu" -o "/home/vlev/Documents/Github/cuno/build/Release/obj.target/hello/geni/kernel.o"
