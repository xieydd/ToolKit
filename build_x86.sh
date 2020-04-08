#!/usr/bin/bash
if [ $# -eq 1 ]; then
	export INSTALL_PREFIX=$1
else
	echo "Please specify the install path"
	echo "$0 install_path"
	exit -1
fi
chmod 777 ./MNN/schema/generate.sh
bash ./MNN/schema/generate.sh
rm -rf build_linux_x86
mkdir -p build_linux_x86
pushd build_linux_x86
/opt/cmake-3.9.0-Linux-x86_64/bin/cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DNCNN_BENCHMARK=ON -DCMAKE_TOOLCHAIN_FILE=../Tengine/toolchains/x86.gcc.toolchain.cmake $1 $2 $3 ..
make -j8 VERBOSE=1
make install
popd
exit 0
