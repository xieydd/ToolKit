#!/usr/bin/bash
if [ $# -eq 1 ]; then
	export INSTALL_PREFIX=$1
else
	echo "Please specify the install path"
	echo "$0 install_path"
	exit -1
fi

rm -rf build_android_armv8
mkdir -p build_android_armv8
pushd build_android_armv8
export ANDROID_NDK=/opt/android-ndk-r15c
#/opt/cmake-3.9.0-Linux-x86_64/bin/cmake DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DMNN_USE_LOGCAT=false -DMNN_BUILD_BENCHMARK=ON -DANDROID_NATIVE_API_LEVEL=android-21 -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. -DANDROID_ARM_NEON=ON -DCONFIG_ARCH_ARM64=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE $1 $2 $3 ..
/opt/cmake-3.9.0-Linux-x86_64/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_shared -DANDROID_ARM_NEON=ON -DCONFIG_ARCH_ARM64=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE -DCMAKE_BUILD_TYPE=Release -DMNN_USE_LOGCAT=false -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE -DARM82=ON -DNCNN_BENCHMARK=ON ..
#/opt/cmake-3.9.0-Linux-x86_64/bin/cmake -DNCNN_OPENMP=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DANDROID=1 -DJNI_DEBUG=1 -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=android-9 -DANDROID_FORCE_ARM_BUILD=OFF -DANDROID_STL_FORCE_FEATURES=OFF ..
make -j8 VERBOSE=1
make install/strip
popd
exit 0
