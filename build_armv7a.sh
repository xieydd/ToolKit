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
rm -rf build_android_armv7a
mkdir -p build_android_armv7a
pushd build_android_armv7a
export ANDROID_NDK=/opt/android-ndk-r15c
#/opt/cmake-3.14.5/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="armeabi-v7a" -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release -DANDROID_NATIVE_API_LEVEL=android-14 -DANDROID_TOOLCHAIN=gcc -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DCONFIG_ARCH_ARM32=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 ..
/opt/cmake-3.14.5/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_PLATFORM=android-22 -DANDROID_STL=c++_shared -DANDROID_ARM_NEON=ON -DCONFIG_ARCH_ARM32=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE DCMAKE_BUILD_TYPE=Release -DANDROID_TOOLCHAIN=gcc -DMNN_BUILD_FOR_ANDROID_COMMAND=true -DNCNN_BENCHMARK=ON ..
#/opt/cmake-3.9.0-Linux-x86_64/bin/cmake .. -DNCNN_OPENMP=ON \
#					-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
#					-DCMAKE_BUILD_TYPE=Release \
#					-DANDROID_STL=c++_static \
#				       	-DANDROID=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#					-DANDROID_ABI="armeabi-v7a with NEON" \
#				       	-DANDROID_ARM_NEON=ON \ 
#					-DANDROID_NATIVE_API_LEVEL=android-14 \
#					-DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9 \
#					-DANDROID_FORCE_ARM_BUILD=OFF \ 
#					-DMNN_USE_LOGCAT=false \
#					-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
#				       	-DANDROID_STL_FORCE_FEATURES=OFF \
#					-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.
make -j8 VERBOSE=1
make install/strip
popd
exit 0
