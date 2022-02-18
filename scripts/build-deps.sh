. scripts/common.sh

check_tools

if [ -d .git ]; then
  git submodule init
  git submodule update
else
  git clone https://github.com/emp-toolkit/emp-tool.git $DEPS_DIR/emp-tool
  git clone https://github.com/emp-toolkit/emp-ot.git $DEPS_DIR/emp-ot
  git clone https://github.com/libigl/eigen.git $DEPS_DIR/eigen
  git clone https://github.com/facebook/zstd.git $DEPS_DIR/zstd
  git clone https://github.com/intel/hexl.git $DEPS_DIR/hexl
  git clone https://github.com/microsoft/SEAL.git $DEPS_DIR/SEAL
fi

target=emp-tool
cd $DEPS_DIR/$target
git checkout 44b1dde
patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/emp-tool.patch -d $DEPS_DIR/$target
mkdir -p $BUILD_DIR/deps/$target
cd $BUILD_DIR/deps/$target
if [ ! -d $BUILD_DIR/include/$target ]
then
  cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR
  make install -j2
fi

target=emp-ot
cd $DEPS_DIR/$target
git checkout 7f3d4f0
mkdir -p $BUILD_DIR/deps/$target
cd $BUILD_DIR/deps/$target
if [ ! -d $BUILD_DIR/include/$target ]
then
  cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DCMAKE_PREFIX_PATH=$BUILD_DIR
  make install -j2
fi

target=eigen
cd $DEPS_DIR/$target
git checkout 1f05f51 #v3.3.3
mkdir -p $BUILD_DIR/deps/$target
cd $BUILD_DIR/deps/$target
if [ ! -d $BUILD_DIR/include/eigen3 ]
then
  cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR
  make install -j2
fi

target=zstd
cd $DEPS_DIR/$target

if [ ! -f $BUILD_DIR/include/$target.h ]
then
  cmake $DEPS_DIR/$target/build/cmake -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_SHARED=OFF\
                                      -DZLIB_BUILD_STATIC=ON -DZSTD_BUILD_TESTS=OFF -DZSTD_MULTITHREAD_SUPPORT=OFF
  make install -j2
fi

target=hexl
cd $DEPS_DIR/$target
git checkout 343acab #v1.2.2
if [ ! -d $BUILD_DIR/include/$target ]
then
  cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DHEXL_BENCHMARK=OFF -DHEXL_COVERAGE=OFF -DHEXL_TESTING=OFF
  make install -j2
fi

target=SEAL
cd $DEPS_DIR/$target
git checkout 7923472 #v3.7.2
patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/SEAL.patch -d $DEPS_DIR/SEAL/
mkdir -p $BUILD_DIR/deps/$target
cd $BUILD_DIR/deps/$target
if [ ! -d $BUILD_DIR/include/SEAL-3.7 ]
then
   cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DCMAKE_PREFIX_PATH=$BUILD_DIR -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF\
	                       -DSEAL_USE_ZSTD=ON -DCMAKE_BUILD_TYPE=Release -DSEAL_USE_INTEL_HEXL=ON -DSEAL_BUILD_DEPS=OFF
   make install -j4
fi

for deps in eigen3 emp-ot emp-tool hexl SEAL-3.7
do
  if [ ! -d $BUILD_DIR/include/$deps ] 
  then
	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
	exit 1
  fi
done

for deps in zstd.h 
do
  if [ ! -f $BUILD_DIR/include/$deps ] 
  then
	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
	exit 1
  fi
done
