RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

function has_tool {
  if ! command -v $1 &> /dev/null 
    then
      echo -e "No ${RED}$1${NC} is found."
      exit
	else
	  echo -e "${GREEN}$1${NC} found."
  fi
}

function check_tools {
  has_tool g++
  has_tool make
  has_tool git
  has_tool cmake
  cmake_major=3
  cmake_minior=10
  cmake_version=`cmake --version | head -n1 | awk '{print $3}'`
  cmake_match=`echo $cmake_version | awk -F. '{ if ($1 < 3 || $1 == 3 && $2 < 10) print(0); else print(1) }'`
  if [ $cmake_match -eq 0 ]; then
	echo -e "${RED}require cmake version >= $cmake_major.$cmake_minior but get $cmake_version${NC}"
	exit
  fi
}

function contains {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    # yes, list include item
    result=0
  else
    result=1
  fi
  return $result
}
WORK_DIR=`pwd`
BUILD_DIR=$WORK_DIR/build
DEPS_DIR=$WORK_DIR/deps
mkdir -p $BUILD_DIR
mkdir -p $DEPS_DIR

# change the ip if running remotely
SERVER_IP=127.0.0.1
SERVER_PORT=12345

# fixed-point scale
FXP_SCALE=12
# secret sharing bit length
SS_BITLEN=37
# number of threads (should <= 4 for the SCI)
NUM_THREADS=4
