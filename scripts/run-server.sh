. scripts/common.sh

if [ ! $# -eq 2 ]
then
  echo -e "${RED}Please specify the network to run.${NC}"
  echo "Usage: run-server.sh [cheetah|SCI_HE] [sqnet/resnet50]"
else
  if ! contains "cheetah SCI_HE" $1; then
    echo -e "Usage: run-server.sh ${RED}[cheetah|SCI_HE]${NC} [sqnet|resnet50|densenet121]"
	exit 1
  fi

  if ! contains "sqnet resnet50 densenet121" $2; then
    echo -e "Usage: run-server.sh [cheetah|SCI_HE] ${RED}[sqnet|resnet50|densenet121]${NC}"
	exit 1
  fi
  # create a data/ to store the Ferret output
  mkdir -p data
  ls -lh pretrained/$2_model_scale12.inp
  echo -e "Runing ${GREEN}build/bin/$2-$1${NC}, which might take a while...."
  cat pretrained/$2_model_scale12.inp | build/bin/$2-$1 r=1 k=$FXP_SCALE ell=$SS_BITLEN nt=$NUM_THREADS p=$SERVER_PORT 1>$1-$2_server.log
  echo -e "Computation done, check out the log file ${GREEN}$1-$2_server.log${NC}"
fi
