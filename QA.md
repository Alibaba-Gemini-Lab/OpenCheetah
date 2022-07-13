## Q&A
* Q: how to run other network and data ?

  A: OpenCheetah is built on top of the SCI library for neural network inference. 
     To test on new network and dataset, just use the scripts provided in the [EzPC project](https://github.com/mpc-msri/EzPC/tree/master/Athos)

* Q: The program throws `result ciphertext is transparent` logic_error
	
  A: That is a runtime error from the SEAL. Usually, when multiplying 0 with a ciphertext will raise that error.
     If we just want to a dry-run, then we can just turn of the flag `-DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT` in `scripts/build-deps.sh` and then rebuild SEAL.
