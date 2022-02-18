#ifndef CF2_OT_PACK_H__
#define CF2_OT_PACK_H__
#include "OT/emp-ot.h"
#include "OT/split-kkot.h"
#include "utils/emp-tool.h"
#define KKOT_TYPES 8

namespace sci {
template <typename T>
class OTPack {
 public:
  SplitKKOT<T> *kkot[KKOT_TYPES];

  // iknp_straight and iknp_reversed: party
  // acts as sender in straight and receiver in reversed.
  // Needed for MUX calls.
  SplitIKNP<T> *iknp_straight;
  SplitIKNP<T> *iknp_reversed;
  T *io;
  int party;
  bool do_setup = false;

  OTPack(T *io, int party, bool do_setup = true) {
    std::cout << "using kkot pack" << std::endl;
    this->party = party;
    this->do_setup = do_setup;
    this->io = io;

    for (int i = 0; i < KKOT_TYPES; i++) {
      kkot[i] = new SplitKKOT<NetIO>(party, io, 1 << (i + 1));
    }

    iknp_straight = new SplitIKNP<NetIO>(party, io);
    iknp_reversed = new SplitIKNP<NetIO>(3 - party, io);

    if (do_setup) {
      SetupBaseOTs();
    }
  }

  ~OTPack() {
    for (int i = 0; i < KKOT_TYPES; i++) delete kkot[i];
    delete iknp_straight;
    delete iknp_reversed;
  }

  void SetupBaseOTs() {
    switch (party) {
      case 1:
        kkot[0]->setup_send();
        iknp_straight->setup_send();
        iknp_reversed->setup_recv();
        for (int i = 1; i < KKOT_TYPES; i++) {
          kkot[i]->setup_send();
        }
        break;
      case 2:
        kkot[0]->setup_recv();
        iknp_straight->setup_recv();
        iknp_reversed->setup_send();
        for (int i = 1; i < KKOT_TYPES; i++) {
          kkot[i]->setup_recv();
        }
        break;
    }
    io->flush();
  }

  /*
   * DISCLAIMER:
   * OTPack copy method avoids computing setup keys for each OT instance by
   * reusing the keys generated (through base OTs) for another OT instance.
   * Ideally, the PRGs within OT instances, using the same keys, should use
   * mutually exclusive counters for security. However, the current
   * implementation does not support this.
   */

  void copy(OTPack<T> *copy_from) {
    assert(this->do_setup == false && copy_from->do_setup == true);
    SplitKKOT<T> *kkot_base = copy_from->kkot[0];
    SplitIKNP<T> *iknp_s_base = copy_from->iknp_straight;
    SplitIKNP<T> *iknp_r_base = copy_from->iknp_reversed;

    switch (this->party) {
      case 1:
        for (int i = 0; i < KKOT_TYPES; i++) {
          this->kkot[i]->setup_send(kkot_base->k0, kkot_base->s);
        }
        this->iknp_straight->setup_send(iknp_s_base->k0, iknp_s_base->s);
        this->iknp_reversed->setup_recv(iknp_r_base->k0, iknp_r_base->k1);
        break;
      case 2:
        for (int i = 0; i < KKOT_TYPES; i++) {
          this->kkot[i]->setup_recv(kkot_base->k0, kkot_base->k1);
        }
        this->iknp_straight->setup_recv(iknp_s_base->k0, iknp_s_base->k1);
        this->iknp_reversed->setup_send(iknp_r_base->k0, iknp_r_base->s);
        break;
    }
    this->do_setup = true;
    return;
  }
};

}  // namespace sci
#endif // CF2_OT_PACK_H__
