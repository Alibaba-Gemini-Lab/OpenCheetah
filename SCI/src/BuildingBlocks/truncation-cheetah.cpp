// Author: Zhichong Huang
void Truncation::truncate(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t shift, int32_t bw, bool signed_arithmetic,
                          uint8_t *msb_x) {
  if (msb_x != nullptr)
    return truncate(dim, inA, outB, shift, bw, signed_arithmetic, msb_x);

  if (shift == 0) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_upper =
      ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_orig = new uint64_t[dim];

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      inA_orig[i] = inA[i];
      inA[i] = ((inA[i] + (1ULL << (bw - 1))) & mask_bw);
    }
  }

  uint64_t *inA_upper = new uint64_t[dim];
  uint8_t *wrap_upper = new uint8_t[dim];
  for (int i = 0; i < dim; i++) {
    inA_upper[i] = inA[i] & mask_bw;
    if (party == sci::BOB) {
      inA_upper[i] = (mask_bw - inA_upper[i]) & mask_bw;
    }
  }

  this->mill->compare(wrap_upper, inA_upper, dim, bw);

  uint64_t *arith_wrap_upper = new uint64_t[dim];
  this->aux->B2A(wrap_upper, arith_wrap_upper, dim, shift);
  io->flush();

  for (int i = 0; i < dim; i++) {
    outB[i] = (((inA[i] >> shift) & mask_upper) -
               (1ULL << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (1ULL << (bw - shift - 1))) & mask_bw);
      inA[i] = inA_orig[i];
    }
  }
  delete[] inA_orig;
  delete[] inA_upper;
  delete[] wrap_upper;
  delete[] arith_wrap_upper;

  return;
}

void Truncation::truncate_msb(int32_t dim, uint64_t *inA, uint64_t *outB,
                              int32_t shift, int32_t bw, bool signed_arithmetic,
                              uint8_t *msb_x) {
  if (shift == 0) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  uint64_t mask_upper =
      ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_orig = new uint64_t[dim];

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      inA_orig[i] = inA[i];
      inA[i] = ((inA[i] + (1ULL << (bw - 1))) & mask_bw);
    }
  }

  uint64_t *inA_upper = new uint64_t[dim];
  uint8_t *wrap_upper = new uint8_t[dim];
  for (int i = 0; i < dim; i++) {
    inA_upper[i] = (inA[i] >> shift) & mask_upper;
    if (party == sci::BOB) {
      inA_upper[i] = (mask_upper - inA_upper[i]) & mask_upper;
    }
  }

  if (signed_arithmetic) {
    uint8_t *inv_msb_x = new uint8_t[dim];
    for (int i = 0; i < dim; i++) {
      inv_msb_x[i] = msb_x[i] ^ (party == sci::ALICE ? 1 : 0);
    }
    this->aux->MSB_to_Wrap(inA, inv_msb_x, wrap_upper, dim, bw);
    delete[] inv_msb_x;
  } else {
    this->aux->MSB_to_Wrap(inA, msb_x, wrap_upper, dim, bw);
  }

  uint64_t *arith_wrap_upper = new uint64_t[dim];
  this->aux->B2A(wrap_upper, arith_wrap_upper, dim, shift);
  io->flush();

  for (int i = 0; i < dim; i++) {
    outB[i] = (((inA[i] >> shift) & mask_upper) -
               (1ULL << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (1ULL << (bw - shift - 1))) & mask_bw);
      inA[i] = inA_orig[i];
    }
  }
  delete[] inA_orig;
  delete[] inA_upper;
  delete[] wrap_upper;
  delete[] arith_wrap_upper;

  return;
}

// Truncate (right-shift) by shift in the same ring (round towards -inf)
// All elements have msb equal to 0.
void Truncation::truncate_msb0(
    // Size of vector
    int32_t dim,
    // input vector
    uint64_t *inA,
    // output vector
    uint64_t *outB,
    // right shift amount
    int32_t shift,
    // Input and output bitwidth
    int32_t bw,
    // signed truncation?
    bool signed_arithmetic) {
  if (shift == 0) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  uint64_t mask_upper =
      ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_orig = new uint64_t[dim];

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      inA_orig[i] = inA[i];
      inA[i] = ((inA[i] + (1ULL << (bw - 1))) & mask_bw);
    }
  }

  uint8_t *wrap_upper = new uint8_t[dim];

  if (signed_arithmetic)
    this->aux->msb1_to_wrap(inA, wrap_upper, dim, bw);
  else
    this->aux->msb0_to_wrap(inA, wrap_upper, dim, bw);

  uint64_t *arith_wrap_upper = new uint64_t[dim];
  this->aux->B2A(wrap_upper, arith_wrap_upper, dim, shift);
  io->flush();

  for (int i = 0; i < dim; i++) {
    outB[i] = (((inA[i] >> shift) & mask_upper) -
               (1ULL << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (1ULL << (bw - shift - 1))) & mask_bw);
      inA[i] = inA_orig[i];
    }
  }
  delete[] inA_orig;
  delete[] wrap_upper;
  delete[] arith_wrap_upper;

  return;
}
