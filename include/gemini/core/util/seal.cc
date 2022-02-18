// Authors: Wen-jie Lu.
#include "gemini/core/util/seal.h"

#include <seal/ciphertext.h>
#include <seal/context.h>
#include <seal/galoiskeys.h>
#include <seal/util/polyarithsmallmod.h>

namespace gemini {

static Code divide_and_round_q_last_ntt_inplace(
    seal::util::RNSIter input, size_t input_base_size,
    seal::util::ConstNTTTablesIter rns_ntt_tables,
    const seal::util::RNSTool *rns_tool, seal::MemoryPoolHandle pool) {
  using namespace seal;
  using namespace seal::util;
  if (!input or !rns_tool or !rns_ntt_tables or !pool) {
    return Code::ERR_NULL_POINTER;
  }

  const size_t coeff_count = input.poly_modulus_degree();
  auto base_q = rns_tool->base_q();
  const size_t base_q_size = base_q->size();
  if (input_base_size >= base_q_size || input_base_size < 1) {
    return Code::ERR_INVALID_ARG;
  }

  CoeffIter last_input = input[input_base_size];
  // Convert to non-NTT form
  inverse_ntt_negacyclic_harvey(last_input, rns_ntt_tables[base_q_size - 1]);
  // Add (qi-1)/2 to change from flooring to rounding
  const Modulus &last_modulus = (*base_q)[base_q_size - 1];
  uint64_t half = last_modulus.value() >> 1;
  add_poly_scalar_coeffmod(last_input, coeff_count, half, last_modulus,
                           last_input);

  SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count, pool);
  SEAL_ITERATE(
      iter(input, rns_tool->inv_q_last_mod_q(), base_q->base(), rns_ntt_tables),
      input_base_size, [&](auto I) {
        // (ct mod qk) mod qi
        if (get<2>(I).value() < last_modulus.value()) {
          modulo_poly_coeffs(last_input, coeff_count, get<2>(I), temp);
        } else {
          set_uint(last_input, coeff_count, temp);
        }

        // Lazy subtraction here. ntt_negacyclic_harvey_lazy can take 0 < x <
        // 4*qi input.
        uint64_t neg_half_mod =
            get<2>(I).value() - barrett_reduce_64(half, get<2>(I));

        // Note: lambda function parameter must be passed by reference here
        SEAL_ITERATE(temp, coeff_count, [&](auto &J) { J += neg_half_mod; });

#if SEAL_USER_MOD_BIT_COUNT_MAX <= 60
        // Since SEAL uses at most 60-bit moduli, 8*qi < 2^63.
        // This ntt_negacyclic_harvey_lazy results in [0, 4*qi).
        uint64_t qi_lazy = get<2>(I).value() << 2;
        ntt_negacyclic_harvey_lazy(temp, get<3>(I));
#else
        // 2^60 < pi < 2^62, then 4*pi < 2^64, we perfrom one reduction from [0, 4*qi) to [0, 2*qi) after ntt.
        uint64_t qi_lazy = get<2>(I).value() << 1;
        ntt_negacyclic_harvey_lazy(temp, get<3>(I));

        // Note: lambda function parameter must be passed by reference here
        SEAL_ITERATE(temp, coeff_count, [&](auto &J) {
            J -= (qi_lazy & static_cast<uint64_t>(-static_cast<int64_t>(J >= qi_lazy)));
        });
#endif
        // Lazy subtraction again, results in [0, 2*qi_lazy),
        // The reduction [0, 2*qi_lazy) -> [0, qi) is done implicitly in
        // multiply_poly_scalar_coeffmod.
        SEAL_ITERATE(iter(get<0>(I), temp), coeff_count,
                     [&](auto J) { get<0>(J) += qi_lazy - get<1>(J); });

        // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
        multiply_poly_scalar_coeffmod(get<0>(I), coeff_count, get<1>(I),
                                      get<2>(I), get<0>(I));
      });
  return Code::OK;
}

static Code switch_key_inplace(seal::Ciphertext &encrypted,
                               seal::util::ConstRNSIter target_iter,
                               const seal::KSwitchKeys &kswitch_keys,
                               size_t kswitch_keys_index,
                               seal::MemoryPoolHandle pool,
                               const seal::SEALContext &context) {
  using namespace ::seal;
  using namespace ::seal::util;
  auto parms_id = encrypted.parms_id();
  auto &context_data = *context.get_context_data(parms_id);
  auto &parms = context_data.parms();
  auto &key_context_data = *context.key_context_data();
  auto &key_parms = key_context_data.parms();

  // Verify parameters.
  if (!is_metadata_valid_for(encrypted, context) ||
      !is_buffer_valid(encrypted)) {
    return Code::ERR_INVALID_ARG;
    // throw invalid_argument("encrypted is not valid for encryption
    // parameters");
  }
  if (!target_iter) {
    return Code::ERR_NULL_POINTER;
    // throw invalid_argument("target_iter");
  }
  if (!context.using_keyswitching()) {
    return Code::ERR_INTERNAL;
    // throw logic_error("keyswitching is not supported by the context");
  }

  // Don't validate all of kswitch_keys but just check the parms_id.
  if (kswitch_keys.parms_id() != context.key_parms_id()) {
    return Code::ERR_INVALID_ARG;
    // throw invalid_argument("parameter mismatch");
  }

  if (kswitch_keys_index >= kswitch_keys.data().size()) {
    return Code::ERR_OUT_BOUND;
    // throw out_of_range("kswitch_keys_index");
  }

  // Extract encryption parameters.
  size_t coeff_count = parms.poly_modulus_degree();
  size_t decomp_modulus_size = parms.coeff_modulus().size();
  auto &key_modulus = key_parms.coeff_modulus();
  size_t key_modulus_size = key_modulus.size();
  size_t rns_modulus_size = decomp_modulus_size + 1;
  auto key_ntt_tables = iter(key_context_data.small_ntt_tables());
  bool is_ntt_form = encrypted.is_ntt_form();

  // Size check
  if (!product_fits_in(coeff_count, rns_modulus_size, size_t(2))) {
    return Code::ERR_DIM_MISMATCH;
    // throw logic_error("invalid parameters");
  }

  // Prepare input
  auto &key_vector = kswitch_keys.data()[kswitch_keys_index];
  size_t key_component_count = key_vector[0].data().size();

  // Check only the used component in KSwitchKeys.
  for (auto &each_key : key_vector) {
    if (!is_metadata_valid_for(each_key, context) ||
        !is_buffer_valid(each_key)) {
      return Code::ERR_INVALID_ARG;
      // throw invalid_argument(
      //     "kswitch_keys is not valid for encryption parameters");
    }
  }

  // Create a copy of target_iter
  SEAL_ALLOCATE_GET_RNS_ITER(t_target, coeff_count, decomp_modulus_size, pool);
  set_uint(target_iter, decomp_modulus_size * coeff_count, t_target);

  // In CKKS t_target is in NTT form; switch back to normal form
  if (is_ntt_form) {
    inverse_ntt_negacyclic_harvey(t_target, decomp_modulus_size,
                                  key_ntt_tables);
  }

  // Temporary result
  auto t_poly_prod(allocate_zero_poly_array(key_component_count, coeff_count,
                                            rns_modulus_size, pool));

  SEAL_ITERATE(iter(size_t(0)), rns_modulus_size, [&](auto I) {
    size_t key_index = (I == decomp_modulus_size ? key_modulus_size - 1 : I);

    // Product of two numbers is up to 60 + 60 = 120 bits, so we can sum up to
    // 256 of them without reduction.
    size_t lazy_reduction_summand_bound =
        size_t(SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX);
    size_t lazy_reduction_counter = lazy_reduction_summand_bound;

    // Allocate memory for a lazy accumulator (128-bit coefficients)
    auto t_poly_lazy(
        allocate_zero_poly_array(key_component_count, coeff_count, 2, pool));

    // Semantic misuse of PolyIter; this is really pointing to the data for a
    // single RNS factor
    PolyIter accumulator_iter(t_poly_lazy.get(), 2, coeff_count);

    // Multiply with keys and perform lazy reduction on product's coefficients
    SEAL_ITERATE(iter(size_t(0)), decomp_modulus_size, [&](auto J) {
      SEAL_ALLOCATE_GET_COEFF_ITER(t_ntt, coeff_count, pool);
      ConstCoeffIter t_operand;

      // RNS-NTT form exists in input
      if (is_ntt_form && (I == J)) {
        t_operand = target_iter[J];
      }
      // Perform RNS-NTT conversion
      else {
        // No need to perform RNS conversion (modular reduction)
        if (key_modulus[J] <= key_modulus[key_index]) {
          set_uint(t_target[J], coeff_count, t_ntt);
        }
        // Perform RNS conversion (modular reduction)
        else {
          modulo_poly_coeffs(t_target[J], coeff_count, key_modulus[key_index],
                             t_ntt);
        }
        // NTT conversion lazy outputs in [0, 4q)
        ntt_negacyclic_harvey_lazy(t_ntt, key_ntt_tables[key_index]);
        t_operand = t_ntt;
      }

      // Multiply with keys and modular accumulate products in a lazy fashion
      SEAL_ITERATE(
          iter(key_vector[J].data(), accumulator_iter), key_component_count,
          [&](auto K) {
            if (!lazy_reduction_counter) {
              SEAL_ITERATE(iter(t_operand, get<0>(K)[key_index], get<1>(K)),
                           coeff_count, [&](auto L) {
                             unsigned long long qword[2]{0, 0};
                             multiply_uint64(get<0>(L), get<1>(L), qword);

                             // Accumulate product of t_operand and t_key_acc to
                             // t_poly_lazy and reduce
                             add_uint128(qword, get<2>(L).ptr(), qword);
                             get<2>(L)[0] = barrett_reduce_128(
                                 qword, key_modulus[key_index]);
                             get<2>(L)[1] = 0;
                           });
            } else {
              // Same as above but no reduction
              SEAL_ITERATE(iter(t_operand, get<0>(K)[key_index], get<1>(K)),
                           coeff_count, [&](auto L) {
                             unsigned long long qword[2]{0, 0};
                             multiply_uint64(get<0>(L), get<1>(L), qword);
                             add_uint128(qword, get<2>(L).ptr(), qword);
                             get<2>(L)[0] = qword[0];
                             get<2>(L)[1] = qword[1];
                           });
            }
          });

      if (!--lazy_reduction_counter) {
        lazy_reduction_counter = lazy_reduction_summand_bound;
      }
    });

    // PolyIter pointing to the destination t_poly_prod, shifted to the
    // appropriate modulus
    PolyIter t_poly_prod_iter(t_poly_prod.get() + (I * coeff_count),
                              coeff_count, rns_modulus_size);

    // Final modular reduction
    SEAL_ITERATE(
        iter(accumulator_iter, t_poly_prod_iter), key_component_count,
        [&](auto K) {
          if (lazy_reduction_counter == lazy_reduction_summand_bound) {
            SEAL_ITERATE(iter(get<0>(K), *get<1>(K)), coeff_count, [&](auto L) {
              get<1>(L) = static_cast<uint64_t>(*get<0>(L));
            });
          } else {
            // Same as above except need to still do reduction
            SEAL_ITERATE(iter(get<0>(K), *get<1>(K)), coeff_count, [&](auto L) {
              get<1>(L) =
                  barrett_reduce_128(get<0>(L).ptr(), key_modulus[key_index]);
            });
          }
        });
  });

  // Accumulated products are now stored in t_poly_prod
  // Perform modulus switching with scaling
  PolyIter t_poly_prod_iter(t_poly_prod.get(), coeff_count, rns_modulus_size);
  auto rns_tool = key_context_data.rns_tool();
  SEAL_ITERATE(
      iter(encrypted, t_poly_prod_iter), key_component_count, [&](auto I) {
        // ntt form -> ntt form
        divide_and_round_q_last_ntt_inplace(get<1>(I), decomp_modulus_size,
                                            key_ntt_tables, rns_tool, pool);
        if (!is_ntt_form) {
          inverse_ntt_negacyclic_harvey(get<1>(I), decomp_modulus_size,
                                        key_ntt_tables);
        }
        add_poly_coeffmod(get<0>(I), get<1>(I), decomp_modulus_size,
                          key_modulus, get<0>(I));
      });
  return Code::OK;
}

Code apply_galois_inplace(::seal::Ciphertext &encrypted, uint32_t galois_elt,
                          const ::seal::GaloisKeys &galois_keys,
                          const ::seal::SEALContext &context) {
  using namespace seal;
  using namespace seal::util;
  // Verify parameters.
  if (!is_metadata_valid_for(encrypted, context) ||
      !is_buffer_valid(encrypted)) {
    std::cerr << "encrypted is not valid for encryption parameters"
              << std::endl;
    return Code::ERR_INVALID_ARG;
  }

  // Don't validate all of galois_keys but just check the parms_id.
  if (galois_keys.parms_id() != context.key_parms_id()) {
    std::cerr << "galois_keys is not valid for encryption parameters"
              << std::endl;
    return Code::ERR_INVALID_ARG;
  }

  auto &context_data = *context.get_context_data(encrypted.parms_id());
  auto &parms = context_data.parms();
  auto &coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_modulus_size = coeff_modulus.size();
  size_t encrypted_size = encrypted.size();
  // Use key_context_data where permutation tables exist since previous runs.
  auto galois_tool = context.key_context_data()->galois_tool();
  auto pool = MemoryManager::GetPool(mm_prof_opt::mm_force_thread_local);

  // Size check
  if (!product_fits_in(coeff_count, coeff_modulus_size)) {
    // throw logic_error("invalid parameters");
    return Code::ERR_INVALID_ARG;
  }

  // Check if Galois key is generated or not.
  if (!galois_keys.has_key(galois_elt)) {
    // throw invalid_argument("Galois key not present");
    return Code::ERR_INVALID_ARG;
  }

  uint64_t m = mul_safe(static_cast<uint64_t>(coeff_count), uint64_t(2));

  // Verify parameters
  if (!(galois_elt & 1) || unsigned_geq(galois_elt, m)) {
    // throw invalid_argument("Galois element is not valid");
    return Code::ERR_INVALID_ARG;
  }
  if (encrypted_size > 2) {
    // throw invalid_argument("encrypted size must be 2");
    return Code::ERR_INVALID_ARG;
  }

  SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count, coeff_modulus_size, pool);

  // DO NOT CHANGE EXECUTION ORDER OF FOLLOWING SECTION
  // BEGIN: Apply Galois for each ciphertext
  // Execution order is sensitive, since apply_galois is not inplace!
  if (!encrypted.is_ntt_form()) {
    // !!! DO NOT CHANGE EXECUTION ORDER!!!

    // First transform encrypted.data(0)
    auto encrypted_iter = iter(encrypted);
    galois_tool->apply_galois(encrypted_iter[0], coeff_modulus_size, galois_elt,
                              coeff_modulus, temp);

    // Copy result to encrypted.data(0)
    set_poly(temp, coeff_count, coeff_modulus_size, encrypted.data(0));

    // Next transform encrypted.data(1)
    galois_tool->apply_galois(encrypted_iter[1], coeff_modulus_size, galois_elt,
                              coeff_modulus, temp);
  } else {
    // !!! DO NOT CHANGE EXECUTION ORDER!!!

    // First transform encrypted.data(0)
    auto encrypted_iter = iter(encrypted);
    galois_tool->apply_galois_ntt(encrypted_iter[0], coeff_modulus_size,
                                  galois_elt, temp);

    // Copy result to encrypted.data(0)
    set_poly(temp, coeff_count, coeff_modulus_size, encrypted.data(0));

    // Next transform encrypted.data(1)
    galois_tool->apply_galois_ntt(encrypted_iter[1], coeff_modulus_size,
                                  galois_elt, temp);
  }

  // Wipe encrypted.data(1)
  set_zero_poly(coeff_count, coeff_modulus_size, encrypted.data(1));

  // END: Apply Galois for each ciphertext
  // REORDERING IS SAFE NOW

  // Calculate (temp * galois_key[0], temp * galois_key[1]) + (ct[0], 0)
  Code ok = Code::OK;
  if (std::any_of(*temp, *temp + coeff_modulus_size * coeff_count,
                  [](uint64_t u) { return u > 0; })) {
    ok = switch_key_inplace(encrypted, temp,
                            static_cast<const KSwitchKeys &>(galois_keys),
                            GaloisKeys::get_index(galois_elt), pool, context);
  }

#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
  // Transparent ciphertext output is not allowed.
  if (encrypted.is_transparent()) {
    throw std::logic_error("result ciphertext is transparent");
  }
#endif
  return ok;
}

}  // namespace gemini
