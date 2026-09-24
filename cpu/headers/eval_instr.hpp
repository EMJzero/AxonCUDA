#pragma once
// -----------------------------------------------------------------------------
// Lightweight phase markers, host-side counterpart of the NVTX ranges in the
// CUDA version ('headers/eval_instr.cuh'), with the very same phase names.
//
// Each range accumulates its wall-clock time under its name; the totals are
// printed by 'evp_report' at the end of the run, giving per-phase times to put
// side by side with the NVTX ranges of an nsys profile of the CUDA version.
//
// Build with -DAXON_NO_NVTX to compile these out entirely.
// -----------------------------------------------------------------------------
#include <map>
#include <string>
#include <vector>
#include <iomanip>
#include <utility>
#include <iostream>

#include <omp.h>

#if defined(AXON_NO_NVTX)
  #define EVP_PUSH(name) do {} while (0)
  #define EVP_POP()      do {} while (0)
  #define EVP_REPORT()   do {} while (0)
#else
  struct EvpState {
      std::vector<std::pair<std::string, double>> stack; // open ranges, with their start time
      std::vector<std::pair<std::string, double>> totals; // closed ranges, in order of first appearance, with their total time
  };

  inline EvpState& evp_state() { static EvpState state; return state; }

  inline void evp_push(const std::string& n) { evp_state().stack.emplace_back(n, omp_get_wtime()); }

  inline void evp_pop() {
      auto& state = evp_state();
      if (state.stack.empty()) return;
      auto [name, start] = state.stack.back();
      state.stack.pop_back();
      const double elapsed = omp_get_wtime() - start;
      for (auto& [n, t] : state.totals) if (n == name) { t += elapsed; return; }
      state.totals.emplace_back(name, elapsed);
  }

  inline void evp_report() {
      std::cout << "Phase times:\n";
      for (const auto& [name, t] : evp_state().totals)
          std::cout << "  " << std::left << std::setw(24) << name << std::right << std::fixed << std::setprecision(3) << t * 1e3 << " ms\n";
  }

  #define EVP_PUSH(name) evp_push(name)
  #define EVP_POP()      evp_pop()
  #define EVP_REPORT()   evp_report()
#endif
