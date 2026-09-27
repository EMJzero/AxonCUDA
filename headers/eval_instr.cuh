#pragma once
// -----------------------------------------------------------------------------
// Lightweight NVTX phase markers for the IPDPS-2027 performance evaluation.
//
// Header-only NVTX v3 (ships with CUDA). When no profiler (nsys/ncu) is
// attached, nvtxRangePush/Pop resolve to an inert function-pointer call --
// order ~1 ns each; a partitioning run issues only a few dozen, so the effect
// on the clean-timing numbers is unmeasurable.
//
// Build with -DAXON_NO_NVTX to compile these out entirely.
// -----------------------------------------------------------------------------
#if defined(AXON_NO_NVTX)
  #define EVP_PUSH(name) do {} while (0)
  #define EVP_POP()      do {} while (0)
  #define EVP_SCOPE(name) do {} while (0)
#else
  #include <nvtx3/nvToolsExt.h>
  #include <string>

  static inline void evp_push(const char* n)        { nvtxRangePushA(n); }
  static inline void evp_push(const std::string& n) { nvtxRangePushA(n.c_str()); }
  static inline void evp_pop()                      { nvtxRangePop(); }

  struct EvpScope {
      explicit EvpScope(const char* n)        { nvtxRangePushA(n); }
      explicit EvpScope(const std::string& n) { nvtxRangePushA(n.c_str()); }
      ~EvpScope()                             { nvtxRangePop(); }
      EvpScope(const EvpScope&) = delete;
      EvpScope& operator=(const EvpScope&) = delete;
  };
  #define EVP_CONCAT_(a, b) a##b
  #define EVP_CONCAT(a, b) EVP_CONCAT_(a, b)

  #define EVP_PUSH(name) evp_push(name)
  #define EVP_POP()      evp_pop()
  #define EVP_SCOPE(name) EvpScope EVP_CONCAT(evp_scope_, __LINE__)(name)
#endif
