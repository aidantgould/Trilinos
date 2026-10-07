// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/*
// @header
//
// ***********************************************************************
//
//      teko: a package for block and physics based preconditioning
//                  copyright 2010 sandia corporation
//
// under the terms of contract de-ac04-94al85000 with sandia corporation,
// the u.s. government retains certain rights in this software.
//
// redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. neither the name of the corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// this software is provided by sandia corporation "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. in no event shall sandia corporation or the
// contributors be liable for any direct, indirect, incidental, special,
// exemplary, or consequential damages (including, but not limited to,
// procurement of substitute goods or services; loss of use, data, or
// profits; or business interruption) however caused and on any theory of
// liability, whether in contract, strict liability, or tort (including
// negligence or otherwise) arising in any way out of the use of this
// software, even if advised of the possibility of such damage.
//
// questions? contact eric c. cyr (eccyr@sandia.gov)
//
// ***********************************************************************
//
// @header
*/

#include "Teko_InverseFactory.hpp"

// Thyra includes
#include "Thyra_DefaultLinearOpSource.hpp"
#include "Thyra_DefaultInverseLinearOp.hpp"
#include "Thyra_DefaultPreconditioner.hpp"

// Stratimikos includes
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

// Teko includes
#include "Teko_Utilities.hpp"
#include "Teko_BlockPreconditionerFactory.hpp"
#include "Teko_FactorTimeRegistry.hpp"
#include "Teko_Preconditioner.hpp"
#include "Teko_PreconditionerLinearOp.hpp"
#include "Teko_SolveInverseFactory.hpp"
#include "Teko_PreconditionerInverseFactory.hpp"
#include "Teko_KrylovSurrogateRegistration.hpp"

#include <iostream>
#include <set>
#include <string>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;

namespace Teko {

// ── Branch tracer ─────────────────────────────────────────────────────────
// Says, once per process per call site, which Trilinos source tree this
// binary was actually built from. Diagnostic only: nothing here throws, and
// no return value or control flow depends on it.
//
// Written to BOTH stdout and stderr, flushed: stdout is where an application
// logs, but is fully buffered to a file or pipe and lost if the run aborts,
// while stderr is unbuffered and survives that but is often redirected away.
// A caller merging the streams sees each line twice, which is the price.
//
// Deliberately a plain function called from real code, NOT a file-scope static
// initializer: a static link drops object files nothing references, which is
// exactly why Teko's adaptive hook registration silently never runs in a
// statically linked application. A call from a function the application
// actually invokes cannot be dropped that way.
namespace {
void announceTrilinosBranchOnce(const char* site) {
  static const char* const kBranch = "aidantgould/teko-reconfig-diag";
  // one line per site, not per call: a per-call line would swamp a real run
  static std::set<std::string> announced;
  if (!announced.insert(site).second) return;
  const std::string line =
      std::string("[Trilinos] branch ") + kBranch + " | " + site + "\n";
  std::cout << line << std::flush;
  std::cerr << line << std::flush;
}
}  // namespace


namespace FactorTimeRegistry {

namespace {
// Single process-wide accumulator; out-of-line on purpose so every shared
// library sees the same instance (see Teko_FactorTimeRegistry.hpp).
std::atomic<double> g_factorSeconds{0.0};
}  // namespace

void add(double seconds) { g_factorSeconds.fetch_add(seconds); }
double read() { return g_factorSeconds.load(); }
void reset() { g_factorSeconds.store(0.0); }

}  // namespace FactorTimeRegistry

namespace {
// Nesting depth of the free buildInverse/rebuildInverse functions on this
// thread. A block preconditioner factory builds its diagonal-block inverses
// through these same functions, so one application-level call can nest several
// deep. Depth 0 on entry marks the outermost call, the only one that warms up
// or records factor time.
thread_local int t_buildDepth = 0;

class BuildDepthGuard {
 public:
  BuildDepthGuard() { ++t_buildDepth; }
  ~BuildDepthGuard() { --t_buildDepth; }
  BuildDepthGuard(const BuildDepthGuard&)            = delete;
  BuildDepthGuard& operator=(const BuildDepthGuard&) = delete;
};

// Adds its lifetime to the FactorTimeRegistry on destruction, so the
// buildInverse/rebuildInverse wall time is recorded on both the normal and
// the throwing path. Records only for the outermost call: a nested build's
// time is already inside its parent's span, and counting it again would
// double it (or, during the warm-up, charge the warm-up to the registry).
class FactorStopwatch {
 public:
  explicit FactorStopwatch(bool record)
      : record_(record), start_(std::chrono::steady_clock::now()) {}
  ~FactorStopwatch() {
    if (!record_) return;
    FactorTimeRegistry::add(std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start_)
                                .count());
  }
  FactorStopwatch(const FactorStopwatch&)            = delete;
  FactorStopwatch& operator=(const FactorStopwatch&) = delete;

 private:
  bool record_;
  std::chrono::steady_clock::time_point start_;
};

// Unset, empty, "0", "false" and "FALSE" are false, anything else is true: the
// same rule as KrylovSurrogate::adaptiveEnabled() for TEKO_ADAPTIVE_RECONFIG.
bool envTruthy(const char* v) {
  const std::string s = v ? std::string(v) : "";
  return !(s.empty() || s == "0" || s == "false" || s == "FALSE");
}

// One-time warm-up so the first *timed* factorization is not charged for
// process-level initialization (Stratimikos/Ifpack2 first-use setup, the
// first Kokkos kernel launch / OpenMP team spawn, allocator warmup). The very
// first outermost buildInverse(factory, A) call factors A once untimed before
// the real timed factorization, so that one-time cost lands outside any
// measured solve and the first and second adaptive solves are compared
// warm-vs-warm.
//
// TEKO_FACTOR_WARMUP turns it on or off explicitly. Left unset, it follows
// TEKO_ADAPTIVE_RECONFIG, so an application that never enables the adaptive
// hook pays for no extra factorization.
//
// Called only at depth 0. The warm-up's own factory.buildInverse(A) then runs
// at depth 1, so the nested buildInverse calls a block preconditioner makes
// skip this function instead of re-entering the call_once below on the same
// thread, which would wait forever.
void maybeWarmupFactor(const InverseFactory& factory, const LinearOp& A) {
  static const bool enabled = []() {
    const char* v = std::getenv("TEKO_FACTOR_WARMUP");
    if (v != nullptr) return envTruthy(v);
    return envTruthy(std::getenv("TEKO_ADAPTIVE_RECONFIG"));
  }();
  if (!enabled) return;
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    try {
      InverseLinearOp warm = factory.buildInverse(A);  // untimed; result discarded
      (void)warm;
    } catch (...) {
      // A warm-up failure (e.g. a singular first operator) must not mask the
      // real, error-reporting factorization below.
    }
  });
}
}  // namespace

//! Build an inverse operator using a factory and a linear operator
InverseLinearOp buildInverse(const InverseFactory& factory, const LinearOp& A) {
  announceTrilinosBranchOnce("Teko::buildInverse(factory,A)");
  KrylovSurrogate::ensureAdaptiveHookRegistered("Teko::buildInverse");
  const bool outermost = (t_buildDepth == 0);
  BuildDepthGuard depth;
  if (outermost) maybeWarmupFactor(factory, A);
  FactorStopwatch stopwatch(outermost);
  InverseLinearOp inv;
  try {
    inv = factory.buildInverse(A);
  } catch (std::exception& e) {
    RCP<Teuchos::FancyOStream> out = Teko::getOutputStream();

    *out << "Teko: \"buildInverse\" could not construct the inverse operator using ";
    *out << "\"" << factory.toString() << "\"" << std::endl;
    *out << std::endl;
    *out << "*** THROWN EXCEPTION ***\n";
    *out << e.what() << std::endl;
    *out << "************************\n";

    throw e;
  }

  return inv;
}

/** Build an inverse operator using a factory and a linear operator
 *
 * \param[in] factory The inverse factory used to construct the inverse
 *                    operator
 * \param[in] precOp  Preconditioning operator
 * \param[in] A       Linear operator whose inverse is required
 *
 * \returns An (approximate) inverse operator is returned for the operator <code>A</code>.
 *
 * \relates InverseFactory
 */
InverseLinearOp buildInverse(const InverseFactory& factory, const LinearOp& A,
                             const LinearOp& precOp) {
  announceTrilinosBranchOnce("Teko::buildInverse(factory,A,precOp)");
  KrylovSurrogate::ensureAdaptiveHookRegistered("Teko::buildInverse");
  Teko_DEBUG_SCOPE("buildInverse(factory,A,precOp)", 10);
  const bool outermost = (t_buildDepth == 0);
  BuildDepthGuard depth;
  FactorStopwatch stopwatch(outermost);
  InverseLinearOp inv;
  try {
    inv = factory.buildInverse(A, precOp);
  } catch (std::exception& e) {
    RCP<Teuchos::FancyOStream> out = Teko::getOutputStream();

    *out << "Teko: \"buildInverse\" could not construct the inverse operator using ";
    *out << "\"" << factory.toString() << "\"" << std::endl;
    *out << std::endl;
    *out << "*** THROWN EXCEPTION ***\n";
    *out << e.what() << std::endl;
    *out << "************************\n";

    throw e;
  }

  return inv;
}

/** Using a prebuilt linear operator, use factory to build an inverse operator
 * given a new forward operator.
 */
void rebuildInverse(const InverseFactory& factory, const LinearOp& A, InverseLinearOp& invA) {
  KrylovSurrogate::ensureAdaptiveHookRegistered("Teko::rebuildInverse");
  const bool outermost = (t_buildDepth == 0);
  BuildDepthGuard depth;
  FactorStopwatch stopwatch(outermost);
  InverseLinearOp inv;
  try {
    factory.rebuildInverse(A, invA);
  } catch (std::exception& e) {
    RCP<Teuchos::FancyOStream> out = Teko::getOutputStream();

    *out << "Teko: \"rebuildInverse\" could not construct the inverse operator using ";
    *out << "\"" << factory.toString() << "\"" << std::endl;
    *out << std::endl;
    *out << "*** THROWN EXCEPTION ***\n";
    *out << e.what() << std::endl;
    *out << "************************\n";

    throw e;
  }
}

/** Using a prebuilt linear operator, use factory to build an inverse operator
 * given a new forward operator.
 *
 * \note This function sometimes fails depending on the underlying type
 *       of the inverse factory.  Use with caution.
 *
 * \param[in] factory The inverse factory used to construct the inverse
 *                    operator
 * \param[in] A       Linear operator whose inverse is required
 * \param[in] precOp  Preconditioning operator
 * \param[in] invA    The inverse operator that is to be rebuilt using
 *                    the <code>A</code> operator.
 *
 * \relates InverseFactory
 */
void rebuildInverse(const InverseFactory& factory, const LinearOp& A, const LinearOp& precOp,
                    InverseLinearOp& invA) {
  KrylovSurrogate::ensureAdaptiveHookRegistered("Teko::rebuildInverse");
  const bool outermost = (t_buildDepth == 0);
  BuildDepthGuard depth;
  FactorStopwatch stopwatch(outermost);
  InverseLinearOp inv;
  try {
    factory.rebuildInverse(A, precOp, invA);
  } catch (std::exception& e) {
    RCP<Teuchos::FancyOStream> out = Teko::getOutputStream();

    *out << "Teko: \"rebuildInverse\" could not construct the inverse operator using ";
    *out << "\"" << factory.toString() << "\"" << std::endl;
    *out << std::endl;
    *out << "*** THROWN EXCEPTION ***\n";
    *out << e.what() << std::endl;
    *out << "************************\n";

    throw e;
  }
}

/** \brief Build an InverseFactory object from a ParameterList, as specified in Stratimikos.
 *
 * Build an InverseFactory object from a ParameterList, as specified in Stratimikos.
 * The specific inverse routine (either solver or preconditioner) to be chosen is specified
 * by a string.
 *
 * \param[in] list ParameterList that describes the available solvers/preconditioners.
 * \param[in] type String saying which solver/preconditioner to use.
 *
 * \returns An inverse factory using the specified inverse operation.
 */
RCP<InverseFactory> invFactoryFromParamList(const Teuchos::ParameterList& list,
                                            const std::string& type) {
  RCP<Teuchos::ParameterList> myList = rcp(new Teuchos::ParameterList(list));

  Stratimikos::DefaultLinearSolverBuilder strat;
  addToStratimikosBuilder(rcpFromRef(strat));
  strat.setParameterList(myList);

  try {
    // try to build a preconditioner factory
    RCP<Thyra::PreconditionerFactoryBase<double> > precFact =
        strat.createPreconditioningStrategy(type);

    // string must map to a preconditioner
    return rcp(new PreconditionerInverseFactory(precFact, Teuchos::null));
  } catch (const Teuchos::Exceptions::InvalidParameterValue& exp) {
  }

  try {
    // try to build a solver factory
    RCP<Thyra::LinearOpWithSolveFactoryBase<double> > solveFact =
        strat.createLinearSolveStrategy(type);

    // if its around, build a InverseFactory
    return rcp(new SolveInverseFactory(solveFact));
  } catch (const Teuchos::Exceptions::InvalidParameterValue& exp) {
  }

  return Teuchos::null;
  ;
}

/** \brief Get a valid parameter list for the inverse factory class.
 *
 * Get a valid parameter list for the inverse factory class. This will
 * specify the set of parameters for each possible "inverse".
 *
 * \returns A parameter list is returned that is suitable to be passed
 *          to <code>invFactoryFromParamList</code>.
 */
Teuchos::RCP<const Teuchos::ParameterList> invFactoryValidParameters() {
  Stratimikos::DefaultLinearSolverBuilder strat;

  // extract valid parameter list from Stratimikos
  return strat.getValidParameters();
}

}  // end namespace Teko
