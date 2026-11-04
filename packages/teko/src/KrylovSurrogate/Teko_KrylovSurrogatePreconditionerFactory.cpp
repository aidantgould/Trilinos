// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogatePreconditionerFactory.hpp"

#include <limits>
#include <string>

#include "Teuchos_Array.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"

#include "Tpetra_Map.hpp"

#include "Teko_InverseLibrary.hpp"
#include "Teko_KrylovSurrogateRecorderOp.hpp"
#include "Teko_KrylovSurrogateReducedOperator.hpp"
#include "Teko_KrylovSurrogateSearch.hpp"

namespace Teko {
namespace KrylovSurrogate {

namespace {

typedef Teuchos::RCP<Handle> HandlePtr;

template <class T>
Teuchos::RCP<const Teuchos::ParameterEntryValidator> inRange(T lo, T hi) {
  return Teuchos::rcp(new Teuchos::EnhancedNumberValidator<T>(lo, hi));
}

// Every option, with its default, documentation and range. Validating against
// this makes an unknown, mistyped or out-of-range option an error, so a
// misspelling does not silently do nothing.
Teuchos::RCP<const Teuchos::ParameterList> validParameters() {
  const Options defaults;
  const int big = std::numeric_limits<int>::max();
  auto pl       = Teuchos::rcp(new Teuchos::ParameterList);
  pl->set("Type", std::string("Krylov Surrogate"), "Selects this preconditioner");
  pl->set("Inverse Factory", std::string(""),
          "The Teko block inverse applied, and recorded (required)");
  pl->set("Report Handle", HandlePtr(),
          "The Teko::KrylovSurrogate::Handle the report is retrieved from (set from code)");
  pl->set("Maximum Samples", defaults.maxSamples,
          "Columns of preconditioner output kept, each a full-length vector of memory",
          inRange(1, big));
  pl->set("Truncation Tolerance", defaults.truncationTol,
          "Singular values kept per block: sigma_k >= tolerance * sigma_1", inRange(0.0, 1.0));
  Teuchos::ParameterList& search = pl->sublist("Search", false, "Candidate scoring");
  search.set("Tolerance", defaults.searchTol,
             "Relative residual each candidate's surrogate solve stops at", inRange(0.0, 1.0));
  search.set("Candidate Count", defaults.candidateCount, "Number of candidates scored",
             inRange(1, big));
  search.set("Seed", defaults.seed, "Seed for sampling candidates");
  search.sublist("Orderings", false, "Orderings always scored, each an Array(int)")
      .disableRecursiveValidation();
  Teuchos::ParameterList& report = pl->sublist("Report", false, "What is done with the report");
  report.set("Print", defaults.printReport,
             "Print a summary to Teko's output stream at each rebuild");
  report.set(
      "Inverse Type", defaults.inverseType,
      "Inverse of single-block groups in the reported Hierarchical Block Gauss-Seidel lists");
  report.set("Merged Inverse Type", defaults.mergedInverseType,
             "Inverse of merged groups in those lists; it receives a blocked operator");
  return pl;
}

}  // namespace

KrylovSurrogatePreconditionerFactory::KrylovSurrogatePreconditionerFactory() {}

KrylovSurrogatePreconditionerFactory::KrylovSurrogatePreconditionerFactory(
    const Teuchos::RCP<Teko::InverseFactory>& invFactory, const Teuchos::RCP<Handle>& handle,
    const Options& options)
    : invFactory_(invFactory), handle_(handle), options_(options) {}

LinearOp KrylovSurrogatePreconditionerFactory::buildPreconditionerOperator(
    LinearOp& lo, PreconditionerState& state) const {
#ifndef TEKO_HAVE_KRYLOV_SURROGATE
  (void)lo;
  (void)state;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Teko \"Krylov Surrogate\" requires Tpetra configured with TSQR "
                             "and Teko built with Belos: configure with "
                             "Trilinos_ENABLE_TpetraTSQR=ON and Teko_ENABLE_Belos=ON");
  return Teuchos::null;
#else
  TEUCHOS_TEST_FOR_EXCEPTION(invFactory_ == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": no \"Inverse Factory\" has been set");
  TEUCHOS_TEST_FOR_EXCEPTION(handle_ == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": no \"Report Handle\" has been set");

  BlockedLinearOp blo = toBlockedLinearOp(lo);
  TEUCHOS_TEST_FOR_EXCEPTION(blo == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\" needs a blocked operator, one block per "
                             "physics field");
  // Every block backed by Tpetra, and the user's orderings valid, checked now
  // rather than at the first report
  const int nb = blockRowCount(blo);
  for (int i = 0; i < nb; ++i)
    for (int j = 0; j < nb; ++j) Impl::tpetraBlock(blo, i, j);  // throws if not Tpetra
  for (const auto& o : options_.orderings)
    TEUCHOS_TEST_FOR_EXCEPTION(!Impl::isValidOrdering(o, nb), std::invalid_argument,
                               "Teko \"Krylov Surrogate\": an ordering in \"Orderings\" is not a "
                               "valid ordering of the operator's "
                                   << nb << " blocks");

  ModifiableLinearOp& recorderOp = state.getModifiableOp("krylovSurrogateOp");
  Teuchos::RCP<RecorderOp> recorder;
  if (recorderOp == Teuchos::null) {
    recorder =
        Teuchos::rcp(new RecorderOp(Teko::buildInverse(*invFactory_, lo), options_.maxSamples));
    recorderOp = recorder;
  } else {
    recorder               = Teuchos::rcp_dynamic_cast<RecorderOp>(recorderOp, true);
    ModifiableLinearOp inv = recorder->getModifiableOp();
    Teko::rebuildInverse(*invFactory_, lo, inv);
    // Report the solves since the previous build before forgetting them. The
    // report comes from this preconditioner's own recorder, through a private
    // handle: the shared one may be attached to another preconditioner built
    // from this factory.
    if (options_.printReport && recorder->numSamples() > 0) {
      auto own = Handle::create();
      own->attach(recorder.getConst(), blo, options_);
      const Report report = own->buildReport();
      // Teko's stream prints on rank 0 only, but a user-supplied one may not
      if (Impl::tpetraBlock(blo, 0, 0)->getRangeMap()->getComm()->getRank() == 0)
        report.print(*Teko::getOutputStream());
    }
    recorder->clearSamples();
  }

  handle_->attach(recorder.getConst(), blo, options_);
  return recorderOp.getConst();
#endif
}

void KrylovSurrogatePreconditionerFactory::initializeFromParameterList(
    const Teuchos::ParameterList& settings) {
  Teuchos::ParameterList pl(settings);
  pl.validateParametersAndSetDefaults(*validParameters());

  const Teuchos::ParameterList& report = pl.sublist("Report");
  options_.printReport                 = report.get<bool>("Print");
  options_.inverseType                 = report.get<std::string>("Inverse Type");
  options_.mergedInverseType           = report.get<std::string>("Merged Inverse Type");

  // Without a handle from the application, the factory keeps its own, used for printing
  handle_ = pl.get<HandlePtr>("Report Handle");
  TEUCHOS_TEST_FOR_EXCEPTION(
      handle_ == Teuchos::null && !options_.printReport, std::runtime_error,
      "Teko \"Krylov Surrogate\" needs a \"Report Handle\" (a "
      "Teuchos::RCP<Teko::KrylovSurrogate::Handle> set from code) or \"Report\" > \"Print\" = "
      "true. Without either, the report could never be seen.");
  if (handle_ == Teuchos::null) handle_ = Handle::create();
  const std::string invName = pl.get<std::string>("Inverse Factory");
  TEUCHOS_TEST_FOR_EXCEPTION(invName.empty(), std::runtime_error,
                             "Parameter \"Inverse Factory\" is required by Teko \"Krylov "
                             "Surrogate\"");

  options_.maxSamples    = pl.get<int>("Maximum Samples");
  options_.truncationTol = pl.get<double>("Truncation Tolerance");

  const Teuchos::ParameterList& search = pl.sublist("Search");
  options_.searchTol                   = search.get<double>("Tolerance");
  options_.candidateCount              = search.get<int>("Candidate Count");
  options_.seed                        = search.get<int>("Seed");
  options_.orderings.clear();
  const Teuchos::ParameterList& orderings = search.sublist("Orderings");
  for (auto it = orderings.begin(); it != orderings.end(); ++it) {
    const std::string& name = orderings.name(it);
    TEUCHOS_TEST_FOR_EXCEPTION(
        !orderings.isType<Teuchos::Array<int> >(name), std::invalid_argument,
        "Teko \"Krylov Surrogate\": ordering \"" << name << "\" must be an Array(int)");
    const Teuchos::Array<int>& o = orderings.get<Teuchos::Array<int> >(name);
    options_.orderings.push_back(std::vector<int>(o.begin(), o.end()));
  }

  invFactory_ = getInverseLibrary()->getInverseFactory(invName);
  TEUCHOS_TEST_FOR_EXCEPTION(invFactory_ == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": \"Inverse Factory\" = \""
                                 << invName << "\" could not be found in the inverse library");
}

Teuchos::RCP<const Teuchos::ParameterList>
KrylovSurrogatePreconditionerFactory::getValidParameters() const {
  return validParameters();
}

Teuchos::RCP<Teuchos::ParameterList> KrylovSurrogatePreconditionerFactory::getRequestedParameters()
    const {
  TEUCHOS_TEST_FOR_EXCEPTION(invFactory_ == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": no \"Inverse Factory\" has been set");
  return invFactory_->getRequestedParameters();
}

bool KrylovSurrogatePreconditionerFactory::updateRequestedParameters(
    const Teuchos::ParameterList& pl) {
  TEUCHOS_TEST_FOR_EXCEPTION(invFactory_ == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": no \"Inverse Factory\" has been set");
  return invFactory_->updateRequestedParameters(pl);
}

}  // namespace KrylovSurrogate
}  // end namespace Teko
