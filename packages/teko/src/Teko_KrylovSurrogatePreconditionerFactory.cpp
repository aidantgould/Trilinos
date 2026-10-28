// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogatePreconditionerFactory.hpp"

#include <string>

#include "Teuchos_Array.hpp"

#include "Teko_InverseLibrary.hpp"
#include "Teko_KrylovSurrogateRecorderOp.hpp"
#include "Teko_KrylovSurrogateReducedOperator.hpp"
#include "Teko_KrylovSurrogateSearch.hpp"

namespace Teko {

namespace {

typedef Teuchos::RCP<KrylovSurrogateHandle> Handle;

// Every option, with its default and documentation. Validating against this
// makes an unknown or mistyped option an error, so a misspelling does not
// silently do nothing.
Teuchos::RCP<const Teuchos::ParameterList> validParameters() {
  const KrylovSurrogateOptions defaults;
  auto pl = Teuchos::rcp(new Teuchos::ParameterList);
  pl->set("Type", std::string("Krylov Surrogate"), "Selects this preconditioner");
  pl->set("Inverse Factory", std::string(""),
          "The Teko block inverse applied, and recorded (required)");
  pl->set("Report Handle", Handle(),
          "The Teko::KrylovSurrogateHandle the report is retrieved from (required, set from code)");
  pl->set("Maximum Samples", defaults.maxSamples,
          "Columns of preconditioner output kept, each a full-length vector of memory");
  pl->set("Truncation Tolerance", defaults.truncationTol,
          "Singular values kept per block: sigma_k >= tolerance * sigma_1");
  Teuchos::ParameterList& search = pl->sublist("Search", false, "Candidate scoring");
  search.set("Tolerance", defaults.searchTol,
             "Relative residual each candidate's surrogate solve stops at");
  search.set("Candidate Count", defaults.candidateCount, "Number of candidates scored");
  search.set("Seed", defaults.seed, "Seed for sampling candidates");
  search.sublist("Orderings", false, "Orderings always scored, each an Array(int)")
      .disableRecursiveValidation();
  return pl;
}

}  // namespace

KrylovSurrogatePreconditionerFactory::KrylovSurrogatePreconditionerFactory() {}

KrylovSurrogatePreconditionerFactory::KrylovSurrogatePreconditionerFactory(
    const Teuchos::RCP<Teko::InverseFactory>& invFactory,
    const Teuchos::RCP<KrylovSurrogateHandle>& handle, const KrylovSurrogateOptions& options)
    : invFactory_(invFactory), handle_(handle), options_(options) {}

LinearOp KrylovSurrogatePreconditionerFactory::buildPreconditionerOperator(
    LinearOp& lo, PreconditionerState& state) const {
#ifndef TEKO_HAVE_KRYLOV_SURROGATE
  (void)lo;
  (void)state;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Teko \"Krylov Surrogate\" requires Tpetra configured with TSQR "
                             "(Trilinos_ENABLE_TpetraTSQR=ON) and Teko built with Belos");
#else
  TEUCHOS_TEST_FOR_EXCEPTION(invFactory_ == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": no \"Inverse Factory\" has been set");
  TEUCHOS_TEST_FOR_EXCEPTION(handle_ == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": no \"Report Handle\" has been set");

  BlockedLinearOp blo = toBlockedLinearOp(lo);
  TEUCHOS_TEST_FOR_EXCEPTION(blo == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\" needs a blocked operator, one block per "
                             "physics field");
  const int nb = blockRowCount(blo);
  for (int i = 0; i < nb; ++i)
    for (int j = 0; j < nb; ++j) KrylovSurrogate::tpetraBlock(blo, i, j);  // throws if not Tpetra
  for (const auto& o : options_.orderings)
    TEUCHOS_TEST_FOR_EXCEPTION(!KrylovSurrogate::isValidOrdering(o, nb), std::invalid_argument,
                               "Teko \"Krylov Surrogate\": an ordering in \"Orderings\" is not a "
                               "valid ordering of the operator's "
                                   << nb << " blocks");

  ModifiableLinearOp& recorderOp = state.getModifiableOp("krylovSurrogateOp");
  Teuchos::RCP<KrylovSurrogateRecorderOp> recorder;
  if (recorderOp == Teuchos::null) {
    recorder = Teuchos::rcp(
        new KrylovSurrogateRecorderOp(Teko::buildInverse(*invFactory_, lo), options_.maxSamples));
    recorderOp = recorder;
  } else {
    recorder               = Teuchos::rcp_dynamic_cast<KrylovSurrogateRecorderOp>(recorderOp, true);
    ModifiableLinearOp inv = recorder->getModifiableOp();
    Teko::rebuildInverse(*invFactory_, lo, inv);
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

  handle_ = pl.get<Handle>("Report Handle");
  TEUCHOS_TEST_FOR_EXCEPTION(
      handle_ == Teuchos::null, std::runtime_error,
      "Teko \"Krylov Surrogate\" requires a \"Report Handle\", a "
      "Teuchos::RCP<Teko::KrylovSurrogateHandle> set from code. Without one the report could "
      "never be retrieved. See Teko::KrylovSurrogateHandle.");
  const std::string invName = pl.get<std::string>("Inverse Factory");
  TEUCHOS_TEST_FOR_EXCEPTION(invName.empty(), std::runtime_error,
                             "Parameter \"Inverse Factory\" is required by Teko \"Krylov "
                             "Surrogate\"");

  options_.maxSamples    = pl.get<int>("Maximum Samples");
  options_.truncationTol = pl.get<double>("Truncation Tolerance");
  TEUCHOS_TEST_FOR_EXCEPTION(options_.maxSamples < 1, std::invalid_argument,
                             "Teko \"Krylov Surrogate\": \"Maximum Samples\" must be at least 1");

  const Teuchos::ParameterList& search = pl.sublist("Search");
  options_.searchTol                   = search.get<double>("Tolerance");
  options_.candidateCount              = search.get<int>("Candidate Count");
  options_.seed                        = search.get<int>("Seed");
  options_.orderings.clear();
  const Teuchos::ParameterList& orderings = search.sublist("Orderings");
  for (auto it = orderings.begin(); it != orderings.end(); ++it) {
    const std::string& name = orderings.name(it);
    TEUCHOS_TEST_FOR_EXCEPTION(
        not orderings.isType<Teuchos::Array<int> >(name), std::invalid_argument,
        "Teko \"Krylov Surrogate\": ordering \"" << name << "\" must be an Array(int)");
    const Teuchos::Array<int>& o = orderings.get<Teuchos::Array<int> >(name);
    options_.orderings.push_back(std::vector<int>(o.begin(), o.end()));
  }

  invFactory_ = getInverseLibrary()->getInverseFactory(invName);
  TEUCHOS_TEST_FOR_EXCEPTION(invFactory_ == Teuchos::null, std::runtime_error,
                             "ERROR: \"Inverse Factory\" = " << invName << " could not be found");
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

}  // end namespace Teko
