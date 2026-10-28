// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogatePreconditionerFactory.hpp"

#include <set>
#include <string>

#include "Teuchos_Array.hpp"

#include "TpetraCore_config.h"

#include "Teko_Config.h"
#include "Teko_InverseLibrary.hpp"
#include "Teko_KrylovSurrogateRecorderOp.hpp"
#include "Teko_KrylovSurrogateReducedOperator.hpp"
#include "Teko_KrylovSurrogateSearch.hpp"

namespace Teko {

namespace {

// An unknown name is an error, so a misspelled option does not silently do nothing
void checkNames(const Teuchos::ParameterList& pl, const std::set<std::string>& valid,
                const std::string& where) {
  for (auto it = pl.begin(); it != pl.end(); ++it) {
    const std::string& name = pl.name(it);
    TEUCHOS_TEST_FOR_EXCEPTION(
        valid.count(name) == 0, std::invalid_argument,
        "Teko \"Krylov Surrogate\": unknown parameter \"" << name << "\" in " << where);
  }
}

}  // namespace

KrylovSurrogatePreconditionerFactory::KrylovSurrogatePreconditionerFactory() {}

KrylovSurrogatePreconditionerFactory::KrylovSurrogatePreconditionerFactory(
    const Teuchos::RCP<Teko::InverseFactory>& invFactory,
    const Teuchos::RCP<KrylovSurrogateHandle>& handle, const KrylovSurrogateOptions& options)
    : invFactory_(invFactory), handle_(handle), options_(options) {}

LinearOp KrylovSurrogatePreconditionerFactory::buildPreconditionerOperator(
    LinearOp& lo, PreconditionerState& state) const {
#ifndef HAVE_TPETRA_TSQR
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Teko \"Krylov Surrogate\" requires Tpetra configured with TSQR "
                             "(Trilinos_ENABLE_TpetraTSQR=ON)");
#endif
#ifndef Teko_ENABLE_Belos
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Teko \"Krylov Surrogate\" requires Teko built with Belos");
#endif
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
}

void KrylovSurrogatePreconditionerFactory::initializeFromParameterList(
    const Teuchos::ParameterList& settings) {
  typedef Teuchos::RCP<KrylovSurrogateHandle> Handle;

  checkNames(settings,
             {"Type", "Inverse Factory", "Report Handle", "Maximum Samples", "Truncation Tolerance",
              "Search"},
             "the \"Krylov Surrogate\" sublist");
  TEUCHOS_TEST_FOR_EXCEPTION(not settings.isParameter("Inverse Factory"), std::runtime_error,
                             "Parameter \"Inverse Factory\" is required by Teko \"Krylov "
                             "Surrogate\"");
  TEUCHOS_TEST_FOR_EXCEPTION(
      not settings.isType<Handle>("Report Handle"), std::runtime_error,
      "Teko \"Krylov Surrogate\" requires a \"Report Handle\", a "
      "Teuchos::RCP<Teko::KrylovSurrogateHandle> set from code. Without one the report could "
      "never be retrieved. See Teko::KrylovSurrogateHandle.");

  handle_ = settings.get<Handle>("Report Handle");

  options_ = KrylovSurrogateOptions();
  if (settings.isParameter("Maximum Samples"))
    options_.maxSamples = settings.get<int>("Maximum Samples");
  if (settings.isParameter("Truncation Tolerance"))
    options_.truncationTol = settings.get<double>("Truncation Tolerance");
  TEUCHOS_TEST_FOR_EXCEPTION(options_.maxSamples < 1, std::invalid_argument,
                             "Teko \"Krylov Surrogate\": \"Maximum Samples\" must be at least 1");

  if (settings.isSublist("Search")) {
    const Teuchos::ParameterList& search = settings.sublist("Search");
    checkNames(search, {"Tolerance", "Candidate Count", "Seed", "Orderings"},
               "the \"Search\" sublist");
    if (search.isParameter("Tolerance")) options_.searchTol = search.get<double>("Tolerance");
    if (search.isParameter("Candidate Count"))
      options_.candidateCount = search.get<int>("Candidate Count");
    if (search.isParameter("Seed")) options_.seed = search.get<int>("Seed");
    if (search.isSublist("Orderings")) {
      const Teuchos::ParameterList& orderings = search.sublist("Orderings");
      for (auto it = orderings.begin(); it != orderings.end(); ++it) {
        const std::string& name = orderings.name(it);
        TEUCHOS_TEST_FOR_EXCEPTION(
            not orderings.isType<Teuchos::Array<int> >(name), std::invalid_argument,
            "Teko \"Krylov Surrogate\": ordering \"" << name << "\" must be an Array(int)");
        const Teuchos::Array<int>& o = orderings.get<Teuchos::Array<int> >(name);
        options_.orderings.push_back(std::vector<int>(o.begin(), o.end()));
      }
    }
  }

  const std::string invName = settings.get<std::string>("Inverse Factory");
  invFactory_               = getInverseLibrary()->getInverseFactory(invName);
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
