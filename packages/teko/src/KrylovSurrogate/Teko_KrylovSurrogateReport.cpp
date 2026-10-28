// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogateReport.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "Teuchos_Array.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_TwoDArray.hpp"

#include "Tpetra_Map.hpp"

#include "Teko_KrylovSurrogateRecorderOp.hpp"
#include "Teko_KrylovSurrogateReducedOperator.hpp"
#include "Teko_KrylovSurrogateSearch.hpp"

namespace Teko {
namespace KrylovSurrogate {

namespace {

std::string xmlEscaped(const std::string& text) {
  std::string out;
  for (char c : text) {
    if (c == '<')
      out += "&lt;";
    else if (c == '>')
      out += "&gt;";
    else if (c == '&')
      out += "&amp;";
    else if (c == '"')
      out += "&quot;";
    else
      out += c;
  }
  return out;
}

// A Hierarchical Block Gauss-Seidel list as the XML people write by hand, without
// the bookkeeping attributes Teuchos's writer adds. It holds only strings, bools
// and sublists.
void writeXml(const Teuchos::ParameterList& pl, const std::string& name, const std::string& indent,
              std::ostream& os) {
  os << indent << "<ParameterList name=\"" << xmlEscaped(name) << "\">\n";
  for (auto it = pl.begin(); it != pl.end(); ++it) {
    const std::string& key = pl.name(it);
    if (pl.isSublist(key))
      writeXml(pl.sublist(key), key, indent + "  ", os);
    else if (pl.isType<bool>(key))
      os << indent << "  <Parameter name=\"" << xmlEscaped(key) << "\" type=\"bool\" value=\""
         << (pl.get<bool>(key) ? "true" : "false") << "\"/>\n";
    else
      os << indent << "  <Parameter name=\"" << xmlEscaped(key) << "\" type=\"string\" value=\""
         << xmlEscaped(pl.get<std::string>(key)) << "\"/>\n";
  }
  os << indent << "</ParameterList>\n";
}

}  // namespace

Teuchos::ParameterList asHierarchicalGaussSeidel(const std::vector<int>& ordering,
                                                 const std::string& inverseType,
                                                 const std::string& mergedInverseType) {
  Teuchos::ParameterList pl;
  pl.set("Type", std::string("Hierarchical Block Gauss-Seidel"));
  pl.set("Use Upper Triangle", false);

  const int groups = ordering.empty() ? 0 : *std::max_element(ordering.begin(), ordering.end()) + 1;
  for (int g = 0; g < groups; ++g) {
    std::ostringstream blocks;
    int members = 0;
    for (std::size_t k = 0; k < ordering.size(); ++k)
      if (ordering[k] == g) blocks << (members++ ? "," : "") << k + 1;

    Teuchos::ParameterList& group = pl.sublist("Hierarchical Block " + std::to_string(g + 1));
    group.set("Included Subblocks", blocks.str());
    if (members == 1)
      group.set("Inverse Type", inverseType.empty() ? "<Inverse Type>" : inverseType);
    else
      group.set("Inverse Type",
                mergedInverseType.empty() ? "<Merged Inverse Type>" : mergedInverseType);
  }
  return pl;
}

Teuchos::RCP<Teuchos::ParameterList> Report::toParameterList() const {
  auto pl = Teuchos::rcp(new Teuchos::ParameterList("Krylov Surrogate Report"));
  pl->set("Number of Blocks", numBlocks);
  pl->set("Number of Samples", numSamples);
  pl->set("Block Sizes", Teuchos::Array<int>(blockSizes));
  pl->set("Basis Sizes", Teuchos::Array<int>(basisSizes));

  Teuchos::TwoDArray<double> C(CHat.numRows(), CHat.numCols());
  for (int i = 0; i < CHat.numRows(); ++i)
    for (int j = 0; j < CHat.numCols(); ++j) C(i, j) = CHat(i, j);
  pl->set("C_hat", C);
  pl->set("b_hat", Teuchos::Array<double>(bHat.values(), bHat.values() + bHat.length()));

  pl->set("RHS Capture", rhsCapture);
  pl->set("Diagonal Deviations", Teuchos::Array<double>(diagonalDeviations));
  pl->set("Search Ranks", searchRanks);

  Teuchos::ParameterList& cl = pl->sublist("Candidates");
  for (std::size_t c = 0; c < candidates.size(); ++c) {
    Teuchos::ParameterList& one = cl.sublist("Candidate " + std::to_string(c));
    one.set("Ordering", Teuchos::Array<int>(candidates[c].ordering));
    one.set("Status", candidates[c].status);
    one.set("Iterations", candidates[c].iterations);
    one.set("Final Relative Residual", candidates[c].finalRelativeResidual);
    one.set("Rate", candidates[c].rate);
    one.set("As Hierarchical Block Gauss-Seidel", candidates[c].asHierarchicalGaussSeidel);
  }
  return pl;
}

void Report::print(std::ostream& os, int maxCandidates) const {
  auto list = [](const std::vector<int>& v) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < v.size(); ++i) ss << (i ? " " : "") << v[i];
    return ss.str();
  };
  const double deviation =
      diagonalDeviations.empty()
          ? 0.0
          : *std::max_element(diagonalDeviations.begin(), diagonalDeviations.end());

  os << "Teko Krylov Surrogate report: " << numBlocks << " blocks (sizes " << list(blockSizes)
     << "), " << numSamples << " samples, basis sizes " << list(basisSizes) << "\n"
     << "  rhs capture " << rhsCapture << ", max diagonal deviation " << deviation << ", "
     << candidates.size() << " candidates scored on " << searchRanks << " ranks\n"
     << "  " << std::left << std::setw(20) << "ordering" << std::setw(22) << "status"
     << std::setw(12) << "iterations"
     << "rate\n";
  const int shown = std::min<int>(maxCandidates, candidates.size());
  for (int c = 0; c < shown; ++c)
    os << "  " << std::setw(20) << "[" + list(candidates[c].ordering) + "]" << std::setw(22)
       << candidates[c].status << std::setw(12) << candidates[c].iterations << candidates[c].rate
       << "\n";
  os << std::right;
  if (!candidates.empty()) {
    os << "  best, as a Teko inverse library entry:\n";
    writeXml(candidates[0].asHierarchicalGaussSeidel, "Krylov Surrogate Best", "  ", os);
  }
}

Teuchos::RCP<Handle> Handle::create() { return Teuchos::rcp(new Handle); }

void Handle::attach(const Teuchos::RCP<const RecorderOp>& recorder, const BlockedLinearOp& A,
                    const Options& options) {
  recorder_ = recorder;
  A_        = A;
  options_  = options;
}

Report Handle::buildReport() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !isAttached(), std::runtime_error,
      "Teko::KrylovSurrogate::Handle::buildReport: no preconditioner has been built from the "
      "\"Krylov Surrogate\" sublist holding this handle. Build it, solve, then ask for the "
      "report.");
  TEUCHOS_TEST_FOR_EXCEPTION(
      recorder_->numSamples() == 0, std::runtime_error,
      "Teko::KrylovSurrogate::Handle::buildReport: the preconditioner has not been applied since "
      "it was last built, so there are no samples. Solve first.");

#ifdef TEKO_HAVE_KRYLOV_SURROGATE
  static const auto buildTimer =
      Teuchos::TimeMonitor::getNewCounter("Teko Krylov Surrogate: Build Surrogate");
  static const auto searchTimer =
      Teuchos::TimeMonitor::getNewCounter("Teko Krylov Surrogate: Search");

  Report report;
  report.numSamples = recorder_->numSamples();
  {
    Teuchos::TimeMonitor monitor(*buildTimer);
    buildReducedOperator(A_, recorder_->getSamples(), recorder_->getFirstInput(),
                         options_.truncationTol, report);
  }

  auto comm = tpetraBlock(A_, 0, 0)->getRangeMap()->getComm();
  {
    Teuchos::TimeMonitor monitor(*searchTimer);
    report.candidates = scoreCandidates(report.CHat, report.bHat, report.basisSizes,
                                        buildCandidates(report.numBlocks, options_.orderings,
                                                        options_.candidateCount, options_.seed),
                                        options_.searchTol, *comm);
  }
  for (auto& c : report.candidates)
    c.asHierarchicalGaussSeidel =
        asHierarchicalGaussSeidel(c.ordering, options_.inverseType, options_.mergedInverseType);
  report.searchRanks = comm->getSize();
  return report;
#else
  // unreachable: without the surrogate the factory throws before attaching
  return Report();
#endif
}

}  // namespace KrylovSurrogate
}  // end namespace Teko
