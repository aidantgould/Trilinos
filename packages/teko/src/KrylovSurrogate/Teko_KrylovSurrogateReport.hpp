// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file Teko_KrylovSurrogateReport.hpp
 *
 * The Krylov surrogate report, and the handle a user passes in the parameter
 * list of a "Krylov Surrogate" inverse to get it back after the solve.
 */

#ifndef __Teko_KrylovSurrogateReport_hpp__
#define __Teko_KrylovSurrogateReport_hpp__

#include <ostream>
#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"

#include "TpetraCore_config.h"

#include "Teko_Config.h"
#include "Teko_Utilities.hpp"

// The Krylov surrogate needs Tpetra's TSQR for its bases and Belos for its
// search. Without either, "Krylov Surrogate" is still registered, but building
// it throws, saying what is missing.
#if defined(HAVE_TPETRA_TSQR) && defined(Teko_ENABLE_Belos)
#define TEKO_HAVE_KRYLOV_SURROGATE
#endif

namespace Teko {
namespace KrylovSurrogate {

class RecorderOp;

/** \brief Options of a "Krylov Surrogate" inverse, read from its parameter list.
 */
struct Options {
  //! Columns of preconditioner output kept for the surrogate
  int maxSamples = 100;
  //! Singular values kept per block: sigma_k >= truncationTol * sigma_1
  double truncationTol = 1e-10;
  //! Relative residual the surrogate GMRES solve of each candidate stops at
  double searchTol = 1e-8;
  //! Number of candidate configurations scored
  int candidateCount = 100;
  //! Seed for sampling candidates when they cannot all be enumerated
  int seed = 0;
  //! Orderings the user asked for, always scored
  std::vector<std::vector<int> > orderings;
  //! Print a summary to Teko's output stream at each rebuild
  bool printReport = false;
  //! Inverses named in the candidates' Hierarchical Block Gauss-Seidel lists:
  //! for single-block groups, and for merged groups (which get a blocked operator)
  std::string inverseType, mergedInverseType;
};

/** \brief An ordering as a Teko "Hierarchical Block Gauss-Seidel" parameter list,
 *         ready to paste into an inverse library.
 *
 * Group g becomes "Hierarchical Block g+1", holding its blocks (numbered from 1)
 * and an inverse: inverseType for a single block, mergedInverseType for a merged
 * group. The sweep is lower triangular, as the surrogate scored it. An empty
 * name is written as a placeholder to fill in.
 */
Teuchos::ParameterList asHierarchicalGaussSeidel(const std::vector<int>& ordering,
                                                 const std::string& inverseType,
                                                 const std::string& mergedInverseType);

/** \brief What one solve's Krylov data predicts about alternative block
 *        configurations.
 *
 * The reduced operator \f$\hat{C}\f$ is \f$R \times R\f$ with \f$R = \sum_j r_j\f$,
 * block \f$j\f$ at offset \f$r_0 + \dots + r_{j-1}\f$. Every candidate is a lower
 * block Gauss-Seidel configuration: <code>ordering[k]</code> is the position of
 * block <code>k</code>'s group, and blocks sharing a position are solved together.
 */
struct Report {
  struct Candidate {
    std::vector<int> ordering;
    //! "converged", "max iterations", "singular group block" or "solver error"
    std::string status;
    int iterations = 0;
    //! \f$\|\hat{b} - K y\| / \|\hat{b}\|\f$ at the end of the surrogate solve
    double finalRelativeResidual = 0.0;
    //! finalRelativeResidual^(1/iterations), the average reduction per iteration
    double rate = 0.0;
    //! The same configuration as a Teko inverse library entry
    Teuchos::ParameterList asHierarchicalGaussSeidel;
  };

  int numBlocks  = 0;
  int numSamples = 0;
  std::vector<int> blockSizes;
  //! Directions retained per block, \f$r_j\f$
  std::vector<int> basisSizes;
  Teuchos::SerialDenseMatrix<int, double> CHat;
  //! Projection of the solve's initial residual direction
  Teuchos::SerialDenseVector<int, double> bHat;
  //! \f$\|\hat{b}\|\f$ over the norm of the initial residual direction, at most 1
  double rhsCapture = 0.0;
  //! Per block, \f$\max|\hat{C}_{jj} - I|\f$ as computed (it is exactly I in theory)
  std::vector<double> diagonalDeviations;
  //! Converged first, then fewest iterations, then best rate
  std::vector<Candidate> candidates;
  //! MPI ranks the candidates were dealt over
  int searchRanks = 1;

  //! The whole report as a parameter list, for logging or output
  Teuchos::RCP<Teuchos::ParameterList> toParameterList() const;

  //! A summary: the surrogate, the best candidates, and the best as XML
  void print(std::ostream& os, int maxCandidates = 10) const;
};

/** \brief Collects the surrogate report of a "Krylov Surrogate" inverse.
 *
 * Create one, put it in the inverse's parameter list as "Report Handle", build
 * the preconditioner by any route (Teko::buildInverse, a Teko preconditioner
 * handed to Belos, or Stratimikos with Teko as the preconditioner), solve with
 * right-preconditioned flexible GMRES, then call buildReport(). An application
 * configured only from XML can use "Report" > "Print" instead.
 *
 \code
    auto handle = Teko::KrylovSurrogate::Handle::create();
    inverseLibraryParams.sublist("Surrogate GS").set("Report Handle", handle);
    // ... build and solve ...
    Teko::KrylovSurrogate::Report report = handle->buildReport();
 \endcode
 *
 * The handle refers to the preconditioner most recently built or rebuilt from
 * that sublist, and keeps it and its operator alive. The report describes the
 * solves since that build. Its right-hand side is the direction of the first
 * vector the preconditioner was applied to, which in right-preconditioned
 * GMRES is the initial residual, so warm-started solves are handled as such.
 */
class Handle {
 public:
  static Teuchos::RCP<Handle> create();

  /** \brief Build the report from what has been recorded so far.
   *
   * Collective: every rank of the operator's communicator must call it, and
   * every rank gets the same report.
   */
  Report buildReport() const;

  //! Called by the factory on every build; not for users
  void attach(const Teuchos::RCP<const RecorderOp>& recorder, const BlockedLinearOp& A,
              const Options& options);

  //! Whether a preconditioner has been attached
  bool isAttached() const { return recorder_ != Teuchos::null; }

  //! The recorder of the attached preconditioner (null if none), mainly for testing
  Teuchos::RCP<const RecorderOp> getRecorder() const { return recorder_; }

 private:
  Handle() = default;

  Teuchos::RCP<const RecorderOp> recorder_;
  BlockedLinearOp A_;
  Options options_;
};

}  // namespace KrylovSurrogate
}  // end namespace Teko

#endif
