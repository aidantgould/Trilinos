// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file Teko_KrylovSurrogateSearch.hpp
 *
 * Scores candidate block configurations on the Krylov surrogate.
 */

#ifndef __Teko_KrylovSurrogateSearch_hpp__
#define __Teko_KrylovSurrogateSearch_hpp__

#include <random>
#include <vector>

#include "Teuchos_Comm.hpp"

#include "Teko_KrylovSurrogateReducedOperator.hpp"
#include "Teko_KrylovSurrogateReport.hpp"

namespace Teko {
namespace KrylovSurrogate {

/** \brief Number of orderings of n blocks: ordered set partitions, the Fubini
 *         numbers 1, 1, 3, 13, 75, 541, ... (a double, since it grows fast).
 */
double numConfigurations(int n);

/** \brief Every ordering of n blocks. ordering[k] is the position of block k's
 *         group; positions run 0 to (number of groups - 1).
 */
std::vector<std::vector<int> > allConfigurations(int n);

/** \brief One ordering of n blocks drawn uniformly from all of them, group order
 *         included.
 */
std::vector<int> randomConfiguration(int n, std::mt19937_64& rng);

//! Whether ordering is a valid ordering of n blocks
bool isValidOrdering(const std::vector<int>& ordering, int n);

/** \brief The candidates to score: the user's orderings, the identity and
 *         all-merged extremes, then every other configuration if there are at
 *         most count in all, or else a seeded uniform sample up to count.
 */
std::vector<std::vector<int> > buildCandidates(int n,
                                               const std::vector<std::vector<int> >& userOrderings,
                                               int count, int seed);

/** \brief \f$[C]_{S_1}\f$ for lower block Gauss-Seidel in the ordering's group
 *         order: block (i, j) is kept when block i's group comes at or after
 *         block j's, and zeroed otherwise.
 */
DenseMatrix gaussSeidelSelection(const DenseMatrix& C, const std::vector<int>& ranks,
                                 const std::vector<int>& ordering);

/** \brief Score every candidate on the surrogate and return them sorted:
 *         converged first, then fewest iterations, then best rate.
 *
 * Each candidate is scored by forming \f$K = C [C]_{S_1}^{-1}\f$ explicitly and
 * running unpreconditioned Belos GMRES on \f$K y = b\f$ from zero. Candidates are
 * dealt round-robin over the ranks of comm and gathered, so the result is
 * bitwise identical on every rank. Collective.
 */
std::vector<KrylovSurrogateReport::Candidate> scoreCandidates(
    const DenseMatrix& C, const DenseVector& b, const std::vector<int>& ranks,
    const std::vector<std::vector<int> >& candidates, double tol, const Teuchos::Comm<int>& comm);

}  // namespace KrylovSurrogate
}  // end namespace Teko

#endif
