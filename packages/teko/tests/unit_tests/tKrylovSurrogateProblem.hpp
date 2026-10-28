// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

// A small blocked test problem for the Krylov surrogate tests. Every entry is
// a formula of its global indices, so any rank can rebuild the whole problem
// densely as a reference, whatever the distribution.

#ifndef __tKrylovSurrogateProblem_hpp__
#define __tKrylovSurrogateProblem_hpp__

#include <cmath>
#include <vector>

#include "Teuchos_SerialDenseMatrix.hpp"

#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"

#include "Thyra_DefaultBlockedLinearOp.hpp"
#include "Thyra_DefaultZeroLinearOp.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

#include "Teko_Utilities.hpp"

namespace KrylovSurrogateTest {

typedef Teko::ST ST;
typedef Teko::LO LO;
typedef Teko::GO GO;
typedef Teko::NT NT;
typedef Tpetra::Map<LO, GO, NT> Map;
typedef Tpetra::CrsMatrix<ST, LO, GO, NT> CrsMatrix;
typedef Teuchos::SerialDenseMatrix<int, double> Dense;

// Block (i, j) entry at (r, c), or 0. Diagonal blocks are diagonally dominant;
// block (0, 2) is structurally zero.
inline double entry(int i, int j, int r, int c, int ni, int nj) {
  if (i == j) {
    if (r == c) return 4.0 + 0.1 * i + 0.01 * r;
    if (c == r - 1 || c == r + 1) return -1.0;
    if (c == (r + 5) % nj) return 0.3;
    return 0.0;
  }
  if (i == 0 && j == 2) return 0.0;
  if (c == (3 * r + i + 2 * j) % nj) return 0.6 * std::sin(r + 2.0 * i + 3.0 * j + 1.0);
  (void)ni;
  return 0.0;
}

inline bool zeroBlock(int i, int j) { return i == 0 && j == 2; }

// How block j's rows are distributed
enum Distribution { EVEN, ALL_ON_ONE_RANK };

inline Teuchos::RCP<const Map> blockMap(int n, Distribution d) {
  auto comm = Tpetra::getDefaultComm();
  if (d == EVEN) return Teuchos::rcp(new Map(n, 0, comm));
  const int owner = std::min(2, comm->getSize() - 1);
  return Teuchos::rcp(new Map(n, comm->getRank() == owner ? n : 0, 0, comm));
}

// The blocked operator, block j's rows distributed as distributions[j]
inline Teko::BlockedLinearOp buildBlockedOp(const std::vector<int>& sizes,
                                            const std::vector<Distribution>& distributions) {
  const int nb = static_cast<int>(sizes.size());
  std::vector<Teuchos::RCP<const Map> > maps(nb);
  std::vector<Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > > spaces(nb);
  for (int j = 0; j < nb; ++j) {
    maps[j]   = blockMap(sizes[j], distributions[j]);
    spaces[j] = Thyra::tpetraVectorSpace<ST, LO, GO, NT>(maps[j]);
  }

  auto op = Thyra::defaultBlockedLinearOp<ST>();
  op->beginBlockFill(nb, nb);
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < nb; ++j) {
      if (zeroBlock(i, j)) {
        op->setBlock(i, j, Thyra::zero<ST>(spaces[i], spaces[j]));
        continue;
      }
      auto A = Teuchos::rcp(new CrsMatrix(maps[i], 5));
      for (size_t lr = 0; lr < maps[i]->getLocalNumElements(); ++lr) {
        const GO r = maps[i]->getGlobalElement(lr);
        std::vector<GO> cols;
        std::vector<ST> vals;
        for (int c = 0; c < sizes[j]; ++c) {
          const double v = entry(i, j, static_cast<int>(r), c, sizes[i], sizes[j]);
          if (v != 0.0) {
            cols.push_back(c);
            vals.push_back(v);
          }
        }
        if (!cols.empty())
          A->insertGlobalValues(r, Teuchos::ArrayView<const GO>(cols),
                                Teuchos::ArrayView<const ST>(vals));
      }
      A->fillComplete(maps[j], maps[i]);
      op->setBlock(i, j, Thyra::tpetraLinearOp<ST, LO, GO, NT>(spaces[i], spaces[j], A));
    }
  }
  op->endBlockFill();
  return op;
}

// Fill a blocked multivector entry by entry from f(block, global row, column)
template <class F>
void fillBlocked(const Teko::MultiVector& mv, F f) {
  Teko::BlockedMultiVector bmv = Teko::toBlockedMultiVector(mv);
  for (int j = 0; j < bmv->productSpace()->numBlocks(); ++j) {
    auto t = Thyra::TpetraOperatorVectorExtraction<ST, LO, GO, NT>::getTpetraMultiVector(
        Teko::getBlock(j, bmv));
    auto map = t->getMap();
    for (size_t c = 0; c < t->getNumVectors(); ++c) {
      auto data = t->getDataNonConst(c);
      for (size_t lr = 0; lr < map->getLocalNumElements(); ++lr)
        data[lr] = f(j, static_cast<int>(map->getGlobalElement(lr)), static_cast<int>(c));
    }
  }
}

// Block (i, j) as a dense matrix
inline Dense denseBlock(const std::vector<int>& sizes, int i, int j) {
  Dense A(sizes[i], sizes[j]);
  for (int r = 0; r < sizes[i]; ++r)
    for (int c = 0; c < sizes[j]; ++c) A(r, c) = entry(i, j, r, c, sizes[i], sizes[j]);
  return A;
}

}  // namespace KrylovSurrogateTest

#endif
