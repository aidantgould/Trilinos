// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file Teko_KrylovSurrogateRecorderOp.hpp
 *
 * A linear operator that applies a preconditioner and records its outputs, the
 * samples a Krylov surrogate is built from.
 */

#ifndef __Teko_KrylovSurrogateRecorderOp_hpp__
#define __Teko_KrylovSurrogateRecorderOp_hpp__

#include "Teko_Utilities.hpp"
#include "Teko_ImplicitLinearOp.hpp"

namespace Teko {
namespace KrylovSurrogate {

/** \brief Applies a preconditioner \f$M_0^{-1}\f$ unchanged, keeping a copy of
 *         each output \f$z = M_0^{-1} v\f$ and of the first input.
 *
 * Every output is a valid sample of the diagonal-block inverses: with
 * \f$\tilde{V}_j := A_{jj} Z_j\f$, \f$A_{jj}^{-1} \tilde{V}_j = Z_j\f$ holds exactly
 * whatever produced \f$Z\f$. Recording stops once maxSamples columns are kept.
 * The store starts small and doubles up to maxSamples as samples arrive.
 *
 * The first input after a clear becomes the surrogate's right-hand side. That
 * relies on how Belos's GMRES applies a right preconditioner:
 * - the first apply is on \f$v_1 = r_0 / \|r_0\|\f$;
 * - with a block size above one, only column 0 is kept, so the right-hand side is
 *   the first system's;
 * - a non-flexible GMRES also applies the preconditioner to \f$V y\f$ at each
 *   restart. Those outputs are recorded too: they are valid samples of the
 *   diagonal-block inverses, though not Krylov vectors.
 *
 * The recorded state is mutable and written in a const apply, so the operator is
 * not thread-safe, as Thyra operators are generally not.
 */
class RecorderOp : public ImplicitLinearOp {
 public:
  RecorderOp(const ModifiableLinearOp& prec, int maxSamples);

  //! @name Inherited methods from Thyra::LinearOpBase
  //@{
  virtual VectorSpace range() const { return prec_->range(); }
  virtual VectorSpace domain() const { return prec_->domain(); }

  /** \brief \f$y = \alpha M_0^{-1} x + \beta y\f$, recording \f$M_0^{-1} x\f$. */
  virtual void implicitApply(const MultiVector& x, MultiVector& y, const double alpha = 1.0,
                             const double beta = 0.0) const;
  //@}

  //! The wrapped preconditioner, for rebuilding it in place
  ModifiableLinearOp getModifiableOp() const { return prec_; }

  //! Forget every recorded sample and the first input. The store stays
  //! allocated, for the next solve after a rebuild.
  void clearSamples();

  //! Number of recorded columns
  int numSamples() const { return count_; }

  //! The recorded columns (null if there are none), \f$Z\f$ in the range space
  MultiVector getSamples() const;

  //! The first input after the last clear (null if none), one column
  MultiVector getFirstInput() const { return firstInput_; }

  virtual void describe(Teuchos::FancyOStream& out_arg,
                        const Teuchos::EVerbosityLevel verbLevel) const;

 private:
  void record(const MultiVector& z) const;

  static constexpr int kInitialCapacity = 16;

  ModifiableLinearOp prec_;
  int maxSamples_;

  mutable MultiVector store_;
  mutable MultiVector firstInput_;
  mutable int count_;
};

}  // namespace KrylovSurrogate
}  // end namespace Teko

#endif
