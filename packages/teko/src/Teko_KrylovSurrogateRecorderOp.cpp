// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogateRecorderOp.hpp"

#include <algorithm>

#include "Thyra_MultiVectorStdOps.hpp"

namespace Teko {

KrylovSurrogateRecorderOp::KrylovSurrogateRecorderOp(const ModifiableLinearOp& prec, int maxSamples)
    : prec_(prec), maxSamples_(maxSamples), count_(0) {
  TEUCHOS_TEST_FOR_EXCEPTION(prec_ == Teuchos::null, std::invalid_argument,
                             "Teko::KrylovSurrogateRecorderOp: the preconditioner is null");
  TEUCHOS_TEST_FOR_EXCEPTION(maxSamples_ < 1, std::invalid_argument,
                             "Teko::KrylovSurrogateRecorderOp: \"Maximum Samples\" must be at "
                             "least 1, got "
                                 << maxSamples_);
}

void KrylovSurrogateRecorderOp::implicitApply(const MultiVector& x, MultiVector& y,
                                              const double alpha, const double beta) const {
  // x arrives const-only, so take a const view of it
  const Teuchos::RCP<const Thyra::MultiVectorBase<double> > cx = x;
  if (firstInput_ == Teuchos::null && cx->domain()->dim() > 0)
    firstInput_ = cx->subView(Teuchos::Range1D(0, 0))->clone_mv();

  // Belos applies with alpha = 1 and beta = 0, so the output is the sample
  // itself. Anything else is applied into a temporary first, so that what is
  // recorded is M0^{-1} x and not the scaled update.
  if (alpha == 1.0 && beta == 0.0) {
    prec_->apply(Thyra::NOTRANS, *x, y.ptr(), 1.0, 0.0);
    record(y);
  } else {
    MultiVector z = Thyra::createMembers(prec_->range(), x->domain()->dim());
    prec_->apply(Thyra::NOTRANS, *x, z.ptr(), 1.0, 0.0);
    record(z);
    if (beta == 0.0)
      Thyra::assign(y.ptr(), 0.0);
    else
      Thyra::scale(beta, y.ptr());
    Thyra::update(alpha, *z, y.ptr());
  }
}

void KrylovSurrogateRecorderOp::record(const MultiVector& z) const {
  const int room = maxSamples_ - count_;
  const int cols = std::min(room, static_cast<int>(z->domain()->dim()));
  if (cols <= 0) return;

  if (store_ == Teuchos::null) store_ = Thyra::createMembers(prec_->range(), maxSamples_);

  const Teuchos::RCP<const Thyra::MultiVectorBase<double> > cz = z;
  Thyra::assign(store_->subView(Teuchos::Range1D(count_, count_ + cols - 1)).ptr(),
                *cz->subView(Teuchos::Range1D(0, cols - 1)));
  count_ += cols;
}

void KrylovSurrogateRecorderOp::clearSamples() {
  count_      = 0;
  firstInput_ = Teuchos::null;
}

MultiVector KrylovSurrogateRecorderOp::getSamples() const {
  if (count_ == 0) return Teuchos::null;
  return store_->subView(Teuchos::Range1D(0, count_ - 1));
}

void KrylovSurrogateRecorderOp::describe(Teuchos::FancyOStream& out_arg,
                                         const Teuchos::EVerbosityLevel verbLevel) const {
  out_arg << "Teko::KrylovSurrogateRecorderOp: " << count_ << " of " << maxSamples_
          << " samples recorded, wrapping:\n";
  prec_->describe(out_arg, verbLevel);
}

}  // end namespace Teko
