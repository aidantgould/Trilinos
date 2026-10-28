// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef __Teko_KrylovSurrogatePreconditionerFactory_hpp__
#define __Teko_KrylovSurrogatePreconditionerFactory_hpp__

#include "Teko_PreconditionerFactory.hpp"
#include "Teko_InverseFactory.hpp"
#include "Teko_KrylovSurrogateReport.hpp"

namespace Teko {

/** \brief Wraps any Teko block inverse, unchanged, and records what a Krylov
 *         surrogate of the preconditioned operator needs.
 *
 * After a right-preconditioned flexible GMRES solve with this preconditioner,
 * the handle from the parameter list returns a KrylovSurrogateReport: the
 * reduced operator \f$\hat{C}\f$, the reduced right-hand side \f$\hat{b}\f$, and how
 * each of a set of alternative lower block Gauss-Seidel configurations
 * converges on the surrogate. The operator must be blocked, with every nonzero
 * block backed by Tpetra, and Teko must be built with Belos and Tpetra with
 * TSQR.
 *
 \code
    auto handle = Teko::KrylovSurrogateHandle::create();
    params.sublist("Surrogate GS").set("Report Handle", handle);
 \endcode
 *
 \code
     <ParameterList name="Surrogate GS">
       <Parameter name="Type" type="string" value="Krylov Surrogate"/>
       <Parameter name="Inverse Factory" type="string" value="<Some Block Inverse>"/>
       <!-- Optional. Columns of preconditioner output kept, each one full-length
            vector of memory. Default 100. -->
       <Parameter name="Maximum Samples" type="int" value="100"/>
       <!-- Optional. Singular values kept per block, relative. Default 1e-10. -->
       <Parameter name="Truncation Tolerance" type="double" value="1e-10"/>
       <ParameterList name="Search">                       <!-- optional -->
         <Parameter name="Tolerance" type="double" value="1e-8"/>
         <Parameter name="Candidate Count" type="int" value="100"/>
         <Parameter name="Seed" type="int" value="0"/>
         <ParameterList name="Orderings">                 <!-- always scored -->
           <Parameter name="merge 0 and 1" type="Array(int)" value="{0,0,1}"/>
         </ParameterList>
       </ParameterList>
     </ParameterList>
 \endcode
 *
 * The "Report Handle" (a Teuchos::RCP<Teko::KrylovSurrogateHandle>) is
 * required, and is set from code: see KrylovSurrogateHandle.
 */
class KrylovSurrogatePreconditionerFactory : public virtual Teko::PreconditionerFactory {
 public:
  //! Default constructor, for use with the AutoClone class.
  KrylovSurrogatePreconditionerFactory();

  KrylovSurrogatePreconditionerFactory(const Teuchos::RCP<Teko::InverseFactory>& invFactory,
                                       const Teuchos::RCP<KrylovSurrogateHandle>& handle,
                                       const KrylovSurrogateOptions& options);

  virtual LinearOp buildPreconditionerOperator(LinearOp& lo, PreconditionerState& state) const;

  virtual void initializeFromParameterList(const Teuchos::ParameterList& settings);

  virtual Teuchos::RCP<Teuchos::ParameterList> getRequestedParameters() const;

  virtual bool updateRequestedParameters(const Teuchos::ParameterList& pl);

 private:
  Teuchos::RCP<Teko::InverseFactory> invFactory_;
  Teuchos::RCP<KrylovSurrogateHandle> handle_;
  KrylovSurrogateOptions options_;
};

}  // end namespace Teko

#endif
