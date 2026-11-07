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
namespace KrylovSurrogate {

/** \brief Wraps any Teko block inverse, unchanged, and records what a Krylov
 *         surrogate of the preconditioned operator needs.
 *
 * After a right-preconditioned flexible GMRES solve with this preconditioner,
 * the handle from the parameter list returns a Report: the
 * reduced operator \f$\hat{C}\f$, the reduced right-hand side \f$\hat{b}\f$, and how
 * each of a set of alternative lower block Gauss-Seidel configurations
 * converges on the surrogate. The operator must be blocked, with every nonzero
 * block backed by Tpetra (a Thyra::TpetraLinearOp: a scaled or transposed
 * wrapper around one is rejected), and Teko must be built with Belos and
 * Tpetra with TSQR.
 *
 \code
    auto handle = Teko::KrylovSurrogate::Handle::create();
    params.sublist("Surrogate GS").set("Report Handle", handle);
 \endcode
 *
 \code
     <ParameterList name="Surrogate GS">
       <Parameter name="Type" type="string" value="Krylov Surrogate"/>
       <Parameter name="Inverse Factory" type="string" value="<Some Block Inverse>"/>
       <!-- Optional. Columns of preconditioner output kept, each one full-length
            vector of memory. The store grows as samples arrive. Building the
            report needs about three times the samples' memory again, at peak.
            Default 100. -->
       <Parameter name="Maximum Samples" type="int" value="100"/>
       <!-- Optional. Singular values kept per block, relative. Default 1e-10.
            At 0, roundoff directions are kept too, and the surrogate is noise
            along them. -->
       <Parameter name="Truncation Tolerance" type="double" value="1e-10"/>
       <ParameterList name="Search">                       <!-- optional -->
         <!-- The relative residual each candidate's surrogate solve stops at.
              Set it to the outer solve's convergence tolerance, or the
              predicted iteration counts do not compare with the solve's own.
              Default 1e-8. -->
         <Parameter name="Tolerance" type="double" value="1e-8"/>
         <Parameter name="Candidate Count" type="int" value="100"/>
         <!-- A seed reproduces the sampled candidates within one C++
              standard library, not across them. -->
         <Parameter name="Seed" type="int" value="0"/>
         <ParameterList name="Orderings">                 <!-- always scored -->
           <Parameter name="merge 0 and 1" type="Array(int)" value="{0,0,1}"/>
         </ParameterList>
       </ParameterList>
       <ParameterList name="Report">                       <!-- optional -->
         <!-- Print, at each build, a summary of the solves of the
              preconditioner that build replaces, to Teko's output stream.
              Default false. -->
         <Parameter name="Print" type="bool" value="false"/>
         <!-- Inverses written into each candidate's Hierarchical Block
              Gauss-Seidel list: single-block groups, and merged groups (which
              get a blocked operator). Placeholders if unset. The surrogate
              scored exact solves of both, so name direct solvers (or tight
              iterative ones) for the list to mean what was predicted. -->
         <Parameter name="Inverse Type" type="string" value="<Some Inverse>"/>
         <Parameter name="Merged Inverse Type" type="string" value="<Some Block Inverse>"/>
       </ParameterList>
     </ParameterList>
 \endcode
 *
 * A "Report Handle" (a Teuchos::RCP<Teko::KrylovSurrogate::Handle>, set from
 * code: see Handle) gives the report to the application. One of it and
 * "Report" > "Print" is required, or the report could never be seen.
 *
 * With "Print" on, a build first prints the report of the solves of the
 * preconditioner it replaces: at a rebuild, that preconditioner's own; at a
 * new build, those of the preconditioner this factory built or rebuilt last,
 * if it still exists. (Stratimikos builds a new preconditioner at every
 * initialization in its "Strided Blocking" route, so the second covers it.)
 * Printed samples are forgotten, so nothing is printed twice. The second rule
 * assumes one factory per system: a factory shared by preconditioners of two
 * systems prints, and forgets, one system's samples when the other's is built. A report that cannot
 be built is printed as its
 * error message, and the build goes on. The print is collective over the
 * operator's communicator, like Handle::buildReport, and runs the search.
 * Nothing is built after the last solve of a run, so that solve is never
 * printed: use a handle to see it.
 *
 * A report uses the operator the samples were recorded against, with its
 * values as they are when the report is built: an operator refilled in place
 * gives the new values on the old samples.
 */
class KrylovSurrogatePreconditionerFactory : public virtual Teko::PreconditionerFactory {
 public:
  //! Default constructor, for use with the AutoClone class.
  KrylovSurrogatePreconditionerFactory();

  //! The options are used as given, without the parameter list's validation
  KrylovSurrogatePreconditionerFactory(const Teuchos::RCP<Teko::InverseFactory>& invFactory,
                                       const Teuchos::RCP<Handle>& handle, const Options& options);

  virtual LinearOp buildPreconditionerOperator(LinearOp& lo, PreconditionerState& state) const;

  virtual void initializeFromParameterList(const Teuchos::ParameterList& settings);

  //! Every option, with its default, documentation and valid range
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  virtual Teuchos::RCP<Teuchos::ParameterList> getRequestedParameters() const;

  virtual bool updateRequestedParameters(const Teuchos::ParameterList& pl);

 private:
  // Print the report of the recorder's samples, if it has any, then forget them
  void printAndClear(const Teuchos::RCP<RecorderOp>& recorder) const;

  Teuchos::RCP<Teko::InverseFactory> invFactory_;
  Teuchos::RCP<Handle> handle_;  // may be null when "Print" is on
  Options options_;
  mutable Teuchos::RCP<RecorderOp> lastRecorder_;  // weak, the last built or rebuilt
};

}  // namespace KrylovSurrogate
}  // end namespace Teko

#endif
