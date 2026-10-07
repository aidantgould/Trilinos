// hook_diag.cpp: an external app reaching Belos through Stratimikos, for the
// diagnostic branch's [TekoDiag] prints.
//
//   hook_diag <pseudo|flex> <flat|blocked> [inner-gmres]
//   hook_diag <tpetra-pseudo|tpetra-flex>
//
// Solves one two-field system with a Teko block Gauss-Seidel preconditioner
// registered in Stratimikos, the way an application configures it from XML.
//   pseudo   Belos "Pseudo Block GMRES", which has no adaptive hook
//   flex     Belos "Block GMRES" with "Flexible Gmres" true
//   flat     one interleaved Tpetra matrix, split by Teko's "Strided
//            Blocking" inside the preconditioner, so Belos sees a flat
//            operator
//   blocked  a Thyra 2x2 blocked operator handed to Stratimikos as is
//   inner-gmres  solve each diagonal block with a non-flexible Belos Block
//            GMRES instead of Ifpack2, so inner Block GMRES solves print the
//            same diagnostics as the outer one and could crowd it out
//
//   tpetra-*  Belos called directly with Tpetra::MultiVector /
//            Tpetra::Operator on the flat interleaved matrix, the right
//            preconditioner a Teko InverseFactoryOperator built over a
//            BlockedTpetraOperator, as an application without Thyra does it;
//            prints TRUERES, the explicit relative residual of the returned x
//
// flex + blocked reaches the Thyra hook, tpetra-flex the type-erased one.
// Driven by test_hook_diag.py, one process per case.

#include <iostream>
#include <string>

#include "Teuchos_ParameterList.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraVectorSpace.hpp"
#include "Thyra_LinearOpWithSolveBase.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"
#include "BelosTpetraAdapter.hpp"

#include "Teko_StratimikosFactory.hpp"
#include "Teko_Utilities.hpp"
#include "Teko_InverseLibrary.hpp"
#include "Teko_BlockedTpetraOperator.hpp"
#include "Teko_TpetraInverseFactoryOperator.hpp"

using SC        = double;
using LO        = Tpetra::Map<>::local_ordinal_type;
using GO        = Tpetra::Map<>::global_ordinal_type;
using Node      = Tpetra::Map<>::node_type;
using Map       = Tpetra::Map<LO, GO, Node>;
using CrsMatrix = Tpetra::CrsMatrix<SC, LO, GO, Node>;
using TMV       = Tpetra::MultiVector<SC, LO, GO, Node>;
using TOP       = Tpetra::Operator<SC, LO, GO, Node>;

constexpr GO kNodes = 20;  // per field

Teko::LinearOp wrap(Teuchos::RCP<CrsMatrix> A)
{
    auto space = Thyra::tpetraVectorSpace<SC, LO, GO, Node>(A->getRowMap());
    return Thyra::tpetraLinearOp<SC, LO, GO, Node>(space, space, A);
}

// diag on the diagonal, off on the two neighbours, over kNodes unknowns
Teko::LinearOp tridiag(Teuchos::RCP<const Map> map, SC diag, SC off)
{
    auto A = Teuchos::rcp(new CrsMatrix(map, 3));
    for (LO l = 0; l < static_cast<LO>(map->getLocalNumElements()); ++l) {
        const GO g = map->getGlobalElement(l);
        for (GO c : {g - 1, g, g + 1}) {
            if (c < 0 || c >= kNodes) continue;
            if (c != g && off == 0.0) continue;
            A->insertGlobalValues(g, Teuchos::tuple(c), Teuchos::tuple(c == g ? diag : off));
        }
    }
    A->fillComplete();
    return wrap(A);
}

// The same system as one matrix, fields interleaved: row 2i is field 0 at node
// i, row 2i+1 is field 1 at node i.
Teuchos::RCP<CrsMatrix> interleavedCrs(Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    auto map = Teuchos::rcp(new Map(2 * kNodes, 0, comm));
    auto A   = Teuchos::rcp(new CrsMatrix(map, 4));
    for (LO l = 0; l < static_cast<LO>(map->getLocalNumElements()); ++l) {
        const GO g = map->getGlobalElement(l);
        const GO node = g / 2, field = g % 2;
        for (GO n : {node - 1, node, node + 1}) {
            if (n < 0 || n >= kNodes) continue;
            const GO c = 2 * n + field;
            A->insertGlobalValues(g, Teuchos::tuple(c), Teuchos::tuple(n == node ? 4.0 : -1.0));
        }
        A->insertGlobalValues(g, Teuchos::tuple(2 * node + (1 - field)), Teuchos::tuple(0.1));
    }
    A->fillComplete();
    return A;
}

Teko::LinearOp interleaved(Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    return wrap(interleavedCrs(comm));
}

// Belos with Tpetra types and a Teko block Gauss-Seidel preconditioner over a
// BlockedTpetraOperator of the interleaved matrix.
int runTpetra(Teuchos::RCP<const Teuchos::Comm<int>> comm, bool flexible)
{
    auto A = interleavedCrs(comm);
    std::vector<std::vector<GO>> vars(2);
    for (LO l = 0; l < static_cast<LO>(A->getRowMap()->getLocalNumElements()); ++l) {
        const GO g = A->getRowMap()->getGlobalElement(l);
        vars[g % 2].push_back(g);
    }
    auto blocked = Teuchos::rcp(new Teko::TpetraHelpers::BlockedTpetraOperator(vars, A));

    Teuchos::ParameterList pl;
    pl.sublist("BGS").set("Type", "Block Gauss-Seidel");
    pl.sublist("BGS").set("Inverse Type", "Ifpack2");
    auto invLib = Teko::InverseLibrary::buildFromParameterList(pl);
    auto prec = Teuchos::rcp(
        new Teko::TpetraHelpers::InverseFactoryOperator(invLib->getInverseFactory("BGS")));
    prec->initInverse();
    prec->buildInverseOperator(Teuchos::rcp_implicit_cast<const TOP>(blocked));

    auto x = Teuchos::rcp(new TMV(A->getDomainMap(), 1));
    auto b = Teuchos::rcp(new TMV(A->getRangeMap(), 1));
    b->putScalar(1.0);
    auto problem = Teuchos::rcp(new Belos::LinearProblem<SC, TMV, TOP>(A, x, b));
    problem->setRightPrec(prec);
    problem->setProblem();

    auto params = Teuchos::rcp(new Teuchos::ParameterList);
    params->set("Convergence Tolerance", 1e-8);
    params->set("Maximum Iterations", 200);
    params->set("Num Blocks", 50);
    Belos::ReturnType ret;
    if (flexible) {
        params->set("Flexible Gmres", true);
        Belos::BlockGmresSolMgr<SC, TMV, TOP> solver(problem, params);
        ret = solver.solve();
    } else {
        Belos::PseudoBlockGmresSolMgr<SC, TMV, TOP> solver(problem, params);
        ret = solver.solve();
    }

    TMV r(A->getRangeMap(), 1);
    A->apply(*x, r);
    r.update(1.0, *b, -1.0);
    Teuchos::Array<SC> rn(1), bn(1);
    r.norm2(rn());
    b->norm2(bn());
    if (comm->getRank() == 0) {
        std::cout << "TRUERES " << rn[0] / bn[0] << std::endl;
        std::cout << "SOLVE " << (ret == Belos::Converged ? "SOLVE_STATUS_CONVERGED"
                                                          : "SOLVE_STATUS_UNCONVERGED")
                  << std::endl;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    if (argc == 2 && std::string(argv[1]).rfind("tpetra-", 0) == 0) {
        const bool flexible = std::string(argv[1]) == "tpetra-flex";
        return runTpetra(Tpetra::getDefaultComm(), flexible);
    }
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: hook_diag <pseudo|flex> <flat|blocked> [inner-gmres]\n"
                     "       hook_diag <tpetra-pseudo|tpetra-flex>\n";
        return 2;
    }
    const std::string solver = argv[1], layout = argv[2];
    const bool innerGmres = (argc == 4 && std::string(argv[3]) == "inner-gmres");
    {
        auto comm = Tpetra::getDefaultComm();

        Teko::LinearOp A;
        if (layout == "blocked") {
            auto map = Teuchos::rcp(new Map(kNodes, 0, comm));
            A = Teko::block2x2(tridiag(map, 4.0, -1.0), tridiag(map, 0.1, 0.0),
                               tridiag(map, 0.1, 0.0), tridiag(map, 4.0, -1.0));
        } else {
            A = interleaved(comm);
        }

        auto pl = Teuchos::rcp(new Teuchos::ParameterList);
        pl->set("Linear Solver Type", "Belos");
        auto& belos = pl->sublist("Linear Solver Types").sublist("Belos");
        belos.set("Solver Type", solver == "flex" ? "Block GMRES" : "Pseudo Block GMRES");
        auto& gm = belos.sublist("Solver Types")
                       .sublist(solver == "flex" ? "Block GMRES" : "Pseudo Block GMRES");
        gm.set("Convergence Tolerance", 1e-8);
        gm.set("Maximum Iterations", 200);
        gm.set("Num Blocks", 50);
        if (solver == "flex") gm.set("Flexible Gmres", true);

        pl->set("Preconditioner Type", "Teko");
        auto& teko = pl->sublist("Preconditioner Types").sublist("Teko");
        teko.set("Inverse Type", "BGS");
        if (layout == "flat") teko.set("Strided Blocking", "1 1");
        auto& bgs = teko.sublist("Inverse Factory Library").sublist("BGS");
        bgs.set("Type", "Block Gauss-Seidel");
        bgs.set("Inverse Type", innerGmres ? "InnerGMRES" : "Ifpack2");
        if (innerGmres) {
            auto& inner = teko.sublist("Inverse Factory Library").sublist("InnerGMRES");
            inner.set("Type", "Belos");
            inner.set("Solver Type", "Block GMRES");
            auto& ig = inner.sublist("Solver Types").sublist("Block GMRES");
            ig.set("Convergence Tolerance", 1e-6);
            ig.set("Maximum Iterations", 100);
        }

        Stratimikos::DefaultLinearSolverBuilder builder;
        Teko::addTekoToStratimikosBuilder(builder);
        builder.setParameterList(pl);
        auto lowsFactory = builder.createLinearSolveStrategy("");
        auto lows = Thyra::linearOpWithSolve(*lowsFactory, A);

        auto x = Thyra::createMember(A->domain());
        auto b = Thyra::createMember(A->range());
        Thyra::assign(x.ptr(), 0.0);
        Thyra::assign(b.ptr(), 1.0);
        const auto status = Thyra::solve<SC>(*lows, Thyra::NOTRANS, *b, x.ptr());

        if (comm->getRank() == 0)
            std::cout << "SOLVE " << Thyra::toString(status.solveStatus) << std::endl;
    }
    return 0;
}
