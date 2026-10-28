// route_driver.cpp: an external app reaching the adaptive hook by each route.
//
//   route_driver <thyra|tpetra> [max_iters]
//
// Solves one two-field system (two tridiagonal fields, weakly coupled) with a
// Teko block Gauss-Seidel preconditioner and flexible Belos Block GMRES.
//   thyra   a Thyra 2x2 blocked operator through Stratimikos, the way an
//           application configures Teko from XML; reaches the Thyra hook
//   tpetra  Belos called directly with Tpetra::MultiVector / Tpetra::Operator
//           on the interleaved matrix, the right preconditioner a Teko
//           InverseFactoryOperator over a BlockedTpetraOperator, as an
//           application without Thyra does it; reaches the type-erased hook
// max_iters is the outer "Maximum Iterations" (default 200).
//
// Three environment knobs set what a real application varies, for the
// residual-target cases:
//   ROUTE_B_SCALE      b is this constant (default 1)
//   ROUTE_X0           the initial guess is this constant (default 0)
//   ROUTE_RES_SCALING  Belos's "Implicit Residual Scaling" (default: Belos's own)
//
// Prints INITRES, the explicit relative residual of the initial guess, TRUERES,
// the same for the x the application got back, and SOLVE CONVERGED or SOLVE
// UNCONVERGED, what the solver reported.
// Driven by test_tpetra_route.py, one process per case.

#include <cstdlib>
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

constexpr GO     kNodes = 20;    // per field
constexpr double kTol   = 1e-8;

double envDouble(const char* name, double dflt)
{
    const char* v = std::getenv(name);
    return (v && *v) ? std::atof(v) : dflt;
}

// The application's Belos settings: the tolerance and iteration limits every
// case uses, plus ROUTE_RES_SCALING when it is set.
void setBelosParams(Teuchos::ParameterList& p, int maxIters)
{
    p.set("Convergence Tolerance", kTol);
    p.set("Maximum Iterations", maxIters);
    p.set("Num Blocks", 50);
    p.set("Flexible Gmres", true);
    if (const char* s = std::getenv("ROUTE_RES_SCALING"); s && *s)
        p.set("Implicit Residual Scaling", std::string(s));
}

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

void report(int rank, double initRes, double trueRes, bool converged)
{
    if (rank != 0) return;
    std::cout.precision(17);
    std::cout << "INITRES " << initRes << std::endl;
    std::cout << "TRUERES " << trueRes << std::endl;
    std::cout << "SOLVE " << (converged ? "CONVERGED" : "UNCONVERGED") << std::endl;
}

int runTpetra(Teuchos::RCP<const Teuchos::Comm<int>> comm, int maxIters)
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
    x->putScalar(envDouble("ROUTE_X0", 0.0));
    b->putScalar(envDouble("ROUTE_B_SCALE", 1.0));
    auto relres = [&] {
        TMV r(A->getRangeMap(), 1);
        A->apply(*x, r);
        r.update(1.0, *b, -1.0);
        Teuchos::Array<SC> rn(1), bn(1);
        r.norm2(rn());
        b->norm2(bn());
        return rn[0] / bn[0];
    };
    const double initRes = relres();
    auto problem = Teuchos::rcp(new Belos::LinearProblem<SC, TMV, TOP>(A, x, b));
    problem->setRightPrec(prec);
    problem->setProblem();

    auto params = Teuchos::rcp(new Teuchos::ParameterList);
    setBelosParams(*params, maxIters);
    Belos::BlockGmresSolMgr<SC, TMV, TOP> solver(problem, params);
    const Belos::ReturnType ret = solver.solve();

    report(comm->getRank(), initRes, relres(), ret == Belos::Converged);
    return 0;
}

int runThyra(Teuchos::RCP<const Teuchos::Comm<int>> comm, int maxIters)
{
    auto map = Teuchos::rcp(new Map(kNodes, 0, comm));
    const Teko::LinearOp A = Teko::block2x2(tridiag(map, 4.0, -1.0), tridiag(map, 0.1, 0.0),
                                            tridiag(map, 0.1, 0.0), tridiag(map, 4.0, -1.0));

    auto pl = Teuchos::rcp(new Teuchos::ParameterList);
    pl->set("Linear Solver Type", "Belos");
    auto& belos = pl->sublist("Linear Solver Types").sublist("Belos");
    belos.set("Solver Type", "Block GMRES");
    setBelosParams(belos.sublist("Solver Types").sublist("Block GMRES"), maxIters);

    pl->set("Preconditioner Type", "Teko");
    auto& teko = pl->sublist("Preconditioner Types").sublist("Teko");
    teko.set("Inverse Type", "BGS");
    auto& bgs = teko.sublist("Inverse Factory Library").sublist("BGS");
    bgs.set("Type", "Block Gauss-Seidel");
    bgs.set("Inverse Type", "Ifpack2");

    Stratimikos::DefaultLinearSolverBuilder builder;
    Teko::addTekoToStratimikosBuilder(builder);
    builder.setParameterList(pl);
    auto lowsFactory = builder.createLinearSolveStrategy("");
    auto lows = Thyra::linearOpWithSolve(*lowsFactory, A);

    auto x = Thyra::createMember(A->domain());
    auto b = Thyra::createMember(A->range());
    Thyra::assign(x.ptr(), envDouble("ROUTE_X0", 0.0));
    Thyra::assign(b.ptr(), envDouble("ROUTE_B_SCALE", 1.0));
    auto relres = [&] {
        auto r = Thyra::createMember(A->range());
        Thyra::V_V(r.ptr(), *b);
        Thyra::apply<SC>(*A, Thyra::NOTRANS, *x, r.ptr(), -1.0, 1.0);
        return Thyra::norm_2(*r) / Thyra::norm_2(*b);
    };
    const double initRes = relres();
    const auto status = Thyra::solve<SC>(*lows, Thyra::NOTRANS, *b, x.ptr());

    report(comm->getRank(), initRes, relres(),
           status.solveStatus == Thyra::SOLVE_STATUS_CONVERGED);
    return 0;
}

int main(int argc, char* argv[])
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    const std::string route = argc >= 2 ? argv[1] : "";
    if ((route != "thyra" && route != "tpetra") || argc > 3) {
        std::cerr << "usage: route_driver <thyra|tpetra> [max_iters]\n";
        return 2;
    }
    const int maxIters = argc == 3 ? std::atoi(argv[2]) : 200;
    auto comm = Tpetra::getDefaultComm();
    return route == "thyra" ? runThyra(comm, maxIters) : runTpetra(comm, maxIters);
}
