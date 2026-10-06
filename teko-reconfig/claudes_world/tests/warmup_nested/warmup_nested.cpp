// warmup_nested.cpp: the factorization warm-up under a nested Teko
// preconditioner.
//
// Builds one block Gauss-Seidel inverse of a 2x2 blocked operator through the
// free Teko::buildInverse. Block GS builds its diagonal-block inverses through
// that same free function, so the call nests, which is what used to deadlock
// in maybeWarmupFactor's call_once. The top-level factory counts its builds
// and the count is printed: 2 when the warm-up ran, 1 when it did not.
//
// Driven by test_warmup_nested.py, one process per environment combination.

#include <iostream>

#include "Teuchos_ParameterList.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraVectorSpace.hpp"

#include "Teko_InverseFactory.hpp"
#include "Teko_InverseLibrary.hpp"
#include "Teko_Utilities.hpp"

using SC       = double;
using LO       = Tpetra::Map<>::local_ordinal_type;
using GO       = Tpetra::Map<>::global_ordinal_type;
using Node     = Tpetra::Map<>::node_type;
using CrsMatrix = Tpetra::CrsMatrix<SC, LO, GO, Node>;

constexpr GO kBlockSize = 20;

// diag on the diagonal, off on the two neighbours
Teko::LinearOp tridiag(Teuchos::RCP<const Tpetra::Map<LO, GO, Node>> map, SC diag, SC off)
{
    auto A = Teuchos::rcp(new CrsMatrix(map, 3));
    for (LO l = 0; l < static_cast<LO>(map->getLocalNumElements()); ++l) {
        const GO g = map->getGlobalElement(l);
        for (GO c : {g - 1, g, g + 1}) {
            if (c < 0 || c >= kBlockSize) continue;
            const SC v = (c == g) ? diag : off;
            A->insertGlobalValues(g, Teuchos::tuple(c), Teuchos::tuple(v));
        }
    }
    A->fillComplete();
    auto space = Thyra::tpetraVectorSpace<SC, LO, GO, Node>(map);
    return Thyra::tpetraLinearOp<SC, LO, GO, Node>(space, space, A);
}

class CountingFactory : public Teko::InverseFactory {
 public:
    explicit CountingFactory(Teuchos::RCP<Teko::InverseFactory> inner) : inner_(inner) {}
    Teko::InverseLinearOp buildInverse(const Teko::LinearOp& A) const override
    {
        ++builds;
        return inner_->buildInverse(A);
    }
    void rebuildInverse(const Teko::LinearOp& A, Teko::InverseLinearOp& dest) const override
    {
        inner_->rebuildInverse(A, dest);
    }
    Teuchos::RCP<const Teuchos::ParameterList> getParameterList() const override
    {
        return inner_->getParameterList();
    }
    std::string toString() const override { return "Counting(" + inner_->toString() + ")"; }

    mutable int builds = 0;

 private:
    Teuchos::RCP<Teko::InverseFactory> inner_;
};

int main(int argc, char* argv[])
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    {
        auto comm = Tpetra::getDefaultComm();
        auto map  = Teuchos::rcp(new Tpetra::Map<LO, GO, Node>(kBlockSize, 0, comm));

        const Teko::LinearOp A = Teko::block2x2(
            tridiag(map, 4.0, -1.0), tridiag(map, 0.1, 0.0),
            tridiag(map, 0.1, 0.0),  tridiag(map, 4.0, -1.0));

        Teuchos::ParameterList pl;
        pl.sublist("BGS").set("Type", "Block Gauss-Seidel");
        pl.sublist("BGS").set("Inverse Type", "Ifpack2");
        auto invLib = Teko::InverseLibrary::buildFromParameterList(pl);

        CountingFactory counting(invLib->getInverseFactory("BGS"));
        Teko::InverseLinearOp inv = Teko::buildInverse(counting, A);

        if (comm->getRank() == 0)
            std::cout << "BUILDS " << counting.builds << std::endl;
    }
    return 0;
}
