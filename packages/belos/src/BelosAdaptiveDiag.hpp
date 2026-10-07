// BelosAdaptiveDiag.hpp — diagnostic prints for the Teko adaptive hook path.
//
// Diagnostic branch only. Every line is
//
//   [TekoDiag <ID>] pid=<pid> <message>
//
// written to BOTH stdout and stderr, flushed, like the branch tracer in
// Teko_InverseFactory.cpp: stdout is where an application logs but is buffered
// and lost on abort, stderr survives that but is often redirected away. IDs are
// unique per call site (see teko-reconfig/claudes_world/plans/
// hook_diag_branch_plan.md for the table), so `grep TekoDiag` reconstructs
// which checks a solve passed and which one stopped it.
//
// Lines are rate-limited per (ID, key): each distinct key prints at most
// kMaxPerKey times per process, then one suppression line. The key defaults to
// the message; call sites whose message carries counters (iterations, curDim)
// pass the message without them. So inner solves (a Belos solve per block
// apply) cannot swamp a run, and, more importantly, cannot use up the budget
// of the one outer solve whose checks come out differently.
//
// No Thyra or MPI dependency, so any solver manager header can include it.
// Nothing here throws or changes control flow.
#pragma once

#include <iostream>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <unistd.h>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace Belos {
namespace AdaptiveDiag {

// Default lines per (ID, key); TEKO_DIAG_MAX overrides it (read once).
constexpr int kMaxPerKey = 3;

inline int maxPerKey()
{
    static const int n = [] {
        const char* v = std::getenv("TEKO_DIAG_MAX");
        if (v == nullptr || *v == '\0') return kMaxPerKey;
        const int parsed = std::atoi(v);
        return parsed > 0 ? parsed : kMaxPerKey;
    }();
    return n;
}

inline void print(const char* id, const std::string& msg, const std::string& key = "")
{
    static std::mutex mtx;
    static std::map<std::string, int> counts;
    std::lock_guard<std::mutex> lock(mtx);
    const int n = ++counts[std::string(id) + "|" + (key.empty() ? msg : key)];
    const int cap = maxPerKey();
    if (n > cap + 1) return;
    std::ostringstream line;
    line << "[TekoDiag " << id << "] pid=" << ::getpid() << " ";
    if (n == cap + 1)
        line << "... further " << id << " lines like this suppressed: "
             << (key.empty() ? msg : key) << "\n";
    else
        line << msg << "\n";
    const std::string s = line.str();
    std::cout << s << std::flush;
    std::cerr << s << std::flush;
}

inline std::string demangle(const char* raw)
{
#if defined(__GNUG__)
    int status = 0;
    char* dem = abi::__cxa_demangle(raw, nullptr, nullptr, &status);
    if (status == 0 && dem) {
        std::string s(dem);
        std::free(dem);
        return s;
    }
#endif
    return raw;
}

// Readable type name, for saying which template instantiation is running.
template <class T>
std::string typeName() { return demangle(typeid(T).name()); }

// Dynamic type of whatever an RCP points at, or "null". This is what names a
// preconditioner as Teko's (Teko::TpetraHelpers::InverseFactoryOperator,
// Teko::PreconditionerLinearOp, ...) when its description() would not.
template <class RCPType>
std::string dynType(const RCPType& p)
{
    if (p.is_null()) return "null";
    return demangle(typeid(*p).name());
}

inline const char* yesNo(bool b) { return b ? "yes" : "no"; }

template <class T, class = void>
struct HasDescription : std::false_type {};
template <class T>
struct HasDescription<T, std::void_t<decltype(std::declval<const T&>().description())>>
    : std::true_type {};

// An object's own description() when it has one (Thyra and Tpetra operators
// are Teuchos::Describable), otherwise its dynamic type name. C++17, since an
// application compiles this header with its own language standard.
template <class T>
std::string describe(const T& obj)
{
    if constexpr (HasDescription<T>::value)
        return obj.description();
    else
        return typeid(obj).name();
}

} // namespace AdaptiveDiag
} // namespace Belos
