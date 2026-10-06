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
#include <cstdlib>
#endif

namespace Belos {
namespace AdaptiveDiag {

constexpr int kMaxPerKey = 3;

inline void print(const char* id, const std::string& msg, const std::string& key = "")
{
    static std::mutex mtx;
    static std::map<std::string, int> counts;
    std::lock_guard<std::mutex> lock(mtx);
    const int n = ++counts[std::string(id) + "|" + (key.empty() ? msg : key)];
    if (n > kMaxPerKey + 1) return;
    std::ostringstream line;
    line << "[TekoDiag " << id << "] pid=" << ::getpid() << " ";
    if (n == kMaxPerKey + 1)
        line << "... further " << id << " lines like this suppressed: "
             << (key.empty() ? msg : key) << "\n";
    else
        line << msg << "\n";
    const std::string s = line.str();
    std::cout << s << std::flush;
    std::cerr << s << std::flush;
}

// Readable type name, for saying which template instantiation is running.
template <class T>
std::string typeName()
{
    const char* raw = typeid(T).name();
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
