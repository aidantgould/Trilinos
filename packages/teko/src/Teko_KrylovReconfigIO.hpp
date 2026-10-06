// Teko_KrylovReconfigIO.hpp — the JSON file formats exchanged with the
// reconfiguration watcher (wait_for_request.py) and the per-solve result logs.
//
// Three file formats live here, all in the requests dir as s<N>_*.json:
//   * s<N>_request.json  (C++ -> watcher)  the surrogate C_hat / b_hat
//   * s<N>_reconfig.json (watcher -> C++)  the solves requested, one row each
//   * s<N>_conv.json     (C++ output)      every solve that ran, one row each
// plus the small record structs they serialize (RequestedSolve,
// ReconfigResponse, SolveRecord). There is no external JSON library — every file is written to
// a *.tmp and atomically renamed, and the reconfig response is read with a
// small targeted parser (not a general JSON reader). See ACTIVATION.md for the
// field-by-field documentation of each format.
//
// The reduced-model math (CHatData / computeCHat) lives in
// Teko_KrylovReducedModel.hpp, which this header includes for SDM / CHatData.
#pragma once

// ── Standard library ──────────────────────────────────────────────────────
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Filesystem (C++17)
#include <filesystem>
namespace fs = std::filesystem;

// ── Teuchos ───────────────────────────────────────────────────────────────
#include "Teuchos_Assert.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

// ── Teko ──────────────────────────────────────────────────────────────────
#include "Teko_KrylovReducedModel.hpp"  // type aliases (SDM) + CHatData

namespace Teko {
namespace KrylovSurrogate {

// ═══════════════════════════════════════════════════════════════════════════
// s<N>_request.json — C++ → watcher (the surrogate)
// ═══════════════════════════════════════════════════════════════════════════

// Write a flat JSON array of ints: [a, b, c].
inline void writeIntArrayJson(std::ostream& os, const std::vector<int>& v)
{
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i) { if (i) os << ", "; os << v[i]; }
    os << "]";
}

// Write a dense matrix as a JSON array-of-arrays. (std::setprecision is sticky,
// so it is set once per stream — here and in the writers below.)
inline void writeDenseJson(std::ostream& os, const SDM& M, int indent)
{
    const std::string pad(indent, ' ');
    const std::string inner(indent + 2, ' ');
    os << std::setprecision(17) << pad << "[\n";
    for (int r = 0; r < M.numRows(); ++r) {
        os << inner << "[";
        for (int c = 0; c < M.numCols(); ++c) {
            if (c) os << ", ";
            os << M(r, c);
        }
        os << "]";
        if (r + 1 < M.numRows()) os << ",";
        os << "\n";
    }
    os << pad << "]";
}

// Write a (k × 1) column vector as a flat JSON array.
inline void writeVectorJson(std::ostream& os, const SDM& v)
{
    os << std::setprecision(17) << "[";
    for (int r = 0; r < v.numRows(); ++r) {
        if (r) os << ", ";
        os << v(r, 0);
    }
    os << "]";
}

// Scan requests_dir for existing s<N>_request.json files and return
// max(N) + 1, or 0 if none exist. This is how the C++ side picks the id for
// the next request, so request ids are unique and monotonically increasing
// across the lifetime of requests_dir (they are never reused, even if old
// s<N>_request.json files are deleted).
inline int nextRequestNumber(const std::string& requests_dir)
{
    int next = 0;
    if (!fs::exists(requests_dir)) return next;

    // Consider both s<N>_request.json and s<N>_conv.json. A non-converged
    // (record-only) solve writes a s<N>_conv.json with no matching
    // request.json; counting conv files too keeps that N reserved so a later
    // request can't reuse it and overwrite the record. _solved.json is a
    // retired format, still counted so a stale one keeps its id.
    const std::string prefix = "s";
    const std::array<std::string, 3> suffixes{"_request.json", "_conv.json", "_solved.json"};
    for (const auto& entry : fs::directory_iterator(requests_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        for (const auto& suffix : suffixes) {
            if (name.size() <= prefix.size() + suffix.size()) continue;
            if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;

            const std::string digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
            if (digits.empty() || !std::all_of(digits.begin(), digits.end(),
                                                [](char c){ return std::isdigit(static_cast<unsigned char>(c)); })) continue;

            const int n = std::stoi(digits);
            if (n + 1 > next) next = n + 1;
        }
    }
    return next;
}

// equation_ends[k] = sum(ranks[0..k-1]) for k = 0..nb, so equation_ends[0]==0
// and equation_ends[nb] == R (the total rank, R = sum(ranks)). Each
// equation_ends[k+1] is one-past-the-last row/col index of block k in the
// assembled C_hat matrix / b_hat vector below.
inline std::vector<int> computeEquationEnds(const CHatData& chat)
{
    std::vector<int> ends(chat.nb + 1, 0);
    for (int j = 0; j < chat.nb; ++j)
        ends[j + 1] = ends[j] + chat.ranks[j];
    return ends;
}

// Assemble the full R x R C_hat matrix (R = sum(ranks)) from chat.blocks,
// placing block (i,j) — size ranks[i] x ranks[j] — at rows
// [equation_ends[i], equation_ends[i+1]) and cols
// [equation_ends[j], equation_ends[j+1]).
inline SDM assembleCHat(const CHatData& chat, const std::vector<int>& equation_ends)
{
    const int R = equation_ends.back();
    SDM C(R, R);
    for (int i = 0; i < chat.nb; ++i) {
        for (int j = 0; j < chat.nb; ++j) {
            const SDM& blk = chat.blocks[i][j];
            for (int r = 0; r < blk.numRows(); ++r)
                for (int c = 0; c < blk.numCols(); ++c)
                    C(equation_ends[i] + r, equation_ends[j] + c) = blk(r, c);
        }
    }
    return C;
}

// Assemble the full R x 1 b_hat vector (R = sum(ranks)) by stacking
// chat.b_hat[j] — size ranks[j] x 1 — at rows
// [equation_ends[j], equation_ends[j+1]).
inline SDM assembleBHat(const CHatData& chat, const std::vector<int>& equation_ends)
{
    const int R = equation_ends.back();
    SDM b(R, 1);
    for (int j = 0; j < chat.nb; ++j) {
        const SDM& bj = chat.b_hat[j];
        for (int r = 0; r < bj.numRows(); ++r)
            b(equation_ends[j] + r, 0) = bj(r, 0);
    }
    return b;
}

// Write s<N>_request.json to requests_dir, where N = nextRequestNumber(requests_dir).
// Returns N so the caller can wait for the matching s<N>_reconfig.json
// and tag s<N>_conv.json with the same id.
//
// Format:
// {
//   "n_blocks": nb,
//   "block_sizes": [...],
//   "krylov_dim": m,
//   "ranks": [...],
//   "equation_ends": [0, r0, r0+r1, ..., R],   // length nb+1, R = sum(ranks)
//   "C_hat": [ [ ... ], ... ],                  // single R x R matrix
//   "b_hat": [ ... ]                            // single length-R vector
// }
//
// C_hat is the full R x R surrogate operator (R = sum(ranks)) and b_hat the
// full length-R reduced RHS, both assembled block-by-block. "equation_ends"
// gives the block boundaries within them: equation_ends[k] is the first
// row/col index of block k and equation_ends[k+1] one past its last, so
// block k occupies [equation_ends[k], equation_ends[k+1]).
//
// Every entry scales with nb (number of blocks) and ranks[j] (Krylov-derived
// rank, <= krylov_dim) only — nothing here scales with the global problem
// size n, so this stays small even for very large systems.
inline int writeRequestJson(const CHatData& chat, const std::string& requests_dir)
{
    const int request_id = nextRequestNumber(requests_dir);

    const std::vector<int> equation_ends = computeEquationEnds(chat);
    const SDM C = assembleCHat(chat, equation_ends);
    const SDM b = assembleBHat(chat, equation_ends);

    // Write to a temp file and atomically rename into place. Without this, a
    // reader's fs::exists(path) check (in wait_for_request.py) can observe
    // the file mid-write and parse incomplete JSON. A same-directory
    // rename() is atomic on POSIX, so readers only ever see the file fully
    // written or not at all.
    const std::string path     = requests_dir + "/s" + std::to_string(request_id) + "_request.json";
    const std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path);
    TEUCHOS_TEST_FOR_EXCEPTION(!ofs.is_open(), std::runtime_error,
        "Teko::KrylovSurrogate::writeRequestJson: cannot open " + tmp_path);

    ofs << "{\n";
    ofs << "  \"n_blocks\": " << chat.nb << ",\n";
    ofs << "  \"block_sizes\": ";   writeIntArrayJson(ofs, chat.block_sizes); ofs << ",\n";
    ofs << "  \"krylov_dim\": " << chat.krylov_dim << ",\n";
    ofs << "  \"ranks\": ";         writeIntArrayJson(ofs, chat.ranks);       ofs << ",\n";
    ofs << "  \"equation_ends\": "; writeIntArrayJson(ofs, equation_ends);    ofs << ",\n";

    // C_hat: single R x R matrix
    ofs << "  \"C_hat\":\n";
    writeDenseJson(ofs, C, 2);
    ofs << ",\n";

    // b_hat: single length-R vector
    ofs << "  \"b_hat\": ";
    writeVectorJson(ofs, b);
    ofs << "\n";

    ofs << "}\n";
    ofs.close();

    fs::rename(tmp_path, path);

    // Absolute, on both streams: this is the file a watcher has to find, and
    // the path is useless to whoever is debugging if it is relative to a
    // working directory they cannot see.
    announceBoth("[TekoAdaptive] wrote request " + absolutePathString(path) + "\n");
    return request_id;
}

// ═══════════════════════════════════════════════════════════════════════════
// s<N>_reconfig.json — watcher → C++ (the solves requested)
// ═══════════════════════════════════════════════════════════════════════════

// ── Minimal strict parsing for the reconfig response ──────────────────────
// s<N>_reconfig.json is machine-generated by wait_for_request.py in a fixed
// shape, e.g.
//   {"request_id": 3,
//    "solves": [
//      {"ordering": [0,1,2], "surrogate_iters": 6,
//       "surrogate_flags": ["use_ordering", "opt_ordering"]},
//      {"ordering": [0,0,1], "surrogate_iters": null, "surrogate_flags": []}]}
// These helpers extract exactly the fields the C++ side consumes: the
// "solves" array, and in each of its objects an int array, an int-or-null,
// and an array of strings. Keyed off field names and rejecting malformed
// numeric tokens. It is deliberately a small targeted parser, not a general
// JSON reader (no escapes inside strings, no nested objects in a row), but
// keying off the field names makes it tolerant of extra fields in any
// position.

// Parse one integer from a token, rejecting anything std::stoi would silently
// truncate (e.g. "0.9" -> 0).
inline int parseStrictInt(std::string tok, const std::string& ctx)
{
    tok.erase(std::remove_if(tok.begin(), tok.end(),
                             [](char c){ return std::isspace(static_cast<unsigned char>(c)); }),
              tok.end());
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(tok, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(ctx + ": non-integer entry \"" + tok + "\"");
    }
    if (consumed != tok.size())
        throw std::runtime_error(ctx + ": non-integer entry \"" + tok + "\"");
    return value;
}

// Index of the closing char matching the opening one at openPos (balanced).
// npos if unbalanced.
inline std::size_t matchClosing(const std::string& s, std::size_t openPos,
                                char open, char close)
{
    int depth = 0;
    for (std::size_t i = openPos; i < s.size(); ++i) {
        if (s[i] == open) ++depth;
        else if (s[i] == close) { if (--depth == 0) return i; }
    }
    return std::string::npos;
}

inline std::size_t matchBracket(const std::string& s, std::size_t openPos)
{
    return matchClosing(s, openPos, '[', ']');
}

// Parse "a, b, c" (comma-separated, no brackets) into ints.
inline std::vector<int> parseIntCsv(const std::string& body, const std::string& ctx)
{
    std::vector<int> out;
    std::istringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const bool nonspace = std::any_of(tok.begin(), tok.end(),
            [](char c){ return !std::isspace(static_cast<unsigned char>(c)); });
        if (nonspace) out.push_back(parseStrictInt(tok, ctx));
    }
    return out;
}

// Optional "key": [ints] field. Returns false if the key is absent.
inline bool parseIntArrayField(const std::string& content, const std::string& key,
                               std::vector<int>& out, const std::string& ctx)
{
    const auto kp = content.find("\"" + key + "\"");
    if (kp == std::string::npos) return false;
    const auto lb = content.find('[', kp);
    if (lb == std::string::npos)
        throw std::runtime_error(ctx + ": \"" + key + "\" has no array");
    const auto rb = matchBracket(content, lb);
    if (rb == std::string::npos)
        throw std::runtime_error(ctx + ": \"" + key + "\" array not closed");
    out = parseIntCsv(content.substr(lb + 1, rb - lb - 1), ctx + " \"" + key + "\"");
    return true;
}

// Optional "key": int-or-null field. A null (or an absent key) gives -1, the
// in-memory spelling of null for an iteration count.
inline int parseIntOrNullField(const std::string& content, const std::string& key,
                               const std::string& ctx)
{
    const auto kp = content.find("\"" + key + "\"");
    if (kp == std::string::npos) return -1;
    const auto colon = content.find(':', kp + key.size() + 2);
    if (colon == std::string::npos)
        throw std::runtime_error(ctx + ": \"" + key + "\" has no value");
    const auto end = content.find_first_of(",}\n", colon + 1);
    std::string tok = content.substr(colon + 1, end == std::string::npos
                                                    ? std::string::npos
                                                    : end - colon - 1);
    tok.erase(std::remove_if(tok.begin(), tok.end(),
                             [](char c){ return std::isspace(static_cast<unsigned char>(c)); }),
              tok.end());
    if (tok == "null") return -1;
    return parseStrictInt(tok, ctx + " \"" + key + "\"");
}

// Optional "key": ["a", "b"] field. Returns false if the key is absent.
inline bool parseStringArrayField(const std::string& content, const std::string& key,
                                  std::vector<std::string>& out, const std::string& ctx)
{
    const auto kp = content.find("\"" + key + "\"");
    if (kp == std::string::npos) return false;
    const auto lb = content.find('[', kp);
    if (lb == std::string::npos)
        throw std::runtime_error(ctx + ": \"" + key + "\" has no array");
    const auto rb = matchBracket(content, lb);
    if (rb == std::string::npos)
        throw std::runtime_error(ctx + ": \"" + key + "\" array not closed");
    out.clear();
    std::size_t i = lb + 1;
    while (true) {
        const auto q1 = content.find('"', i);
        if (q1 == std::string::npos || q1 > rb) break;
        const auto q2 = content.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > rb)
            throw std::runtime_error(ctx + ": \"" + key + "\" string not closed");
        out.push_back(content.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return true;
}

// One row of s<N>_reconfig.json: a solve the watcher is asking for.
struct RequestedSolve {
    std::vector<int>         ordering;
    int                      surrogate_iters = -1;  // -1 == null (no search ran)
    std::vector<std::string> surrogate_flags;       // e.g. use_ordering, opt_ordering
};

// The flag that makes a row the final solve. Without it on any row, the final
// is the requested solve that performed best (see adaptiveLoop).
inline const std::string kUseOrderingFlag = "use_ordering";

inline bool hasFlag(const RequestedSolve& s, const std::string& flag)
{
    return std::find(s.surrogate_flags.begin(), s.surrogate_flags.end(), flag)
           != s.surrogate_flags.end();
}

// Parsed reconfig response. received is false on timeout. received with no
// solves is an answer asking for nothing, and is treated the same way.
struct ReconfigResponse {
    bool                        received = false;
    std::vector<RequestedSolve> solves;
};

// Parse the "solves" array of a reconfig response, one object at a time.
inline std::vector<RequestedSolve> parseSolvesField(const std::string& content,
                                                    const std::string& ctx)
{
    const auto kp = content.find("\"solves\"");
    if (kp == std::string::npos)
        throw std::runtime_error(ctx + ": no \"solves\" field found");
    const auto outerL = content.find('[', kp);
    if (outerL == std::string::npos)
        throw std::runtime_error(ctx + ": \"solves\" has no array");
    const auto outerR = matchBracket(content, outerL);
    if (outerR == std::string::npos)
        throw std::runtime_error(ctx + ": \"solves\" array not closed");

    std::vector<RequestedSolve> out;
    std::size_t i = outerL + 1;
    while (true) {
        const auto objL = content.find('{', i);
        if (objL == std::string::npos || objL > outerR) break;
        const auto objR = matchClosing(content, objL, '{', '}');
        if (objR == std::string::npos || objR > outerR)
            throw std::runtime_error(ctx + ": \"solves\" row not closed");
        const std::string obj = content.substr(objL, objR - objL + 1);
        const std::string rowCtx = ctx + " solves[" + std::to_string(out.size()) + "]";

        RequestedSolve s;
        if (!parseIntArrayField(obj, "ordering", s.ordering, rowCtx))
            throw std::runtime_error(rowCtx + ": no \"ordering\" field found");
        s.surrogate_iters = parseIntOrNullField(obj, "surrogate_iters", rowCtx);
        parseStringArrayField(obj, "surrogate_flags", s.surrogate_flags, rowCtx);
        out.push_back(std::move(s));
        i = objR + 1;
    }
    return out;
}

// Poll for s<request_id>_reconfig.json and parse it. The file is left in place
// after being read (NOT deleted) — its presence is how wait_for_request.py
// recognizes an already-answered request. Returns received == false on
// timeout.
inline ReconfigResponse waitForReconfig(
    const std::string& requests_dir,
    int                request_id,
    int                timeout_s = 600)
{
    const std::string path = requests_dir + "/s" + std::to_string(request_id) + "_reconfig.json";
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);

    // The whole point of this line is to be actionable while the solve is
    // blocked: it names the exact file being waited for, in full, plus the
    // directory a watcher must be pointed at and how long before giving up.
    // On both streams, since a run that is stuck here is exactly the run whose
    // buffered stdout may never flush.
    const std::string abs_path = absolutePathString(path);
    const std::string dir = std::filesystem::path(abs_path).parent_path().string();
    announceBoth("[TekoAdaptive] waiting up to " + std::to_string(timeout_s)
                 + "s for the watcher to write " + abs_path + "\n"
                 + "[TekoAdaptive]   a watcher must be running on: " + dir + "\n");

    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(path)) {
            std::ifstream ifs(path);
            std::string content((std::istreambuf_iterator<char>(ifs)),
                                 std::istreambuf_iterator<char>());
            ifs.close();

            const std::string ctx = "[TekoAdaptive] " + path;
            ReconfigResponse r;
            r.received = true;
            r.solves   = parseSolvesField(content, ctx);

            const auto n_use = std::count_if(r.solves.begin(), r.solves.end(),
                [](const RequestedSolve& s){ return hasFlag(s, kUseOrderingFlag); });
            std::cout << "[TekoAdaptive] reconfig: " << r.solves.size()
                      << " solves requested, "
                      << (n_use ? "use_ordering given" : "no use_ordering (best time wins)")
                      << "\n";
            if (n_use > 1)
                std::cerr << "[TekoAdaptive] " << n_use << " rows flagged use_ordering;"
                             " the first one is used.\n";
            return r;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cerr << "[TekoAdaptive] timed out waiting for " << path << "\n";
    return {};
}

// ═══════════════════════════════════════════════════════════════════════════
// s<N>_conv.json — C++ output (every solve that ran)
// ═══════════════════════════════════════════════════════════════════════════

// One solve, one row of s<N>_conv.json. Fields are declared in the order they
// are written: what was solved and what the surrogate predicted for it, what
// it took, then the flags, then residuals and timing.
struct SolveRecord {
    std::vector<int>         ordering;
    int                      surrogate_iters = -1;  // -1 == null
    int                      iters           = 0;
    std::string              type;                  // initial | test | final from selection mode <m>
    std::vector<std::string> surrogate_flags;
    bool                     converged       = false;
    double                   initial_residual      = 0.0;
    double                   final_residual        = 0.0;
    double                   factor_wall_time_sec  = 0.0;
    double                   iterate_wall_time_sec = 0.0;
    double                   total_wall_time_sec   = 0.0;
};

// The reconfig row's own fields, carried onto the record of its solve so
// conv.json holds everything reconfig.json did.
inline void copySurrogateFields(const RequestedSolve& from, SolveRecord& to)
{
    to.surrogate_iters = from.surrogate_iters;
    to.surrogate_flags = from.surrogate_flags;
}

// A JSON string. Escapes only '"' and '\', which is all a flag or type name
// could plausibly contain.
inline void writeStringJson(std::ostream& os, const std::string& s)
{
    os << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') os << '\\';
        os << c;
    }
    os << '"';
}

// A JSON number, or null when it is not finite: a solve that breaks down
// (a singular merged block, say) can leave a NaN residual, and NaN is not
// JSON.
inline void writeDoubleJson(std::ostream& os, double v)
{
    if (std::isfinite(v)) os << v; else os << "null";
}

// Write s<request_id>_conv.json (atomically, via a .tmp file + rename): every
// solve this request ran, in execution order (the initial solve, the tests,
// the final).
inline void writeConvergenceJson(
    const std::string&              convergence_dir,
    int                              request_id,
    const std::vector<SolveRecord>& solves)
{
    fs::create_directories(convergence_dir);

    const std::string path     = convergence_dir + "/s" + std::to_string(request_id) + "_conv.json";
    const std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path);
    TEUCHOS_TEST_FOR_EXCEPTION(!ofs.is_open(), std::runtime_error,
        "Teko::KrylovSurrogate::writeConvergenceJson: cannot open " + tmp_path);
    ofs << std::setprecision(17);

    ofs << "{\n";
    ofs << "  \"request_id\": " << request_id << ",\n";
    ofs << "  \"solves\": [\n";
    for (std::size_t i = 0; i < solves.size(); ++i) {
        const SolveRecord& s = solves[i];
        ofs << "    {\n";
        ofs << "      \"ordering\": "; writeIntArrayJson(ofs, s.ordering); ofs << ",\n";
        ofs << "      \"surrogate_iters\": ";
        if (s.surrogate_iters < 0) ofs << "null"; else ofs << s.surrogate_iters;
        ofs << ",\n";
        ofs << "      \"iters\": " << s.iters << ",\n";
        ofs << "      \"type\": "; writeStringJson(ofs, s.type); ofs << ",\n";
        ofs << "      \"surrogate_flags\": [";
        for (std::size_t f = 0; f < s.surrogate_flags.size(); ++f) {
            if (f) ofs << ", ";
            writeStringJson(ofs, s.surrogate_flags[f]);
        }
        ofs << "],\n";
        ofs << "      \"converged\": " << (s.converged ? "true" : "false") << ",\n";
        ofs << "      \"initial_residual\": "; writeDoubleJson(ofs, s.initial_residual); ofs << ",\n";
        ofs << "      \"final_residual\": "; writeDoubleJson(ofs, s.final_residual); ofs << ",\n";
        ofs << "      \"factor_wall_time_sec\": "; writeDoubleJson(ofs, s.factor_wall_time_sec); ofs << ",\n";
        ofs << "      \"iterate_wall_time_sec\": "; writeDoubleJson(ofs, s.iterate_wall_time_sec); ofs << ",\n";
        ofs << "      \"total_wall_time_sec\": "; writeDoubleJson(ofs, s.total_wall_time_sec); ofs << "\n";
        ofs << "    }" << (i + 1 < solves.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();

    fs::rename(tmp_path, path);
    std::cout << "[TekoAdaptive] wrote " << path << "\n";
}

} // namespace KrylovSurrogate
} // namespace Teko
