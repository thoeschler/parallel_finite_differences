#include "cg.hpp"
#include "la.hpp"

#include <chrono>


void cg(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &u, const double tol, bool verbose) {
    bool converged = false;
    double alpha, gamma;
    std::size_t size = b.size();
    std::vector<double> p(size), Ap(size);

    double bb = dot(b, b);
    double rr, rr_old;

    // initialize p (direction) and r (residual)
    matvec(A, u, Ap);
    std::vector<double> r = b - Ap;
    p = r;
    rr_old = dot(r, r);

    std::size_t counter = 0;

    const auto start = std::chrono::high_resolution_clock::now();
    while (!converged) {
        matvec(A, p, Ap);
        alpha = dot(r, r) / dot(Ap, p);
        u += alpha * p;
        r -= alpha * Ap;
        rr = dot(r, r);
        gamma = rr / rr_old;
        rr_old = rr;
        p = r + gamma * p;

        converged = (rr <= tol * tol * bb);
        if (verbose && counter % 100 == 0) {
            std::cout << "it " << counter << ": rr / bb = " << std::sqrt(rr / bb) << "\n";
        }
        ++counter;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> avg_time = (end - start) / counter;
    std::cout << "average time / it: " << avg_time.count() << "s\n";
}
