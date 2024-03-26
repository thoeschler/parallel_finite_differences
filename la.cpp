#include "la.hpp"

#include <iostream>
#include <assert.h>
#include <cmath>

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &out) {
    std::fill(out.begin(), out.end(), 0.0);
    std::size_t row = 0;
    std::size_t row_index = A.row_index(row + 1);
    for (std::size_t value_count = 0; value_count < A.size(); ++value_count) {
        if (value_count == row_index) {
            ++row;
            row_index = A.row_index(row + 1);
        }
        out[row] += A.value(value_count) * b[A.col_index(value_count)];
    }
}

double dot(std::vector<double> const& a, std::vector<double> const& b) {
    assert(a.size() == b.size());

    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double dot(std::vector<double> const& a, std::vector<double>::const_iterator b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i, ++b) {
        result += a[i] * (*b);
    }

    return result;
}

void add(std::vector<double> &inout, std::vector<double> const&in) {
    assert(inout.size() == in.size());

    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += in[i];
    }
}

void add_mult(std::vector<double> const&in1, std::vector<double> const&in2, std::vector<double> &out, double multiplier) {
    assert(in1.size() == in2.size());
    assert(in2.size() == out.size());

    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = in1[i] + multiplier * in2[i];
    }
}

void add_mult_finout(std::vector<double> &inout, std::vector<double> const&in, double multiplier) {
    assert(inout.size() == in.size());

    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += multiplier * in[i];
    }
}

void add_mult_finout(std::vector<double> &inout, std::vector<double>::iterator in, double multiplier) {
    for (std::size_t i = 0; i < inout.size(); ++i, ++in) {
        inout[i] += multiplier * (*in);
    }
}

void add_mult_sinout(std::vector<double> const&in, std::vector<double> &inout, double multiplier) {
    assert(inout.size() == in.size());

    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] = in[i] + multiplier * inout[i];
    }
}

void add_mult_sinout(std::vector<double> const&in, std::vector<double>::iterator& inout, double multiplier) {
    for (std::size_t i = 0; i < in.size(); ++i, ++inout) {
        *inout = in[i] + multiplier * (*inout);
    }
}

void get_neighbor_ranks(int &up, int &down, int &left, int &right, MPI_Comm comm_cart) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::vector<int> neighbor_ranks(4, -1); // upper / lower / left / right order
    MPI_Neighbor_allgather(&rank, 1, MPI_INT, neighbor_ranks.data(), 1, MPI_INT, comm_cart);
    up = neighbor_ranks[0] >= 0 ? neighbor_ranks[0] : MPI_PROC_NULL;
    down = neighbor_ranks[1] >= 0 ? neighbor_ranks[1] : MPI_PROC_NULL;
    left = neighbor_ranks[2] >= 0 ? neighbor_ranks[2] : MPI_PROC_NULL;
    right = neighbor_ranks[3] >= 0 ? neighbor_ranks[3] : MPI_PROC_NULL;
}

void copy_b_loc_to_p_loc(std::vector<double> &p_loc, std::vector<double> const& b_loc,
                         LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    for (std::size_t idx = 0; idx < local_grid.Nx; ++idx) {
        for (std::size_t idy = 0; idy > local_grid.Ny; ++idy) {
            int index = (local_grid.has_lower_neighbor + idy) * Nxt + 1 + idx ;
            p_loc[index] = b_loc[idy * local_grid.Nx + idx];
        }
    }
}

void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc,
                 LocalUnitSquareGrid const& local_grid, MPI_Comm comm_cart, const double tol) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);

    // allocate vectors u, r, Ap
    assert(b_loc.size() == local_grid.Nx * local_grid.Ny);
    std::size_t size_loc = b_loc.size();
    u_loc.resize(size_loc);
    std::vector<double> r_loc(size_loc), Ap_loc(size_loc);

    // get padded local size (including exchange with neighboring processes) 
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    std::size_t Nyt = local_grid.Ny + local_grid.has_lower_neighbor + local_grid.has_upper_neighbor;
    std::size_t size_loc_padded = Nxt * Nyt;

    // in p_loc data with neighboring processes is exchanged, so it must be larger
    std::vector<double> p_loc_padded(size_loc_padded);
    // get start iterator of local data
    auto p_loc_start = p_loc_padded.begin() + local_grid.has_lower_neighbor * Nxt;

    // 0. compute p = r = b - Ax0, initial guess is always x0=0 here
    r_loc = b_loc;
    copy_b_loc_to_p_loc(p_loc_padded, b_loc, local_grid);
    double rr_loc = dot(r_loc, r_loc);
    double bb_loc = dot(b_loc, b_loc);
    double rr, bb;
    MPI_Allreduce(&rr_loc, &rr, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
    MPI_Allreduce(&bb_loc, &bb, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    // initialize other variables
    double alpha, gamma, rr_old;
    bool converged = (rr <= tol * tol * bb);
    std::size_t counter = 0;

    // get neighbor ranks
    int up, down, left, right;
    get_neighbor_ranks(up, down, left, right, comm_cart);

    // define "column" type
    MPI_Datatype col_type;
    MPI_Type_vector(local_grid.Ny, 1, Nxt, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    while (!converged && counter < 10) {
        // 1. communicate pk
        MPI_Request req_send_left, req_send_right, req_send_up, req_send_down,
            req_recv_left, req_recv_right, req_recv_up, req_recv_down;

        // up
        MPI_Isend(p_loc_padded.data() + (Nyt - 2) * Nxt + 1, local_grid.Nx, MPI_DOUBLE, up, rank, comm_cart, &req_send_up);
        MPI_Irecv(p_loc_padded.data() + (Nyt - 1) * Nxt + 1, local_grid.Nx, MPI_DOUBLE, up, up, comm_cart, &req_recv_up);

        // down
        MPI_Isend(p_loc_padded.data() + Nxt + 1, local_grid.Nx, MPI_DOUBLE, down, rank, comm_cart, &req_send_down);
        MPI_Irecv(p_loc_padded.data() + 1, local_grid.Nx, MPI_DOUBLE, down, down, comm_cart, &req_recv_down);

        // left
        MPI_Isend(p_loc_padded.data() + Nxt + 1, 1, col_type, left, rank, comm_cart, &req_send_left);
        MPI_Irecv(p_loc_padded.data() + Nxt, 1, col_type, left, left, comm_cart, &req_recv_left);

        // right
        MPI_Isend(p_loc_padded.data() + 2 * Nxt - 2, 1, col_type, right, rank, comm_cart, &req_send_right);
        MPI_Irecv(p_loc_padded.data() + 2 * Nxt - 1, 1, col_type, right, right, comm_cart, &req_recv_right);

        MPI_Wait(&req_recv_down, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_up, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_left, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_right, MPI_STATUS_IGNORE);

        // 2. compute Apk (locally) --> parallel matvec multiplication
        matvec(A_loc, p_loc_padded, Ap_loc);        

        // 3. compute ak = (rk, rk) / (Apk, pk)
        auto p_loc_start = p_loc_padded.begin() + local_grid.has_lower_neighbor * Nxt;
        double App_loc = dot(Ap_loc, p_loc_start);
        double App;
        MPI_Allreduce(&App_loc, &App, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        alpha = rr / App;

        // 4. xk+1 = xk + ak * pk
        add_mult_finout(u_loc, p_loc_start, alpha);

        // 5. rk+1 = rk - ak Apk
        add_mult_finout(r_loc, Ap_loc, -alpha);

        // 6. gk = (rk+1, rk+1) / (rk, rk)
        rr_old = rr;
        rr_loc = dot(r_loc, r_loc);
        MPI_Allreduce(&rr_loc, &rr, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        gamma = rr / rr_old;

        // 7. pk+1 = rk+1 + gk * pk
        p_loc_start = p_loc_padded.begin() + local_grid.has_lower_neighbor * Nxt;
        add_mult_sinout(r_loc, p_loc_start, gamma);

        if (rank == 0) {// && counter % 100 == 0) {
            std::cout << "it " << counter << ": rr / bb = " << std::sqrt(rr / bb) << "\n";
        }
        ++counter;
        converged = (rr <= tol * tol * bb);
    }
}
