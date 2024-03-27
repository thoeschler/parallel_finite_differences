#include "cg.hpp"
#include "la.hpp"


void get_neighbor_ranks(int &up, int &down, int &left, int &right, MPI_Comm comm_cart) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::vector<int> neighbor_ranks(4, -1); // upper / lower / left / right order
    MPI_Neighbor_allgather(&rank, 1, MPI_INT, neighbor_ranks.data(), 1, MPI_INT, comm_cart);
    // TODO: is the ordering implementation dependent?

    up = neighbor_ranks[0] >= 0 ? neighbor_ranks[0] : MPI_PROC_NULL;
    down = neighbor_ranks[1] >= 0 ? neighbor_ranks[1] : MPI_PROC_NULL;
    left = neighbor_ranks[2] >= 0 ? neighbor_ranks[2] : MPI_PROC_NULL;
    right = neighbor_ranks[3] >= 0 ? neighbor_ranks[3] : MPI_PROC_NULL;
}

void copy_b_loc_to_p_loc(std::vector<double> &p_loc, std::vector<double> const& b_loc,
                         LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    for (std::size_t idx = 0; idx < local_grid.Nx; ++idx) {
        for (std::size_t idy = 0; idy < local_grid.Ny; ++idy) {
            int index = (local_grid.has_lower_neighbor + idy) * Nxt + local_grid.has_left_neighbor + idx ;
            p_loc[index] = b_loc[idy * local_grid.Nx + idx];
        }
    }
}

double dot_padded(std::vector<double> const& Ap_loc, std::vector<double> const& p_loc_padded,
                  LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;

    double result = 0.0;
    std::size_t index;
    for (std::size_t row = 0; row < local_grid.Ny; ++row) {
        for (std::size_t col = 0; col < local_grid.Nx; ++col) {
            index = Nxt * (row + local_grid.has_lower_neighbor) + local_grid.has_left_neighbor + col;
            result += p_loc_padded[index] * Ap_loc[row * local_grid.Nx + col];
        }
    }
    return result;
}

void add_mult_finout_padded(std::vector<double>& inout, std::vector<double> const& in_padded, double multiplier,
                     LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;

    std::size_t index;
    for (std::size_t row = 0; row < local_grid.Ny; ++row) {
        for (std::size_t col = 0; col < local_grid.Nx; ++col) {
            index = Nxt * (row + local_grid.has_lower_neighbor) + local_grid.has_left_neighbor + col;
            inout[row * local_grid.Nx + col] += multiplier * in_padded[index];
        }
    }
}

void add_mult_sinout_padded(std::vector<double> const& in, std::vector<double>& inout_padded, double multiplier,
                     LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;

    std::size_t index;
    for (std::size_t row = 0; row < local_grid.Ny; ++row) {
        for (std::size_t col = 0; col < local_grid.Nx; ++col) {
            index = Nxt * (row + local_grid.has_lower_neighbor) + local_grid.has_left_neighbor + col;
            inout_padded[index] += in[local_grid.Nx * row + col] + multiplier * inout_padded[index];
        }
    }
}

void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc,
                 LocalUnitSquareGrid const& local_grid, MPI_Comm comm_cart, const double tol) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);

    // allocate vectors u, r, Ap
    std::size_t size_loc = local_grid.Nx * local_grid.Ny;
    u_loc.resize(size_loc);
    std::vector<double> r_loc(size_loc), Ap_loc(size_loc);

    // get padded local size (including exchange with neighboring processes) 
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    std::size_t Nyt = local_grid.Ny + local_grid.has_lower_neighbor + local_grid.has_upper_neighbor;
    std::size_t size_loc_padded = Nxt * Nyt;

    // in p_loc data with neighboring processes is exchanged, so it must be larger
    std::vector<double> p_loc_padded(size_loc_padded);

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

    // define "column" type for communication with left/right neighbors
    MPI_Datatype col_type;
    MPI_Type_vector(local_grid.Ny, 1, Nxt, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    while (!converged && counter < 1000) {
        /*
        1st step:
        Start exchange of pk. Communication is only initialized if
        a valid rank is specified, i.e. if dest/source != MPI_PROC_NULL.
        */
        MPI_Request req_send_left, req_send_right, req_send_up, req_send_down,
            req_recv_left, req_recv_right, req_recv_up, req_recv_down;

        // up
        double *sendbuf_up = p_loc_padded.data() + (Nyt - 2) * Nxt + local_grid.has_left_neighbor;
        double *recvbuf_up = p_loc_padded.data() + (Nyt - 1) * Nxt + local_grid.has_left_neighbor;
        MPI_Isend(sendbuf_up, local_grid.Nx, MPI_DOUBLE, up, rank, comm_cart, &req_send_up);
        MPI_Irecv(recvbuf_up, local_grid.Nx, MPI_DOUBLE, up, MPI_ANY_TAG, comm_cart, &req_recv_up);

        // down
        double *sendbuf_down = p_loc_padded.data() + Nxt + local_grid.has_left_neighbor;
        double *recvbuf_down = p_loc_padded.data() + local_grid.has_left_neighbor;
        MPI_Isend(sendbuf_down, local_grid.Nx, MPI_DOUBLE, down, rank, comm_cart, &req_send_down);
        MPI_Irecv(recvbuf_down, local_grid.Nx, MPI_DOUBLE, down, MPI_ANY_TAG, comm_cart, &req_recv_down);

        // left
        double *sendbuf_left = p_loc_padded.data() + local_grid.has_lower_neighbor * Nxt + 1;
        double *recvbuf_left = p_loc_padded.data() + local_grid.has_lower_neighbor * Nxt;
        MPI_Isend(sendbuf_left, 1, col_type, left, rank, comm_cart, &req_send_left);
        MPI_Irecv(recvbuf_left, 1, col_type, left, MPI_ANY_TAG, comm_cart, &req_recv_left);

        // right
        double *sendbuf_right = p_loc_padded.data() + (1 + local_grid.has_lower_neighbor) * Nxt - 2;
        double *recvbuf_right = p_loc_padded.data() + (1 + local_grid.has_lower_neighbor) * Nxt - 1;
        MPI_Isend(sendbuf_right, 1, col_type, right, rank, comm_cart, &req_send_right);
        MPI_Irecv(recvbuf_right, 1, col_type, right, MPI_ANY_TAG, comm_cart, &req_recv_right);

        MPI_Wait(&req_recv_down, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_up, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_left, MPI_STATUS_IGNORE);
        MPI_Wait(&req_recv_right, MPI_STATUS_IGNORE);

        /*
        2nd step:
        Compute A * pk locally.
        */
        matvec(A_loc, p_loc_padded, Ap_loc);

        /*
        3rd step:
        Compute alphak = (rk, rk) / (A * pk, pk).
        */
        double App_loc = dot_padded(Ap_loc, p_loc_padded, local_grid);
        double App;
        MPI_Allreduce(&App_loc, &App, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        alpha = rr / App;

        /*
        4th step:
        Compute xk+1 = xk + alphak * pk.
        */
        add_mult_finout_padded(u_loc, p_loc_padded, alpha, local_grid);

        /*
        5th step:
        Compute rk+1 = rk - alphak * A * pk.
        */
        add_mult_finout(r_loc, Ap_loc, -alpha);

        /*
        6th step:
        Compute gammak = (rk+1, rk+1) / (rk, rk)
        */
        rr_old = rr;
        rr_loc = dot(r_loc, r_loc);
        MPI_Allreduce(&rr_loc, &rr, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        gamma = rr / rr_old;

        /*
        7th step:
        Compute pk+1 = rk+1 + gk * pk.
        */
        add_mult_sinout_padded(r_loc, p_loc_padded, gamma, local_grid);

        if (rank == 0 && counter % 100 == 0) {
            std::cout << "it " << counter << ": rr / bb = " << std::sqrt(rr / bb) << "\n";
        }
        ++counter;
        converged = (rr <= tol * tol * bb);
    }
}
