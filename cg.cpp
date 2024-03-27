#include "cg.hpp"
#include "la.hpp"

#include <assert.h>

enum Side { top = 0, bottom = 1, left = 2, right = 3 };
enum CornerPoints {topleft = 0, topright = 1, bottomright = 2, bottomleft = 3 };

void get_neighbor_ranks(int &top, int &bottom, int &left, int &right, MPI_Comm comm_cart);
void copy_b_loc_to_p_loc(std::vector<double> &p_loc, std::vector<double> const& b_loc, LocalUnitSquareGrid const& local_grid);
double dot_padded(std::vector<double> const& not_padded, std::vector<double> const& padded, LocalUnitSquareGrid const& local_grid);
void add_mult_finout_padded(std::vector<double>& inout, std::vector<double> const& in_padded, double multiplier,
    LocalUnitSquareGrid const& local_grid);
void add_mult_sinout_padded(std::vector<double> const& in, std::vector<double>& inout_padded, double multiplier,
    LocalUnitSquareGrid const& local_grid);
void matvec_top_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);
void matvec_bottom_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);
void matvec_left_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);
void matvec_right_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);
void matvec_topleft_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);
void matvec_topright_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);
void matvec_bottomright_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);
void matvec_bottomleft_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid);

void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc,
    LocalUnitSquareGrid const& local_grid, MPI_Comm comm_cart, const double tol, bool verbose) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);

    // allocate vectors u, r, Ap
    std::size_t size_loc = local_grid.Nx * local_grid.Ny;
    u_loc.resize(size_loc);
    std::vector<double> r_loc(size_loc), Ap_loc(size_loc);

    // get padded local size (including exchange with neighboring processes) 
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    std::size_t Nyt = local_grid.Ny + local_grid.has_bottom_neighbor + local_grid.has_top_neighbor;
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
    int top, bottom, left, right;
    get_neighbor_ranks(top, bottom, left, right, comm_cart);

    // define "column" type for communication with left/right neighbors
    MPI_Datatype col_type;
    MPI_Type_vector(local_grid.Ny, 1, Nxt, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    while (!converged) {
        /*
        1st step:
        Start exchange of pk. Communication is only initialized if
        a valid rank is specified, i.e. if dest/source != MPI_PROC_NULL.
        */
        std::vector<MPI_Request> send_requests(4, MPI_REQUEST_NULL);
        std::vector<MPI_Request> recv_requests(4, MPI_REQUEST_NULL);

        // top
        double *sendbuf_top = p_loc_padded.data() + (Nyt - 2) * Nxt + local_grid.has_left_neighbor;
        double *recvbuf_top = p_loc_padded.data() + (Nyt - 1) * Nxt + local_grid.has_left_neighbor;
        MPI_Isend(sendbuf_top, local_grid.Nx, MPI_DOUBLE, top, rank, comm_cart, &send_requests[Side::top]);
        MPI_Irecv(recvbuf_top, local_grid.Nx, MPI_DOUBLE, top, MPI_ANY_TAG, comm_cart, &recv_requests[Side::top]);

        // bottom
        double *sendbuf_bottom = p_loc_padded.data() + Nxt + local_grid.has_left_neighbor;
        double *recvbuf_bottom = p_loc_padded.data() + local_grid.has_left_neighbor;
        MPI_Isend(sendbuf_bottom, local_grid.Nx, MPI_DOUBLE, bottom, rank, comm_cart, &send_requests[Side::bottom]);
        MPI_Irecv(recvbuf_bottom, local_grid.Nx, MPI_DOUBLE, bottom, MPI_ANY_TAG, comm_cart, &recv_requests[Side::bottom]);

        // left
        double *sendbuf_left = p_loc_padded.data() + local_grid.has_bottom_neighbor * Nxt + 1;
        double *recvbuf_left = p_loc_padded.data() + local_grid.has_bottom_neighbor * Nxt;
        MPI_Isend(sendbuf_left, 1, col_type, left, rank, comm_cart, &send_requests[Side::left]);
        MPI_Irecv(recvbuf_left, 1, col_type, left, MPI_ANY_TAG, comm_cart, &recv_requests[Side::left]);

        // right
        double *sendbuf_right = p_loc_padded.data() + (1 + local_grid.has_bottom_neighbor) * Nxt - 2;
        double *recvbuf_right = p_loc_padded.data() + (1 + local_grid.has_bottom_neighbor) * Nxt - 1;
        MPI_Isend(sendbuf_right, 1, col_type, right, rank, comm_cart, &send_requests[Side::right]);
        MPI_Irecv(recvbuf_right, 1, col_type, right, MPI_ANY_TAG, comm_cart, &recv_requests[Side::right]);

        /*
        2nd step:
        Compute A * pk locally.
        */
        // 2.1: matvec for all "inner nodes" that do not require any data exchange
        MPI_Waitall(4, recv_requests.data(), MPI_STATUS_IGNORE);
        std::fill(Ap_loc.begin(), Ap_loc.end(), 0.0);
        matvec(A_loc, p_loc_padded, Ap_loc);

        // 2.2: matvec for boundary nodes, skipping corner points that have two neighboring processes 
        // int side;
        // for (int i = 0; i < 4; ++i) {
        //     MPI_Waitany(4, recv_requests.data(), &side, MPI_STATUS_IGNORE);
        //     if (side == Side::top) matvec_top_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        //     else if (side == Side::bottom) matvec_bottom_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        //     else if (side == Side::left) matvec_left_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        //     else if (side == Side::right) matvec_right_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        // }

        // // compute matvec for boundary points
        // bool computation_done[4];
        // if (!local_grid.has_left_neighbor || !local_grid.has_top_neighbor) {computation_done[CornerPoints::topleft] = true;
        // if (!local_grid.has_top_neighbor || !local_grid.has_right_neighbor) computation_done[CornerPoints::topright] = true;
        // if (!local_grid.has_right_neighbor || !local_grid.has_bottom_neighbor) computation_done[CornerPoints::bottomright] = true;
        // if (!local_grid.has_bottom_neighbor || !local_grid.has_left_neighbor) computation_done[CornerPoints::bottomleft] = true;

        // int recv_done[4] = {false}, receive_done_count;
        // bool all_computations_done = false;
        // while (!all_computations_done) {
        //     MPI_Testsome(4, recv_requests.data(), &receive_done_count, recv_done, MPI_STATUSES_IGNORE);
        //     if (!computation_done[CornerPoints::topleft] && recv_done[CornerPoints::topleft]) {
        //         matvec_topleft_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
        //     };
        //     if (!computation_done[CornerPoints::topright] && recv_done[CornerPoints::topright]) {
        //         matvec_topright_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
        //     };
        //     if (!computation_done[CornerPoints::bottomright] && recv_done[CornerPoints::bottomright]) {
        //         matvec_bottomright_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
        //     };
        //     if (!computation_done[CornerPoints::bottomleft] && recv_done[CornerPoints::bottomleft]) {
        //         matvec_bottomleft_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
        //     };

        //     // if all receives are done all corner points 
        //     all_computations_done = (receive_done_count == 4);
        // }

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
        Compute uk+1 = uk + alphak * pk.
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
        MPI_Waitall(4, send_requests.data(), MPI_STATUS_IGNORE);
        add_mult_sinout_padded(r_loc, p_loc_padded, gamma, local_grid);

        if (verbose && rank == 0 && counter % 100 == 0) {
            std::cout << "it " << counter << ": rr / bb = " << std::sqrt(rr / bb) << "\n";
        }
        ++counter;
        converged = (rr <= tol * tol * bb);
    }
}

void get_neighbor_ranks(int &top, int &bottom, int &left, int &right, MPI_Comm comm_cart) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::vector<int> neighbor_ranks(4, -1); // top / bottom / left / right order
    MPI_Neighbor_allgather(&rank, 1, MPI_INT, neighbor_ranks.data(), 1, MPI_INT, comm_cart);
    // TODO: is the ordering implementation dependent?

    top = neighbor_ranks[0] >= 0 ? neighbor_ranks[0] : MPI_PROC_NULL;
    bottom = neighbor_ranks[1] >= 0 ? neighbor_ranks[1] : MPI_PROC_NULL;
    left = neighbor_ranks[2] >= 0 ? neighbor_ranks[2] : MPI_PROC_NULL;
    right = neighbor_ranks[3] >= 0 ? neighbor_ranks[3] : MPI_PROC_NULL;
}

void copy_b_loc_to_p_loc(std::vector<double> &p_loc, std::vector<double> const& b_loc,
    LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    for (std::size_t idx = 0; idx < local_grid.Nx; ++idx) {
        for (std::size_t idy = 0; idy < local_grid.Ny; ++idy) {
            int index = (local_grid.has_bottom_neighbor + idy) * Nxt + local_grid.has_left_neighbor + idx ;
            p_loc[index] = b_loc[idy * local_grid.Nx + idx];
        }
    }
}

double dot_padded(std::vector<double> const& not_padded, std::vector<double> const& padded,
    LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;

    double result = 0.0;
    std::size_t index;
    for (std::size_t row = 0; row < local_grid.Ny; ++row) {
        for (std::size_t col = 0; col < local_grid.Nx; ++col) {
            index = Nxt * (row + local_grid.has_bottom_neighbor) + local_grid.has_left_neighbor + col;
            result += padded[index] * not_padded[row * local_grid.Nx + col];
        }
    }
    return result;
}

void add_mult_finout_padded(std::vector<double>& inout, std::vector<double> const& in_padded,
    double multiplier, LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;

    std::size_t index;
    for (std::size_t row = 0; row < local_grid.Ny; ++row) {
        for (std::size_t col = 0; col < local_grid.Nx; ++col) {
            index = Nxt * (row + local_grid.has_bottom_neighbor) + local_grid.has_left_neighbor + col;
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
            index = Nxt * (row + local_grid.has_bottom_neighbor) + local_grid.has_left_neighbor + col;
            inout_padded[index] = in[local_grid.Nx * row + col] + multiplier * inout_padded[index];
        }
    }
}

void matvec_bottom_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    if (!local_grid.has_bottom_neighbor) return;

    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    std::size_t row_index_start, row_index_end, col_index, index_padded;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid
    for (std::size_t row = local_grid.has_left_neighbor; row < local_grid.Nx - local_grid.has_right_neighbor; ++row) {
        row_index_start = A_loc.row_index(row);
        row_index_end = A_loc.row_index(row + 1);

        for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
            col_index = A_loc.col_index(value_count);
            // compute index in padded vector from column index in matrix
            index_padded = col_index - Nxt;
            out[row] += A_loc.value(value_count) * in_padded[index_padded];
        }
    }
}

void matvec_top_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    if (!local_grid.has_top_neighbor) return;

    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    std::size_t row_index_start, row_index_end, col_index, index_padded;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid
    std::size_t first_row = local_grid.Nx * (local_grid.Ny - 1) + local_grid.has_left_neighbor;
    std::size_t final_row = local_grid.Nx * local_grid.Ny - 1 - local_grid.has_right_neighbor;
    for (std::size_t row = first_row; row < final_row; ++row) {
        row_index_start = A_loc.row_index(row);
        row_index_end = A_loc.row_index(row + 1);

        for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
            col_index = A_loc.col_index(value_count);
            // compute index in padded vector from column index in matrix
            index_padded = col_index - Nxt;
            out[row] += A_loc.value(value_count) * in_padded[index_padded];
        }
    }
}

void matvec_left_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    if (!local_grid.has_left_neighbor) return;

    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    std::size_t row_index_start, row_index_end, col_index, index_padded;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid

}

void matvec_right_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    if (!local_grid.has_right_neighbor) return;

    std::size_t Nxt = local_grid.Nx + local_grid.has_left_neighbor + local_grid.has_right_neighbor;
    std::size_t row_index_start, row_index_end, col_index, index_padded;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid
}

void matvec_topleft_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {

    }
void matvec_topright_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
        
    }
void matvec_bottomright_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {

    }
void matvec_bottomleft_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {

    }