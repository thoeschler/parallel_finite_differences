#include "cg.hpp"
#include "la.hpp"

#include <assert.h>

enum Side { top = 0, bottom = 1, left = 2, right = 3 };

void get_neighbor_ranks(int &top, int &bottom, int &left, int &right, MPI_Comm comm_cart);
void copy_b_loc_to_p_loc(std::vector<double> &p_loc, std::vector<double> const& b_loc, LocalUnitSquareGrid const& local_grid);
double dot_padded(std::vector<double> const& not_padded, std::vector<double> const& padded, LocalUnitSquareGrid const& local_grid);
void add_mult_finout_padded(std::vector<double>& inout, std::vector<double> const& in_padded, double multiplier,
    LocalUnitSquareGrid const& local_grid);
void add_mult_sinout_padded(std::vector<double> const& in, std::vector<double>& inout_padded, double multiplier,
    LocalUnitSquareGrid const& local_grid);
void matvec_inner(CRSMatrix const&A_loc, std::vector<double> const&in_padded, std::vector<double> &out,
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
void cg_matvec_blocking(CRSMatrix const&A_loc, std::vector<double> &Ap_loc, std::vector<double> &p_loc_padded,
    LocalUnitSquareGrid const& local_grid, std::vector<MPI_Request> &send_requests, std::vector<MPI_Request> &recv_requests,
    MPI_Comm comm_cart, MPI_Datatype &col_type, int top, int bottom, int left, int right);
void cg_matvec_point_to_point(CRSMatrix const&A_loc, std::vector<double> &Ap_loc, std::vector<double> &p_loc_padded,
    LocalUnitSquareGrid const& local_grid, std::vector<MPI_Request> &send_requests, std::vector<MPI_Request> &recv_requests,
    MPI_Comm comm_cart, MPI_Datatype &col_type, int top, int bottom, int left, int right);
void cg_matvec_one_sided(CRSMatrix const&A_loc, std::vector<double> &Ap_loc, std::vector<double> &p_loc_padded,
    LocalUnitSquareGrid const& local_grid, std::vector<MPI_Request> &get_requests, MPI_Comm comm_cart, MPI_Datatype &col_type,
    MPI_Datatype &col_type_left, MPI_Datatype &col_type_right, MPI_Win &window, MPI_Group const&get_group,
    std::vector<int> const&Nx_neighbors, std::vector<int> const&Ny_neighbors, int top, int bottom, int left, int right);

/**
 * @brief Parallel Conjugate Gradient (CG) Method.
 * 
 * Different ways of communication are possible, namely:
 * 1) "Blocking" communication (Matrix vector product is computed only once the communication is done).
 * 2) Nonblocking point to point communication using Isend/Irecv.
 * 3) Onesided communication.
 * 
 * For options 2) and 3) the part of the matrix vector product which does not require any communication
 * is computed during data exchange.
 * 
 * @param A_loc Local matrix.
 * @param b_loc Local right hand side.
 * @param u_loc Solution vector.
 * @param local_grid Local UnitSquareGrid holding grid and neighbor information.
 * @param comm_cart MPI communicator.
 * @param tol Error tolerance.
 * @param verbose Print status of CG iteration.
 */
void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc,
    LocalUnitSquareGrid const& local_grid, MPI_Comm comm_cart, const double tol, bool verbose) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);

    // allocate vectors u, r, Ap
    std::size_t size_loc = local_grid.Nx * local_grid.Ny;
    u_loc.resize(size_loc);
    std::vector<double> r_loc(size_loc), Ap_loc(size_loc);

    /*
    Padded local size (to include exchange with neighboring processes).
    Padding is applied in every direction no matter if a neighboring
    process exists or not to make indexing simpler.
    */
    std::size_t Nxt = local_grid.Nx + 2;
    std::size_t Nyt = local_grid.Ny + 2;
    std::size_t size_loc_padded = Nxt * Nyt;

    /*
    In p_loc_padded data with neighboring processes is exchanged,
    so it is allocated using the padded size.
    */
    std::vector<double> p_loc_padded(size_loc_padded);

    // 0th step. Compute p = r = b - Ax0, initial guess is always x0=0 here.
    r_loc = b_loc;
    // copy data from non-padded (b_loc) to padded vector (p_loc_padded)
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

    /*
    Only for option 3).
    The neighboring local grid sizes must be known for
    onesided communication.
    */
    std::vector<int> Nx_neighbors(4), Ny_neighbors(4);
    MPI_Neighbor_allgather(&local_grid.Nx, 1, MPI_INT, Nx_neighbors.data(), 1, MPI_INT, comm_cart);
    MPI_Neighbor_allgather(&local_grid.Ny, 1, MPI_INT, Ny_neighbors.data(), 1, MPI_INT, comm_cart);

    // define "column" type for communication with left/right neighbors
    MPI_Datatype col_type;
    MPI_Type_vector(local_grid.Ny, 1, Nxt, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);
    
    /*
    Only for option 3).
    The local grid size of left and right neighbors may be different,
    so separate data types are needed here.
    */
    MPI_Datatype col_type_left, col_type_right;
    MPI_Type_vector(local_grid.Ny, 1, Nx_neighbors[Side::left] + 2, MPI_DOUBLE, &col_type_left);
    MPI_Type_vector(local_grid.Ny, 1, Nx_neighbors[Side::right] + 2, MPI_DOUBLE, &col_type_right);
    MPI_Type_commit(&col_type_left);
    MPI_Type_commit(&col_type_right);

    // only for option 3)
    MPI_Win window;
    int disp_unit = sizeof(double);
    MPI_Aint win_size = p_loc_padded.size() * disp_unit;
    MPI_Win_create(p_loc_padded.data(), win_size, disp_unit, MPI_INFO_NULL, comm_cart, &window);

    MPI_Group get_group;
    std::vector<int> group_ranks;
    if (top != MPI_PROC_NULL) group_ranks.push_back(top);
    if (bottom != MPI_PROC_NULL) group_ranks.push_back(bottom);
    if (left != MPI_PROC_NULL) group_ranks.push_back(left);
    if (right != MPI_PROC_NULL) group_ranks.push_back(right);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, group_ranks.size(), group_ranks.data(), &get_group);

    double start = MPI_Wtime();
    while (!converged) {
        /*
        1st and 2nd step:
        Exchange data from p_loc and compute matvec product locally.
        */
        // only for options 1) and 2)
        // std::vector<MPI_Request> send_requests(4, MPI_REQUEST_NULL);
        // std::vector<MPI_Request> recv_requests(4, MPI_REQUEST_NULL);

        // only for option 3)
        std::vector<MPI_Request> get_requests(4, MPI_REQUEST_NULL);

        // // 1) "Blocking" communication
        // cg_matvec_blocking(A_loc, Ap_loc, p_loc_padded, local_grid, send_requests, recv_requests, comm_cart, col_type,
        //         top, bottom, left, right);
        // // 2) Point to point communication
        // cg_matvec_point_to_point(A_loc, Ap_loc, p_loc_padded, local_grid, send_requests, recv_requests, comm_cart, col_type,
        //         top, bottom, left, right);
        // 3) Onesided communication
        cg_matvec_one_sided(A_loc, Ap_loc, p_loc_padded, local_grid, get_requests, comm_cart, col_type, col_type_left, col_type_right, 
            window, get_group, Nx_neighbors, Ny_neighbors, top, bottom, left, right);

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
        // only for options 1) and 2)
        // MPI_Waitall(4, send_requests.data(), MPI_STATUS_IGNORE);
        add_mult_sinout_padded(r_loc, p_loc_padded, gamma, local_grid);

        // info
        if (verbose && rank == 0 && counter % 100 == 0) {
            std::cout << "it " << counter << ": rr / bb = " << std::sqrt(rr / bb) << "\n";
        }
        ++counter;
        converged = (rr <= tol * tol * bb);
    }
    double end = MPI_Wtime();
    double avg_time = counter == 0 ? 0.0: (end - start) / counter;
    if (rank == 0) std::cout << "average time / it: " << avg_time << "s\n";

    MPI_Type_free(&col_type);
    // only for option 3)
    MPI_Win_free(&window);
    MPI_Type_free(&col_type_left);
    MPI_Type_free(&col_type_right);
}

void cg_matvec_blocking(CRSMatrix const&A_loc, std::vector<double> &Ap_loc, std::vector<double> &p_loc_padded,
        LocalUnitSquareGrid const& local_grid, std::vector<MPI_Request> &send_requests, std::vector<MPI_Request> &recv_requests,
        MPI_Comm comm_cart, MPI_Datatype &col_type, int top, int bottom, int left, int right) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::size_t Nxt = local_grid.Nx + 2;
    std::size_t Nyt = local_grid.Ny + 2;
    /*
    1st step:
    Start exchange of pk. Communication is only initialized if
    a valid rank is specified, i.e. if dest/source != MPI_PROC_NULL.
    */
    // top
    double *sendbuf_top = p_loc_padded.data() + (Nyt - 2) * Nxt + 1;
    double *recvbuf_top = p_loc_padded.data() + (Nyt - 1) * Nxt + 1;
    MPI_Isend(sendbuf_top, local_grid.Nx, MPI_DOUBLE, top, rank, comm_cart, &send_requests[Side::top]);
    MPI_Irecv(recvbuf_top, local_grid.Nx, MPI_DOUBLE, top, MPI_ANY_TAG, comm_cart, &recv_requests[Side::top]);

    // bottom
    double *sendbuf_bottom = p_loc_padded.data() + Nxt + 1;
    double *recvbuf_bottom = p_loc_padded.data() + 1;
    MPI_Isend(sendbuf_bottom, local_grid.Nx, MPI_DOUBLE, bottom, rank, comm_cart, &send_requests[Side::bottom]);
    MPI_Irecv(recvbuf_bottom, local_grid.Nx, MPI_DOUBLE, bottom, MPI_ANY_TAG, comm_cart, &recv_requests[Side::bottom]);

    // left
    double *sendbuf_left = p_loc_padded.data() + Nxt + 1;
    double *recvbuf_left = p_loc_padded.data() + Nxt;
    MPI_Isend(sendbuf_left, 1, col_type, left, rank, comm_cart, &send_requests[Side::left]);
    MPI_Irecv(recvbuf_left, 1, col_type, left, MPI_ANY_TAG, comm_cart, &recv_requests[Side::left]);

    // right
    double *sendbuf_right = p_loc_padded.data() + 2 * Nxt - 2;
    double *recvbuf_right = p_loc_padded.data() + 2 * Nxt - 1;
    MPI_Isend(sendbuf_right, 1, col_type, right, rank, comm_cart, &send_requests[Side::right]);
    MPI_Irecv(recvbuf_right, 1, col_type, right, MPI_ANY_TAG, comm_cart, &recv_requests[Side::right]);

    /*
    2nd step:
    Compute A * pk locally. 
    */
    // matvec for all nodes after synchronization
    std::fill(Ap_loc.begin(), Ap_loc.end(), 0.0);

    MPI_Waitall(4, recv_requests.data(), MPI_STATUSES_IGNORE);
    matvec(A_loc, p_loc_padded, Ap_loc);
}

void cg_matvec_point_to_point(CRSMatrix const&A_loc, std::vector<double> &Ap_loc, std::vector<double> &p_loc_padded,
        LocalUnitSquareGrid const& local_grid, std::vector<MPI_Request> &send_requests, std::vector<MPI_Request> &recv_requests,
        MPI_Comm comm_cart, MPI_Datatype &col_type, int top, int bottom, int left, int right) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::size_t Nxt = local_grid.Nx + 2;
    std::size_t Nyt = local_grid.Ny + 2;
    /*
    1st step:
    Start exchange of pk. Communication is only initialized if
    a valid rank is specified, i.e. if dest/source != MPI_PROC_NULL.
    */
    // top
    double *sendbuf_top = p_loc_padded.data() + (Nyt - 2) * Nxt + 1;
    double *recvbuf_top = p_loc_padded.data() + (Nyt - 1) * Nxt + 1;
    MPI_Isend(sendbuf_top, local_grid.Nx, MPI_DOUBLE, top, rank, comm_cart, &send_requests[Side::top]);
    MPI_Irecv(recvbuf_top, local_grid.Nx, MPI_DOUBLE, top, MPI_ANY_TAG, comm_cart, &recv_requests[Side::top]);

    // bottom
    double *sendbuf_bottom = p_loc_padded.data() + Nxt + 1;
    double *recvbuf_bottom = p_loc_padded.data() + 1;
    MPI_Isend(sendbuf_bottom, local_grid.Nx, MPI_DOUBLE, bottom, rank, comm_cart, &send_requests[Side::bottom]);
    MPI_Irecv(recvbuf_bottom, local_grid.Nx, MPI_DOUBLE, bottom, MPI_ANY_TAG, comm_cart, &recv_requests[Side::bottom]);

    // left
    double *sendbuf_left = p_loc_padded.data() + Nxt + 1;
    double *recvbuf_left = p_loc_padded.data() + Nxt;
    MPI_Isend(sendbuf_left, 1, col_type, left, rank, comm_cart, &send_requests[Side::left]);
    MPI_Irecv(recvbuf_left, 1, col_type, left, MPI_ANY_TAG, comm_cart, &recv_requests[Side::left]);

    // right
    double *sendbuf_right = p_loc_padded.data() + 2 * Nxt - 2;
    double *recvbuf_right = p_loc_padded.data() + 2 * Nxt - 1;
    MPI_Isend(sendbuf_right, 1, col_type, right, rank, comm_cart, &send_requests[Side::right]);
    MPI_Irecv(recvbuf_right, 1, col_type, right, MPI_ANY_TAG, comm_cart, &recv_requests[Side::right]);

    /*
    2nd step:
    Compute A * pk locally.
    */
    // 2.1: matvec for all "inner nodes" (those which do not require any data exchange)
    std::fill(Ap_loc.begin(), Ap_loc.end(), 0.0);
    matvec_inner(A_loc, p_loc_padded, Ap_loc, local_grid);

    // 2.2: matvec for boundary nodes, skipping corner points that have two neighboring processes
    int side;
    for (int i = 0; i < 4; ++i) {
        MPI_Waitany(4, recv_requests.data(), &side, MPI_STATUS_IGNORE);
        if (side == Side::top) matvec_top_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        else if (side == Side::bottom) matvec_bottom_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        else if (side == Side::left) matvec_left_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        else if (side == Side::right) matvec_right_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
    }

    // 2.3: compute matvec for corner points
    matvec_topleft_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
    matvec_topright_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
    matvec_bottomright_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
    matvec_bottomleft_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
}

void cg_matvec_one_sided(CRSMatrix const&A_loc, std::vector<double> &Ap_loc, std::vector<double> &p_loc_padded,
        LocalUnitSquareGrid const& local_grid, std::vector<MPI_Request> &get_requests, MPI_Comm comm_cart,
        MPI_Datatype &col_type, MPI_Datatype &col_type_left, MPI_Datatype &col_type_right, MPI_Win &window, MPI_Group const&get_group,
        std::vector<int> const&Nx_neighbors, std::vector<int> const&Ny_neighbors, int top, int bottom, int left, int right) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::size_t Nxt = local_grid.Nx + 2;
    std::size_t Nyt = local_grid.Ny + 2;

    // top
    MPI_Aint targetdisp_top = Nxt + 1;
    double *originbuf_top = p_loc_padded.data() + (Nyt - 1) * Nxt + 1;

    // bottom
    MPI_Aint targetdisp_bottom = Ny_neighbors[Side::bottom] * Nxt + 1;
    double *originbuf_bottom = p_loc_padded.data() + 1;

    // left
    MPI_Aint targetdisp_left = 2 * (Nx_neighbors[Side::left] + 2) - 2;
    double *originbuf_left = p_loc_padded.data() + Nxt;

    // right
    MPI_Aint targetdisp_right = (Nx_neighbors[Side::right] + 2) + 1;
    double *originbuf_right = p_loc_padded.data() + 2 * Nxt - 1;

    /*
    1st step:
    Start exchange of pk. Communication is only initialized if
    a valid rank is specified, i.e. if dest/source != MPI_PROC_NULL.
    */
    // p_loc_padded is used for both sending and receiving data 
    MPI_Win_post(get_group, 0, window);
    MPI_Win_start(get_group, 0, window);

    MPI_Rget(originbuf_top, local_grid.Nx, MPI_DOUBLE, top, targetdisp_top, local_grid.Nx, MPI_DOUBLE, window, &get_requests[Side::top]);
    MPI_Rget(originbuf_bottom, local_grid.Nx, MPI_DOUBLE, bottom, targetdisp_bottom, local_grid.Nx, MPI_DOUBLE, window, &get_requests[Side::bottom]);
    MPI_Rget(originbuf_left, 1, col_type, left, targetdisp_left, 1, col_type_left, window, &get_requests[Side::left]);
    MPI_Rget(originbuf_right, 1, col_type, right, targetdisp_right, 1, col_type_right, window, &get_requests[Side::right]);

    /*
    2nd step:
    Compute A * pk locally.
    */
    // 2.1: matvec for all "inner nodes" (those which do not require any data exchange)
    std::fill(Ap_loc.begin(), Ap_loc.end(), 0.0);
    matvec_inner(A_loc, p_loc_padded, Ap_loc, local_grid);

    MPI_Win_complete(window);
    MPI_Win_wait(window);

    // 2.2: matvec for boundary nodes, skipping corner points that have two neighboring processes
    int side;
    for (int i = 0; i < 4; ++i) {
        MPI_Waitany(4, get_requests.data(), &side, MPI_STATUS_IGNORE);
        if (side == Side::top) matvec_top_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        else if (side == Side::bottom) matvec_bottom_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        else if (side == Side::left) matvec_left_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
        else if (side == Side::right) matvec_right_boundary(A_loc, p_loc_padded, Ap_loc, local_grid);
    }

    // 2.3: compute matvec for corner points
    matvec_topleft_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
    matvec_topright_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
    matvec_bottomright_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
    matvec_bottomleft_corner(A_loc, p_loc_padded, Ap_loc, local_grid);
}

void get_neighbor_ranks(int &top, int &bottom, int &left, int &right, MPI_Comm comm_cart) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::vector<int> neighbor_ranks(4, -1); // top / bottom / left / right order
    MPI_Neighbor_allgather(&rank, 1, MPI_INT, neighbor_ranks.data(), 1, MPI_INT, comm_cart);
    // TODO: is the ordering implementation dependent?

    top = neighbor_ranks[Side::top] >= 0 ? neighbor_ranks[Side::top] : MPI_PROC_NULL;
    bottom = neighbor_ranks[Side::bottom] >= 0 ? neighbor_ranks[Side::bottom] : MPI_PROC_NULL;
    left = neighbor_ranks[Side::left] >= 0 ? neighbor_ranks[Side::left] : MPI_PROC_NULL;
    right = neighbor_ranks[Side::right] >= 0 ? neighbor_ranks[Side::right] : MPI_PROC_NULL;
}

/**
 * @brief Copy data from padded vecor p_loc_padded to non-padded vector b_loc.
 * 
 * @param p_loc_padded Padded vector.
 * @param b_loc Non-padded vector.
 * @param local_grid Local UnitSquareGrid.
 */
void copy_b_loc_to_p_loc(std::vector<double> &p_loc_padded, std::vector<double> const& b_loc,
    LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + 2;
    for (std::size_t idx = 0; idx < local_grid.Nx; ++idx) {
        for (std::size_t idy = 0; idy < local_grid.Ny; ++idy) {
            int index = (idy + 1) * Nxt + idx + 1;
            p_loc_padded[index] = b_loc[idy * local_grid.Nx + idx];
        }
    }
}

/**
 * @brief Dot product between non-padded and padded vector.
 * 
 * @param not_padded Non-padded vector.
 * @param padded Padded vector.
 * @param local_grid Local UnitSquareGrid.
 * @return double 
 */
double dot_padded(std::vector<double> const& not_padded, std::vector<double> const& padded,
    LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + 2;

    double result = 0.0;
    std::size_t index;
    #pragma omp parallel shared(not_padded, padded) private(index)
    {
    #pragma omp for reduction(+:result)
        for (std::size_t row = 0; row < local_grid.Ny; ++row) {
            for (std::size_t col = 0; col < local_grid.Nx; ++col) {
                index = Nxt * (row + 1) + col + 1;
                result += padded[index] * not_padded[row * local_grid.Nx + col];
            }
        }
    }
    return result;
}

/**
 * @brief Combined addition and multiplication between non-padded and padded vector.
 * 
 * This routine overwrites the first input vector.
 * 
 * @param inout Non-padded vector.
 * @param in_padded Padded vector.
 * @param multiplier Multiplier.
 * @param local_grid Local UnitSquareGrid.
 */
void add_mult_finout_padded(std::vector<double>& inout, std::vector<double> const& in_padded,
    double multiplier, LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + 2;

    std::size_t index;
    for (std::size_t row = 0; row < local_grid.Ny; ++row) {
        for (std::size_t col = 0; col < local_grid.Nx; ++col) {
            index = Nxt * (row + 1) + col + 1;
            inout[row * local_grid.Nx + col] += multiplier * in_padded[index];
        }
    }
}

/**
 * @brief Combined addition and multiplication between non-padded and padded vector.
 * 
 * This routine overwrites the second input vector.
 * 
 * @param inout Non-padded vector.
 * @param in_padded Padded vector.
 * @param multiplier Multiplier.
 * @param local_grid Local UnitSquareGrid.
 */
void add_mult_sinout_padded(std::vector<double> const& in, std::vector<double>& inout_padded, double multiplier,
    LocalUnitSquareGrid const& local_grid) {
    std::size_t Nxt = local_grid.Nx + 2;

    std::size_t index;
    for (std::size_t row = 0; row < local_grid.Ny; ++row) {
        for (std::size_t col = 0; col < local_grid.Nx; ++col) {
            index = Nxt * (row + 1) + col + 1;
            inout_padded[index] = in[local_grid.Nx * row + col] + multiplier * inout_padded[index];
        }
    }
}

/**
 * @brief Matrix vector product for "inner" grid points.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_inner(CRSMatrix const&A_loc, std::vector<double> const&in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    // row refers to the row of the matrix, not to the row in the local grid
    // the indices refer to the local grid, not the padded data
    std::size_t row, row_index_start, row_index_end;
    #pragma omp parallel for shared(out) private(row, row_index_start, row_index_end)
    for (std::size_t idy = local_grid.has_bottom_neighbor; idy < local_grid.Ny - local_grid.has_top_neighbor; ++idy) {
        for (std::size_t idx = local_grid.has_left_neighbor; idx < local_grid.Nx - local_grid.has_right_neighbor; ++idx) {
            row = local_grid.Nx * idy + idx; // row in the matrix
            row_index_start = A_loc.row_index(row);
            row_index_end = A_loc.row_index(row + 1);

            for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
                out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
            }
        }
    }
}

/**
 * @brief Matrix vector product for bottom boundary grid points.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_bottom_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    // if there is no bottom neighbor the bottom part is part of the "inner" nodes
    if (!local_grid.has_bottom_neighbor) return;

    std::size_t row_index_start, row_index_end;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid
    #pragma omp parallel for private(row_index_start, row_index_end)
    for (std::size_t row = local_grid.has_left_neighbor; row < local_grid.Nx - local_grid.has_right_neighbor; ++row) {
        row_index_start = A_loc.row_index(row);
        row_index_end = A_loc.row_index(row + 1);

        for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
            out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
        }
    }
}

/**
 * @brief Matrix vector product for top boundary grid points.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_top_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
        // if there is no top neighbor the top part is part of the "inner" nodes
    if (!local_grid.has_top_neighbor) return;

    std::size_t row_index_start, row_index_end;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid
    std::size_t first_row = local_grid.Nx * (local_grid.Ny - 1) + local_grid.has_left_neighbor;
    std::size_t final_row = local_grid.Nx * local_grid.Ny - local_grid.has_right_neighbor;
    #pragma omp parallel for private(row_index_start, row_index_end)
    for (std::size_t row = first_row; row < final_row; ++row) {
        row_index_start = A_loc.row_index(row);
        row_index_end = A_loc.row_index(row + 1);

        for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
            out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
        }
    }
}

/**
 * @brief Matrix vector product for left boundary grid points.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_left_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    // if there is no left neighbor the left part is part of the "inner" nodes
    if (!local_grid.has_left_neighbor) return;

    std::size_t row, row_index_start, row_index_end;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid
    #pragma omp parallel for private(row, row_index_start, row_index_end)
    for (std::size_t idy = local_grid.has_bottom_neighbor; idy < local_grid.Ny - local_grid.has_top_neighbor; ++idy) {
        row = local_grid.Nx * idy;
        row_index_start = A_loc.row_index(row);
        row_index_end = A_loc.row_index(row + 1);

        for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
            out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
        }
    }
}

/**
 * @brief Matrix vector product for right boundary grid points.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_right_boundary(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    // if there is no right neighbor the right part is part of the "inner" nodes
    if (!local_grid.has_right_neighbor) return;

    std::size_t row, row_index_start, row_index_end;
    // skip corner points if they require two data exchanges
    // row refers to the row of the matrix, not to the row in the local grid
    #pragma omp parallel for private(row, row_index_start, row_index_end)
    for (std::size_t idy = local_grid.has_bottom_neighbor; idy < local_grid.Ny - local_grid.has_top_neighbor; ++idy) {
        row = local_grid.Nx * (idy + 1) - 1;
        row_index_start = A_loc.row_index(row);
        row_index_end = A_loc.row_index(row + 1);

        for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
            out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
        }
    }
}

/**
 * @brief Matrix vector product for top left corner grid point.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_topleft_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    // if no left or top neighbor exist, the "corner" point has already been accounted for
    if (!local_grid.has_left_neighbor || !local_grid.has_top_neighbor) return;

    std::size_t row = local_grid.Nx * (local_grid.Ny - 1);
    std::size_t row_index_start = A_loc.row_index(row);
    std::size_t row_index_end = A_loc.row_index(row + 1);

    for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
        out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
    }
}

/**
 * @brief Matrix vector product for top right corner grid point.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_topright_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    // if no top or right neighbor exist, the "corner" point has already been accounted for
    if (!local_grid.has_top_neighbor || !local_grid.has_right_neighbor) return;

    std::size_t row = local_grid.Nx * local_grid.Ny - 1;
    std::size_t row_index_start = A_loc.row_index(row);
    std::size_t row_index_end = A_loc.row_index(row + 1);

    for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
        out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
    }
}

/**
 * @brief Matrix vector product for bottom right corner grid point.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_bottomright_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    if (!local_grid.has_right_neighbor || !local_grid.has_bottom_neighbor) return;
    // if no right or bottom neighbor exist, the "corner" point has already been accounted for

    std::size_t row = local_grid.Nx - 1;
    std::size_t row_index_start = A_loc.row_index(row);
    std::size_t row_index_end = A_loc.row_index(row + 1);

    for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
        out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
    }

}

/**
 * @brief Matrix vector product for bottom left corner grid point.
 * 
 * @param A_loc Local Finite Difference Matrix.
 * @param in_padded Padded vector.
 * @param out Result.
 * @param local_grid Local UnitSquareGrid.
 */
void matvec_bottomleft_corner(CRSMatrix const&A_loc, std::vector<double> const& in_padded, std::vector<double> &out,
    LocalUnitSquareGrid const& local_grid) {
    // if no bottom or left neighbor exist, the "corner" point has already been accounted for
    if (!local_grid.has_bottom_neighbor || !local_grid.has_left_neighbor) return;

    std::size_t row = 0;
    std::size_t row_index_start = A_loc.row_index(row);
    std::size_t row_index_end = A_loc.row_index(row + 1);

    for (std::size_t value_count = row_index_start; value_count < row_index_end; ++value_count) {
        out[row] += A_loc.value(value_count) * in_padded[A_loc.col_index(value_count)];
    }
}
