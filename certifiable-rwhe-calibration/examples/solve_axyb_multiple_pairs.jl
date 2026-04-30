using LinearAlgebra
using Random
using Rotations

include(joinpath(@__DIR__, "..", "src", "calibration", "robot_world_costs.jl"))
include(joinpath(@__DIR__, "..", "src", "rotation_sdp_solver_jump.jl"))

function make_pose(R::AbstractMatrix, t::AbstractVector)
    T = Matrix{Float64}(I, 4, 4)
    T[1:3, 1:3] .= R
    T[1:3, 4] .= t
    return T
end

function random_pose(; translation_scale=0.5)
    R = Matrix(rand(RotMatrix{3}))
    t = translation_scale .* (2 .* rand(3) .- 1)
    return make_pose(R, t)
end

function pose_from_solution(solution::AbstractVector, rotation_offset::Int, translation_offset::Int)
    R = reshape(solution[rotation_offset:rotation_offset + 8], 3, 3)
    t = solution[translation_offset:translation_offset + 2]
    return make_pose(R, t)
end

function max_pose_residual(A, B, X, Y)
    max_rot_err = 0.0
    max_trans_err = 0.0

    for i in axes(A, 1)
        residual = A[i, :, :] * X - Y * B[i, :, :]
        max_rot_err = max(max_rot_err, opnorm(residual[1:3, 1:3]))
        max_trans_err = max(max_trans_err, norm(residual[1:3, 4]))
    end

    return max_rot_err, max_trans_err
end

function main()
    Random.seed!(7)

    num_pairs = 6

    X_gt = make_pose(
        Matrix(RotZ(0.35) * RotY(-0.25) * RotX(0.15)),
        [0.35, -0.20, 0.45],
    )
    Y_gt = make_pose(
        Matrix(RotX(-0.30) * RotZ(0.40) * RotY(0.20)),
        [-0.15, 0.30, 0.25],
    )

    A = zeros(num_pairs, 4, 4)
    B = zeros(num_pairs, 4, 4)

    for i in 1:num_pairs
        B[i, :, :] = random_pose()
        A[i, :, :] = Y_gt * B[i, :, :] * inv(X_gt)
    end

    kappa = ones(num_pairs)
    tau = ones(num_pairs)
    Q = sparse_robot_world_transformation_cost(A, B, kappa, tau)
    Z, model = solve_sdp_dual_jump(Q, 2, true, true)
    solution = vec(extract_solution_from_dual(Z))

    X_est = pose_from_solution(solution, 7, 1)
    Y_est = pose_from_solution(solution, 16, 4)

    rot_residual, trans_residual = max_pose_residual(A, B, X_est, Y_est)

    println("Solver status: ", JuMP.termination_status(model))
    println()
    println("Ground-truth X:")
    display(X_gt)
    println("Estimated X:")
    display(X_est)
    println()
    println("Ground-truth Y:")
    display(Y_gt)
    println("Estimated Y:")
    display(Y_est)
    println()
    println("Max rotation residual ||A_i * X - Y * B_i||_2: ", rot_residual)
    println("Max translation residual ||A_i * X - Y * B_i||_2: ", trans_residual)
end

main()
