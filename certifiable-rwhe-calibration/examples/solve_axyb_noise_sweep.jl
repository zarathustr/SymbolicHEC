using LinearAlgebra
using Printf
using Random
using Rotations
using Statistics
using JuMP

include(joinpath(@__DIR__, "..", "src", "utils", "rotation_noise.jl"))
include(joinpath(@__DIR__, "..", "src", "calibration", "robot_world_costs.jl"))
include(joinpath(@__DIR__, "..", "src", "rotation_sdp_solver_jump.jl"))

const DEFAULT_LOOPS = 3
const DEFAULT_NOISES = [0.0, 0.01, 0.03]
const DEFAULT_NUM_PAIRS = 6

function print_usage()
    println("Usage:")
    println("  julia --project=. examples/solve_axyb_noise_sweep.jl [--loops N] [--pairs N] [--noises s1,s2,...]")
    println()
    println("Options:")
    println("  --loops    Number of random calibration problems to solve per noise level.")
    println("  --pairs    Number of SE(3) measurement pairs per random calibration problem.")
    println("  --noises   Comma-separated noise scales. Each scale is used as both")
    println("             rotation-angle std-dev (radians) and translation std-dev.")
end

function parse_cli(args::Vector{String})
    loops = DEFAULT_LOOPS
    num_pairs = DEFAULT_NUM_PAIRS
    noises = copy(DEFAULT_NOISES)

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            print_usage()
            exit(0)
        elseif arg == "--loops"
            i += 1
            i > length(args) && error("Missing value after --loops")
            loops = parse(Int, args[i])
        elseif startswith(arg, "--loops=")
            loops = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--pairs" || arg == "--num-pairs"
            i += 1
            i > length(args) && error("Missing value after $arg")
            num_pairs = parse(Int, args[i])
        elseif startswith(arg, "--pairs=")
            num_pairs = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--num-pairs=")
            num_pairs = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--noises"
            i += 1
            i > length(args) && error("Missing value after --noises")
            noises = parse_noise_list(args[i])
        elseif startswith(arg, "--noises=")
            noises = parse_noise_list(split(arg, "=", limit=2)[2])
        else
            error("Unknown argument: $arg")
        end

        i += 1
    end

    loops > 0 || error("--loops must be positive")
    num_pairs > 0 || error("--pairs must be positive")
    isempty(noises) && error("--noises must contain at least one value")
    all(noise -> noise >= 0.0, noises) || error("--noises must be nonnegative")

    return loops, num_pairs, noises
end

function parse_noise_list(raw::AbstractString)
    tokens = filter(!isempty, strip.(split(raw, ",")))
    return parse.(Float64, tokens)
end

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

function noisy_pose(T::AbstractMatrix, noise_scale::Float64)
    if noise_scale == 0.0
        return copy(T)
    end

    R_noise = Matrix(random_rotation_sample_normal_magnitude(0.0, noise_scale))
    t_noise = noise_scale .* randn(3)
    return make_pose(R_noise * T[1:3, 1:3], T[1:3, 4] + t_noise)
end

function pose_from_solution(solution::AbstractVector, rotation_offset::Int, translation_offset::Int)
    R = reshape(solution[rotation_offset:rotation_offset + 8], 3, 3)
    t = solution[translation_offset:translation_offset + 2]
    return make_pose(R, t)
end

function rotation_error_deg(R_est::AbstractMatrix, R_gt::AbstractMatrix)
    ΔR = RotMatrix{3}(R_est * transpose(R_gt))
    return rad2deg(abs(rotation_angle(ΔR)))
end

function mean_pose_residual(A, B, X, Y)
    rot_residuals = zeros(size(A, 1))
    trans_residuals = zeros(size(A, 1))

    for i in axes(A, 1)
        residual = A[i, :, :] * X - Y * B[i, :, :]
        rot_residuals[i] = opnorm(residual[1:3, 1:3])
        trans_residuals[i] = norm(residual[1:3, 4])
    end

    return mean(rot_residuals), mean(trans_residuals)
end

function solve_trial(A, B, X_gt, Y_gt)
    Z = nothing
    model = nothing
    elapsed_sec = @elapsed begin
        weights = ones(size(A, 1))
        Q = sparse_robot_world_transformation_cost(A, B, weights, weights)
        Z, model = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                solve_sdp_dual_jump(Q, 2, true, true)
            end
        end
    end
    solution = vec(extract_solution_from_dual(Z))
    termination = JuMP.termination_status(model)
    status = string(termination)

    X_est = pose_from_solution(solution, 7, 1)
    Y_est = pose_from_solution(solution, 16, 4)
    x_rot_err = rotation_error_deg(X_est[1:3, 1:3], X_gt[1:3, 1:3])
    x_trans_err = norm(X_est[1:3, 4] - X_gt[1:3, 4])
    y_rot_err = rotation_error_deg(Y_est[1:3, 1:3], Y_gt[1:3, 1:3])
    y_trans_err = norm(Y_est[1:3, 4] - Y_gt[1:3, 4])
    rot_residual, trans_residual = mean_pose_residual(A, B, X_est, Y_est)

    return (
        solved = status == "OPTIMAL",
        status = status,
        x_rot_err = x_rot_err,
        x_trans_err = x_trans_err,
        y_rot_err = y_rot_err,
        y_trans_err = y_trans_err,
        rot_residual = rot_residual,
        trans_residual = trans_residual,
        solve_time_ms = 1000 * elapsed_sec,
    )
end

function make_clean_case(num_pairs::Int)
    X_gt = random_pose()
    Y_gt = random_pose()
    A_clean = zeros(num_pairs, 4, 4)
    B_clean = zeros(num_pairs, 4, 4)

    for i in 1:num_pairs
        B_clean[i, :, :] = random_pose()
        A_clean[i, :, :] = Y_gt * B_clean[i, :, :] * inv(X_gt)
    end

    return X_gt, Y_gt, A_clean, B_clean
end

function make_noisy_measurements(A_clean, B_clean, noise_scale::Float64)
    A_noisy = similar(A_clean)
    B_noisy = similar(B_clean)

    for i in axes(A_clean, 1)
        A_noisy[i, :, :] = noisy_pose(A_clean[i, :, :], noise_scale)
        B_noisy[i, :, :] = noisy_pose(B_clean[i, :, :], noise_scale)
    end

    return A_noisy, B_noisy
end

function warm_up_solver(num_pairs::Int)
    X_gt, Y_gt, A_clean, B_clean = make_clean_case(num_pairs)
    solve_trial(A_clean, B_clean, X_gt, Y_gt)
    return nothing
end

function summarize(results)
    solved = filter(result -> result.solved, results)
    if isempty(solved)
        return (
            success = "0/$(length(results))",
            x_rot_err = NaN,
            x_trans_err = NaN,
            y_rot_err = NaN,
            y_trans_err = NaN,
            rot_residual = NaN,
            trans_residual = NaN,
            solve_time_ms = NaN,
        )
    end

    return (
        success = "$(length(solved))/$(length(results))",
        x_rot_err = mean(getfield.(solved, :x_rot_err)),
        x_trans_err = mean(getfield.(solved, :x_trans_err)),
        y_rot_err = mean(getfield.(solved, :y_rot_err)),
        y_trans_err = mean(getfield.(solved, :y_trans_err)),
        rot_residual = mean(getfield.(solved, :rot_residual)),
        trans_residual = mean(getfield.(solved, :trans_residual)),
        solve_time_ms = mean(getfield.(solved, :solve_time_ms)),
    )
end

function main()
    loops, num_pairs, noises = parse_cli(ARGS)
    Random.seed!(11)
    warm_up_solver(num_pairs)

    results_by_noise = Dict(noise => NamedTuple[] for noise in noises)

    for loop_idx in 1:loops
        X_gt, Y_gt, A_clean, B_clean = make_clean_case(num_pairs)

        for noise in noises
            A_noisy, B_noisy = make_noisy_measurements(A_clean, B_clean, noise)
            trial_result = try
                solve_trial(A_noisy, B_noisy, X_gt, Y_gt)
            catch err
                (
                    solved = false,
                    status = sprint(showerror, err),
                    x_rot_err = NaN,
                    x_trans_err = NaN,
                    y_rot_err = NaN,
                    y_trans_err = NaN,
                    rot_residual = NaN,
                    trans_residual = NaN,
                    solve_time_ms = NaN,
                )
            end

            push!(results_by_noise[noise], trial_result)
            println(
                @sprintf(
                    "loop %d/%d, noise %.4f -> %s",
                    loop_idx,
                    loops,
                    noise,
                    trial_result.status,
                ),
            )
        end
    end

    println()
    println("Summary over $(loops) random problems with $(num_pairs) SE(3) pairs each")
    println("Noise scale is reused as rotation-angle std-dev [rad] and translation std-dev.")
    println()
    println(rpad("noise", 10),
            rpad("solved", 10),
            lpad("X rot err(deg)", 16),
            lpad("X trans err", 14),
            lpad("Y rot err(deg)", 16),
            lpad("Y trans err", 14),
            lpad("mean rot resid", 16),
            lpad("mean trans resid", 18),
            lpad("mean solve ms", 15))
    println(repeat("-", 129))

    for noise in noises
        summary = summarize(results_by_noise[noise])
        println(
            @sprintf(
                "%-10.4f%-10s%16.6f%14.6f%16.6f%14.6f%16.6e%18.6e%15.3f",
                noise,
                summary.success,
                summary.x_rot_err,
                summary.x_trans_err,
                summary.y_rot_err,
                summary.y_trans_err,
                summary.rot_residual,
                summary.trans_residual,
                summary.solve_time_ms,
            ),
        )
    end
end

main()
