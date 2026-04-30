using LinearAlgebra
using JuMP
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "calibration", "robot_world_costs.jl"))
include(joinpath(@__DIR__, "..", "src", "rotation_sdp_solver_jump.jl"))

const DEFAULT_SUCCESS_TOL = 1e-2

function print_usage()
    println("Usage:")
    println("  julia --project=. examples/solve_axyb_input_file.jl --input_meas PATH [--success_tol X]")
    println()
    println("Options:")
    println("  --input_meas   Path to an AXYB_LOOP_MEAS_V1 measurement file.")
    println("  --success_tol  Pose-error threshold for both X and Y; default $(DEFAULT_SUCCESS_TOL).")
end

function parse_cli(args::Vector{String})
    input_meas = ""
    success_tol = DEFAULT_SUCCESS_TOL

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help" || arg == "-h"
            print_usage()
            exit(0)
        elseif arg == "--input_meas" || arg == "--input"
            i += 1
            i > length(args) && error("Missing value after $arg")
            input_meas = args[i]
        elseif startswith(arg, "--input_meas=")
            input_meas = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--input=")
            input_meas = split(arg, "=", limit=2)[2]
        elseif arg == "--success_tol"
            i += 1
            i > length(args) && error("Missing value after --success_tol")
            success_tol = parse(Float64, args[i])
        elseif startswith(arg, "--success_tol=")
            success_tol = parse(Float64, split(arg, "=", limit=2)[2])
        else
            error("Unknown argument: $arg")
        end

        i += 1
    end

    isempty(input_meas) && error("--input_meas is required")
    success_tol >= 0.0 || error("--success_tol must be nonnegative")

    return input_meas, success_tol
end

function take_token!(tokens::Vector{SubString{String}}, index_ref::Base.RefValue{Int}, context::AbstractString)
    index_ref[] <= length(tokens) || error("Unexpected end of file while reading $context")
    token = tokens[index_ref[]]
    index_ref[] += 1
    return token
end

function expect_token!(tokens::Vector{SubString{String}}, index_ref::Base.RefValue{Int}, expected::AbstractString, context::AbstractString)
    token = take_token!(tokens, index_ref, context)
    token == expected || error("Expected '$expected' while reading $context, got '$token'")
    return nothing
end

function take_int!(tokens::Vector{SubString{String}}, index_ref::Base.RefValue{Int}, context::AbstractString)
    return parse(Int, take_token!(tokens, index_ref, context))
end

function take_float!(tokens::Vector{SubString{String}}, index_ref::Base.RefValue{Int}, context::AbstractString)
    return parse(Float64, take_token!(tokens, index_ref, context))
end

function take_matrix4!(tokens::Vector{SubString{String}}, index_ref::Base.RefValue{Int}, label::AbstractString)
    expect_token!(tokens, index_ref, label, label)
    values = [take_float!(tokens, index_ref, label) for _ in 1:16]
    return reshape(values, 4, 4)
end

function load_problem_sequence(path::AbstractString)
    text = read(path, String)
    tokens = split(text)
    index_ref = Ref(1)

    expect_token!(tokens, index_ref, "AXYB_LOOP_MEAS_V1", "file header")
    expect_token!(tokens, index_ref, "problems", "problem count")
    problem_count = take_int!(tokens, index_ref, "problem count")
    problem_count >= 0 || error("Invalid negative problem count")

    problems = Vector{NamedTuple}(undef, problem_count)
    for expected_index in 1:problem_count
        expect_token!(tokens, index_ref, "problem", "problem header")
        file_index = take_int!(tokens, index_ref, "problem index")
        file_index == expected_index || error("Expected problem index $expected_index, got $file_index")
        expect_token!(tokens, index_ref, "len", "problem length")
        len = take_int!(tokens, index_ref, "problem length")
        len >= 0 || error("Invalid negative problem length")

        expect_token!(tokens, index_ref, "g_ground_truth", "ground truth")
        g_ground_truth = [take_float!(tokens, index_ref, "ground truth") for _ in 1:6]
        X0 = take_matrix4!(tokens, index_ref, "X0")
        Y0 = take_matrix4!(tokens, index_ref, "Y0")

        A = zeros(len, 4, 4)
        B = zeros(len, 4, 4)
        for i in 1:len
            A[i, :, :] = take_matrix4!(tokens, index_ref, "A")
            B[i, :, :] = take_matrix4!(tokens, index_ref, "B")
        end

        expect_token!(tokens, index_ref, "end_problem", "problem terminator")
        problems[expected_index] = (
            g_ground_truth = g_ground_truth,
            X0 = X0,
            Y0 = Y0,
            A = A,
            B = B,
        )
    end

    expect_token!(tokens, index_ref, "end_file", "file terminator")
    return problems
end

function pose_from_solution(solution::AbstractVector, rotation_offset::Int, translation_offset::Int)
    T = Matrix{Float64}(I, 4, 4)
    T[1:3, 1:3] .= reshape(solution[rotation_offset:rotation_offset + 8], 3, 3)
    T[1:3, 4] .= solution[translation_offset:translation_offset + 2]
    return T
end

function pose_error(T_est::AbstractMatrix, T_gt::AbstractMatrix)
    return sum(abs2, T_est .- T_gt)
end

function mean_pose_residual(A, B, X, Y)
    residuals = zeros(size(A, 1))
    for i in axes(A, 1)
        residual = A[i, :, :] * X - Y * B[i, :, :]
        residuals[i] = sum(abs2, residual)
    end
    return mean(residuals)
end

function solve_problem(A, B)
    Z = nothing
    model = nothing
    display_device = TextDisplay(devnull)
    elapsed_sec = @elapsed begin
        weights = ones(size(A, 1))
        Q = sparse_robot_world_transformation_cost(A, B, weights, weights)
        pushdisplay(display_device)
        try
            Z, model = redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    solve_sdp_dual_jump(Q, 2, true, true)
                end
            end
        finally
            popdisplay(display_device)
        end
    end

    solution = vec(extract_solution_from_dual(Z))
    X_est = pose_from_solution(solution, 7, 1)
    Y_est = pose_from_solution(solution, 16, 4)
    return X_est, Y_est, model, elapsed_sec
end

function warm_up_solver(problems)
    isempty(problems) && return nothing
    first_problem = first(problems)
    solve_problem(first_problem.A, first_problem.B)
    return nothing
end

function short_status(status::AbstractString; max_chars=16)
    return length(status) <= max_chars ? status : string(first(status, max_chars - 3), "...")
end

function summarize_results(results)
    successful = filter(result -> result.successful, results)
    optimal = filter(result -> result.status == "OPTIMAL", results)
    finite_times = [result.solve_time_ms for result in results if isfinite(result.solve_time_ms)]

    return (
        total = length(results),
        successful = length(successful),
        optimal = length(optimal),
        mean_x_error = isempty(successful) ? NaN : mean(getfield.(successful, :x_error)),
        mean_y_error = isempty(successful) ? NaN : mean(getfield.(successful, :y_error)),
        mean_solve_time_ms = isempty(finite_times) ? NaN : mean(finite_times),
    )
end

function main()
    input_meas, success_tol = parse_cli(ARGS)
    problems = load_problem_sequence(input_meas)

    isempty(problems) && error("No problems found in $input_meas")
    warm_up_solver(problems)

    results = NamedTuple[]
    for (problem_index, problem) in enumerate(problems)
        result = try
            X_est, Y_est, model, elapsed_sec = solve_problem(problem.A, problem.B)
            status = string(JuMP.termination_status(model))
            x_error = pose_error(X_est, problem.X0)
            y_error = pose_error(Y_est, problem.Y0)
            objective_gt = mean_pose_residual(problem.A, problem.B, problem.X0, problem.Y0)
            objective_est = mean_pose_residual(problem.A, problem.B, X_est, Y_est)
            successful = status == "OPTIMAL" && x_error <= success_tol && y_error <= success_tol

            (
                problem_index = problem_index,
                len = size(problem.A, 1),
                status = status,
                objective_gt = objective_gt,
                objective_est = objective_est,
                x_error = x_error,
                y_error = y_error,
                solve_time_ms = 1000 * elapsed_sec,
                successful = successful,
            )
        catch err
            (
                problem_index = problem_index,
                len = size(problem.A, 1),
                status = sprint(showerror, err),
                objective_gt = NaN,
                objective_est = NaN,
                x_error = NaN,
                y_error = NaN,
                solve_time_ms = NaN,
                successful = false,
            )
        end

        push!(results, result)
        println(
            @sprintf(
                "problem %d/%d -> %s, successful=%s",
                problem_index,
                length(problems),
                short_status(result.status),
                result.successful ? "true" : "false",
            ),
        )
    end

    summary = summarize_results(results)

    println()
    println("input_meas = ", input_meas)
    println("success_tol = ", success_tol)
    println("problems = ", length(problems))
    println()
    println(rpad("problem", 10),
            rpad("len", 8),
            rpad("status", 18),
            lpad("gt objective", 16),
            lpad("est objective", 16),
            lpad("pose err X", 14),
            lpad("pose err Y", 14),
            lpad("solve ms", 12),
            lpad("successful", 12))
    println(repeat("-", 120))

    for result in results
        println(
            @sprintf(
                "%-10d%-8d%-18s%16.6e%16.6e%14.6e%14.6e%12.3f%12s",
                result.problem_index,
                result.len,
                short_status(result.status),
                result.objective_gt,
                result.objective_est,
                result.x_error,
                result.y_error,
                result.solve_time_ms,
                result.successful ? "true" : "false",
            ),
        )
    end

    println()
    println("Successful problems: ", summary.successful, "/", summary.total)
    println("Optimal problems: ", summary.optimal, "/", summary.total)
    println("Mean X pose error over successful problems: ", summary.mean_x_error)
    println("Mean Y pose error over successful problems: ", summary.mean_y_error)
    println("Mean solve time (ms): ", summary.mean_solve_time_ms)
end

main()
