#!/bin/bash

# Run All Experiments Script
# This script runs each experiment configuration and captures output to experiment_logs/

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create experiment_logs directory if it doesn't exist
mkdir -p experiment_logs

# Get the absolute path to the experiment_logs directory
LOG_DIR="$(pwd)/experiment_logs"

# Parse command line arguments
FORCE_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_MODE=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

echo -e "${BLUE}=== Lightning-Hydra-Template-Extended Experiment Runner ===${NC}"
echo -e "${BLUE}Log directory: ${LOG_DIR}${NC}"
if [ "$FORCE_MODE" = true ]; then
    echo -e "${YELLOW}Force mode: ON (ignoring all experiment markers)${NC}"
fi
echo ""

# Array of all experiment names (without .yaml extension)
experiments=(
    "Y cifar10_benchmark_cnn"
    "N cifar10_benchmark_convnext"
    "cifar10_benchmark_efficientnet"
    "cifar10_benchmark_vit"
    "cifar10_cnn_cpu"
    "cifar10_cnn"
    "cifar10_convnext_128k_optimized"
    "cifar10_convnext_64k_optimized"
    "cifar100_benchmark_cnn_improved"
    "cifar100_benchmark_cnn"
    "cifar100_benchmark_convnext"
    "cifar100_coarse_cnn"
    "cifar100_cnn"
    "cifar100mh_cnn"
    "cifar100mh_convnext"
    "cifar100mh_efficientnet"
    "cifar100mh_vit"
    "cnn_mnist"
    "convnext_mnist"
    "convnext_v2_official_tiny_benchmark"
    "example"
    "multihead_cnn_cifar10"
    "multihead_cnn_mnist"
    "vimh_cnn_16kdss"
    "vimh_cnn_16kdss_ordinal"
    "vimh_cnn_16kdss_regression"
    "vimh_cnn"
    "vit_mnist_995"
    "vit_mnist"
)

# Function to parse experiment name and return (marker, name)
parse_experiment() {
    local experiment_entry="$1"

    # Check if experiment has a marker (Y/N prefix)
    if [[ "$experiment_entry" =~ ^([YN])\ (.+)$ ]]; then
        local marker="${BASH_REMATCH[1]}"
        local name="${BASH_REMATCH[2]}"
        echo "$marker $name"
    else
        echo "  $experiment_entry"
    fi
}

# Function to run a single experiment
run_experiment() {
    local experiment_name="$1"
    local mode="${2:-run}"  # Default mode is "run" if not specified
    local log_file="${LOG_DIR}/${experiment_name}-log.txt"

    case "$mode" in
        "skip")
            echo -e "${BLUE}[$(date '+%H:%M:%S')] Skipping experiment: ${experiment_name} (marked as successful)${NC}"
            return 0
            ;;
        "debug")
            echo -e "${YELLOW}[$(date '+%H:%M:%S')] Debugging experiment: ${experiment_name} (marked as failed)${NC}"
            echo -e "${BLUE}  Output will be saved to: ${log_file}${NC}"
            echo -e "${YELLOW}  Debug mode: Will show detailed error information${NC}"
            ;;
        "run")
            echo -e "${YELLOW}[$(date '+%H:%M:%S')] Starting experiment: ${experiment_name}${NC}"
            echo -e "${BLUE}  Output will be saved to: ${log_file}${NC}"
            ;;
        *)
            echo -e "${RED}Error: Invalid mode '$mode' for experiment '$experiment_name'${NC}"
            return 1
            ;;
    esac

    # Check if log file already exists and rename it if so
    if [ -f "${log_file}" ]; then
        # Get the modification time of the existing file (macOS compatible)
        local mod_time=$(stat -f %Sm -t %Y%m%d%H%M%S "${log_file}" 2>/dev/null)
        local exit_code=$?

        if [ $exit_code -eq 0 ] && [ -n "${mod_time}" ]; then
            # Convert YYYYMMDDHHMMSS to YYYY-MM-DD-HH-MM-SS format
            local formatted_date=$(echo "${mod_time}" | sed -E 's/([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})/\1-\2-\3-\4-\5-\6/')
            local backup_file="${log_file%.*}-${formatted_date}.txt"
            echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
            mv "${log_file}" "${backup_file}"
        else
            # Fallback: parse from stat output or use current timestamp
            local stat_output=$(stat "${log_file}" 2>/dev/null)
            if [ $? -eq 0 ]; then
                # Extract date from stat output (format: "Sep 20 23:47:44 2025")
                local extracted_date=$(echo "${stat_output}" | awk '{print $9, $10, $11}' | sed 's/ /-/g' | tr -d '"')
                if [ -n "${extracted_date}" ]; then
                    # Convert month name to number and format as YYYY-MM-DD-HH:MM:SS
                    local month_num
                    case "${extracted_date%%-*}" in
                        "Jan") month_num="01" ;;
                        "Feb") month_num="02" ;;
                        "Mar") month_num="03" ;;
                        "Apr") month_num="04" ;;
                        "May") month_num="05" ;;
                        "Jun") month_num="06" ;;
                        "Jul") month_num="07" ;;
                        "Aug") month_num="08" ;;
                        "Sep") month_num="09" ;;
                        "Oct") month_num="10" ;;
                        "Nov") month_num="11" ;;
                        "Dec") month_num="12" ;;
                        *) month_num="??"
                    esac

                    if [ "${month_num}" != "??" ]; then
                        # Extract components: Sep-20-23:47:47 -> year=20, month=Sep, day=20, time=23:47:47
                        local year=$(echo "${extracted_date}" | cut -d'-' -f2)
                        local day=$(echo "${extracted_date}" | cut -d'-' -f2 | cut -d':' -f1)
                        local time_part=$(echo "${extracted_date}" | cut -d'-' -f3)
                        local formatted_date="20${year}-${month_num}-${day}-${time_part}"
                        local backup_file="${log_file%.*}-${formatted_date}.txt"
                        echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                        mv "${log_file}" "${backup_file}"
                    else
                        # Fallback to simple date format
                        local backup_file="${log_file%.*}-${extracted_date}.txt"
                        echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                        mv "${log_file}" "${backup_file}"
                    fi
                else
                    # Final fallback: use current timestamp
                    local backup_file="${log_file%.*}-$(date +%Y-%m-%d-%H-%M-%S).txt"
                    echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                    mv "${log_file}" "${backup_file}"
                fi
            else
                # Final fallback: use current timestamp
                local backup_file="${log_file%.*}-$(date +%Y-%m-%d-%H-%M-%S).txt"
                echo -e "${YELLOW}  Existing log file found, renaming to: ${backup_file}${NC}"
                mv "${log_file}" "${backup_file}"
            fi
        fi
    fi

    # Add experiment header to log file
    {
        echo "==================================================================="
        echo "EXPERIMENT: ${experiment_name}"
        echo "STARTED: $(date)"
        echo "COMMAND: python src/train.py experiment=${experiment_name}"
        echo "==================================================================="
        echo ""
    } > "${log_file}"

    # Run the experiment and capture both stdout and stderr
    if [ "$mode" = "debug" ]; then
        echo -e "${YELLOW}  Executing: python src/train.py experiment=${experiment_name}${NC}"
        echo -e "${YELLOW}  Debug mode: Detailed output follows...${NC}"
        echo ""

        # In debug mode, show real-time output and capture to log
        python src/train.py experiment="${experiment_name}" 2>&1 | tee "${log_file}"
        local exit_code=${PIPESTATUS[0]}

        echo ""
        echo -e "${YELLOW}  Debug mode: Execution completed with exit code $exit_code${NC}"
    else
        # Normal mode - capture output to log file
        python src/train.py experiment="${experiment_name}" >> "${log_file}" 2>&1
        local exit_code=$?
    fi

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ Completed: ${experiment_name}${NC}"
        # Add completion footer to log file
        {
            echo ""
            echo "==================================================================="
            echo "EXPERIMENT COMPLETED SUCCESSFULLY"
            echo "FINISHED: $(date)"
            echo "==================================================================="
        } >> "${log_file}"
    else
        echo -e "${RED}[$(date '+%H:%M:%S')] ✗ Failed: ${experiment_name} (exit code: $exit_code)${NC}"
        # Add failure footer to log file
        {
            echo ""
            echo "==================================================================="
            echo "EXPERIMENT FAILED"
            echo "EXIT CODE: $exit_code"
            echo "FINISHED: $(date)"
            echo "==================================================================="
        } >> "${log_file}"
    fi

    echo ""
}

# Function to run experiments with optional filtering
run_experiments() {
    local start_time=$(date +%s)
    local total_experiments=${#experiments[@]}
    local completed=0
    local failed=0
    local skipped=0

    echo -e "${BLUE}Total experiments to process: ${total_experiments}${NC}"
    echo ""

    # Determine processing mode for all experiments
    local processing_mode
    if [ "$FORCE_MODE" = true ]; then
        processing_mode="force"
        echo -e "${YELLOW}Force mode: All experiments will be run normally${NC}"
    else
        processing_mode="normal"
    fi
    echo ""

    # If arguments provided, run only specified experiments
    if [ $# -gt 0 ]; then
        echo -e "${YELLOW}Running specified experiments: $@${NC}"
        for experiment_spec in "$@"; do
            # Parse the experiment specification
            local parsed=$(parse_experiment "$experiment_spec")
            local marker=$(echo "$parsed" | cut -d' ' -f1)
            local experiment_name=$(echo "$parsed" | cut -d' ' -f2-)

            if [[ " ${experiments[@]} " =~ " ${experiment_spec} " ]]; then
                if [ "$processing_mode" = "force" ]; then
                    # Force mode: ignore markers and run all experiments normally
                    echo -e "${YELLOW}  Running (force): $experiment_spec${NC}"
                    run_experiment "$experiment_name" "run"
                    if [ $? -eq 0 ]; then
                        ((completed++))
                    else
                        ((failed++))
                    fi
                else
                    # Normal mode: respect markers
                    case "$marker" in
                        "Y")
                            echo -e "${BLUE}  Skipping: $experiment_spec${NC}"
                            run_experiment "$experiment_name" "skip"
                            ((skipped++))
                            ;;
                        "N")
                            echo -e "${YELLOW}  Debugging: $experiment_spec${NC}"
                            run_experiment "$experiment_name" "debug"
                            if [ $? -eq 0 ]; then
                                ((completed++))
                            else
                                ((failed++))
                            fi
                            ;;
                        " ")
                            echo -e "${YELLOW}  Running: $experiment_spec${NC}"
                            run_experiment "$experiment_name" "run"
                            if [ $? -eq 0 ]; then
                                ((completed++))
                            else
                                ((failed++))
                            fi
                            ;;
                    esac
                fi
            else
                echo -e "${RED}Warning: Experiment '${experiment_spec}' not found in experiment list${NC}"
            fi
        done
    else
        # Run all experiments
        if [ "$processing_mode" = "force" ]; then
            echo -e "${YELLOW}Running all experiments in force mode...${NC}"
        else
            echo -e "${YELLOW}Processing all experiments...${NC}"
        fi
        for experiment in "${experiments[@]}"; do
            # Parse the experiment entry
            local parsed=$(parse_experiment "$experiment")
            local marker=$(echo "$parsed" | cut -d' ' -f1)
            local experiment_name=$(echo "$parsed" | cut -d' ' -f2-)

            if [ "$processing_mode" = "force" ]; then
                # Force mode: ignore markers and run all experiments normally
                echo -e "${YELLOW}  Running (force): $experiment${NC}"
                run_experiment "$experiment_name" "run"
                if [ $? -eq 0 ]; then
                    ((completed++))
                else
                    ((failed++))
                fi
            else
                # Normal mode: respect markers
                case "$marker" in
                    "Y")
                        echo -e "${BLUE}  Skipping: $experiment${NC}"
                        run_experiment "$experiment_name" "skip"
                        ((skipped++))
                        ;;
                    "N")
                        echo -e "${YELLOW}  Debugging: $experiment${NC}"
                        run_experiment "$experiment_name" "debug"
                        if [ $? -eq 0 ]; then
                            ((completed++))
                        else
                            ((failed++))
                        fi
                        ;;
                    " ")
                        echo -e "${YELLOW}  Running: $experiment${NC}"
                        run_experiment "$experiment_name" "run"
                        if [ $? -eq 0 ]; then
                            ((completed++))
                        else
                            ((failed++))
                        fi
                        ;;
                esac
            fi
        done
    fi

    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    echo -e "${BLUE}=== EXPERIMENT SUMMARY ===${NC}"
    echo -e "${GREEN}Completed: ${completed}${NC}"
    echo -e "${RED}Failed: ${failed}${NC}"
    echo -e "${BLUE}Skipped: ${skipped}${NC}"
    echo -e "${BLUE}Total processed: $((completed + failed + skipped))${NC}"
    echo -e "${BLUE}Total time: ${hours}h ${minutes}m ${seconds}s${NC}"
    echo -e "${BLUE}Logs saved in: ${LOG_DIR}${NC}"
}

# Help function
show_help() {
    echo "Usage: $0 [experiment_names...]"
    echo ""
    echo "Run Lightning-Hydra-Template-Extended experiments and save logs."
    echo ""
    echo "Experiment Marking Scheme:"
    echo "  Y experiment_name    - Skip (already successful)"
    echo "  N experiment_name    - Debug mode (failed, needs attention)"
    echo "  experiment_name      - Run normally (no marking)"
    echo ""
    echo "Options:"
    echo "  No arguments         - Process all experiments according to markings"
    echo "  experiment_names     - Process only specified experiments"
    echo "  --force, -f         - Ignore markers and run all experiments normally"
    echo "  --list              - List all available experiments"
    echo "  --help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Process all experiments"
    echo "  $0 cifar10_cnn example               # Process specific experiments"
    echo "  $0 \"N cifar10_benchmark_convnext\"   # Debug a failed experiment"
    echo "  $0 --force                           # Run all experiments ignoring markers"
    echo "  $0 -f cifar10_cnn                   # Force-run specific experiment"
    echo "  $0 --list                            # List available experiments"
    echo ""
    echo "Log files are saved to: experiment_logs/EXPERIMENT_NAME-log.txt"
    echo "Debug mode shows real-time output and captures detailed error information."
}

# List experiments function
list_experiments() {
    echo -e "${BLUE}Available experiments:${NC}"
    for experiment in "${experiments[@]}"; do
        # Parse the experiment entry
        local parsed=$(parse_experiment "$experiment")
        local marker=$(echo "$parsed" | cut -d' ' -f1)
        local name=$(echo "$parsed" | cut -d' ' -f2-)

        case "$marker" in
            "Y")
                echo -e "  ${GREEN}✓${NC} ${experiment} ${BLUE}(successful)${NC}"
                ;;
            "N")
                echo -e "  ${RED}✗${NC} ${experiment} ${YELLOW}(failed)${NC}"
                ;;
            " ")
                echo -e "  ${BLUE}•${NC} ${experiment} ${BLUE}(pending)${NC}"
                ;;
        esac
    done
    echo ""
    echo -e "${BLUE}Total: ${#experiments[@]} experiments${NC}"
    echo ""
    echo -e "${BLUE}Legend:${NC}"
    echo -e "  ${GREEN}✓${NC} = Successful (will be skipped)"
    echo -e "  ${RED}✗${NC} = Failed (will be debugged)"
    echo -e "  ${BLUE}•${NC} = Pending (will be run normally)"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --list|-l)
        list_experiments
        exit 0
        ;;
    *)
        run_experiments "$@"
        ;;
esac