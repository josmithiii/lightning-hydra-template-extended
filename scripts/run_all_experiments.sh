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

echo -e "${BLUE}=== Lightning-Hydra-Template-Extended Experiment Runner ===${NC}"
echo -e "${BLUE}Log directory: ${LOG_DIR}${NC}"
echo ""

# Array of all experiment names (without .yaml extension)
experiments=(
    "cifar10_benchmark_cnn"
    "cifar10_benchmark_convnext"
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

# Function to run a single experiment
run_experiment() {
    local experiment_name="$1"
    local log_file="${LOG_DIR}/${experiment_name}-log.txt"

    echo -e "${YELLOW}[$(date '+%H:%M:%S')] Starting experiment: ${experiment_name}${NC}"
    echo -e "${BLUE}  Output will be saved to: ${log_file}${NC}"

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
    if python src/train.py experiment="${experiment_name}" >> "${log_file}" 2>&1; then
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
        echo -e "${RED}[$(date '+%H:%M:%S')] ✗ Failed: ${experiment_name}${NC}"
        # Add failure footer to log file
        {
            echo ""
            echo "==================================================================="
            echo "EXPERIMENT FAILED"
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

    echo -e "${BLUE}Total experiments to run: ${total_experiments}${NC}"
    echo ""

    # If arguments provided, run only specified experiments
    if [ $# -gt 0 ]; then
        echo -e "${YELLOW}Running specified experiments: $@${NC}"
        for experiment in "$@"; do
            if [[ " ${experiments[@]} " =~ " ${experiment} " ]]; then
                run_experiment "${experiment}"
                if [ $? -eq 0 ]; then
                    ((completed++))
                else
                    ((failed++))
                fi
            else
                echo -e "${RED}Warning: Experiment '${experiment}' not found in experiment list${NC}"
            fi
        done
    else
        # Run all experiments
        echo -e "${YELLOW}Running all experiments...${NC}"
        for experiment in "${experiments[@]}"; do
            run_experiment "${experiment}"
            if [ $? -eq 0 ]; then
                ((completed++))
            else
                ((failed++))
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
    echo -e "${BLUE}Total time: ${hours}h ${minutes}m ${seconds}s${NC}"
    echo -e "${BLUE}Logs saved in: ${LOG_DIR}${NC}"
}

# Help function
show_help() {
    echo "Usage: $0 [experiment_names...]"
    echo ""
    echo "Run Lightning-Hydra-Template-Extended experiments and save logs."
    echo ""
    echo "Options:"
    echo "  No arguments    - Run all experiments"
    echo "  experiment_names - Run only specified experiments"
    echo "  --list          - List all available experiments"
    echo "  --help          - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all experiments"
    echo "  $0 cifar10_cnn example               # Run specific experiments"
    echo "  $0 --list                            # List available experiments"
    echo ""
    echo "Log files are saved to: experiment_logs/EXPERIMENT_NAME-log.txt"
}

# List experiments function
list_experiments() {
    echo -e "${BLUE}Available experiments:${NC}"
    for experiment in "${experiments[@]}"; do
        echo "  ${experiment}"
    done
    echo ""
    echo -e "${BLUE}Total: ${#experiments[@]} experiments${NC}"
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