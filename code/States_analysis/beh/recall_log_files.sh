#!/bin/bash
# =============================================================================
# Two-step process to recall log files from ESI archive for 18 missing
# LeafForaging sessions and copy them to the raw_data folder.
#
# USAGE:
#   Step 1 - Submit the recall (run from an interactive node):
#     ./recall_log_files.sh recall
#
#   Step 2 - After the recall job finishes, copy files to raw_data:
#     ./recall_log_files.sh copy
#
#   Optional - Check if recall is done:
#     ./recall_log_files.sh check
# =============================================================================

set -euo pipefail

ARCHIVE_BASE="/mnt/as/projects/MWzeronoise/test_recordings/Cosmos"
RAW_DATA="/mnt/cs/projects/MWzeronoise/Analysis/4Shivangi/Datasets/raw_data"
RECALL_DIR="${RAW_DATA}/recall_tmp"

# Missing sessions: date -> LeafForaging session number
declare -A SESSIONS=(
    ["20230105"]="001"
    ["20230106"]="001"
    ["20230109"]="001"
    ["20230110"]="001"
    ["20230111"]="001"
    ["20230112"]="001"
    ["20230113"]="001"
    ["20230116"]="001"
    ["20230117"]="001"
    ["20230118"]="001"
    ["20230119"]="001"
    ["20230120"]="001"
    ["20230123"]="001"
    ["20230124"]="001"
    ["20230125"]="001"
    ["20230126"]="001"
    ["20230127"]="001"
    ["20230130"]="001"
)

# --- Build file list (shared by all steps) ---
build_file_list() {
    RECALL_LIST=""
    declare -gA LOG_PATHS
    for date in $(echo "${!SESSIONS[@]}" | tr ' ' '\n' | sort); do
        sess="${SESSIONS[$date]}"
        archive_dir="${ARCHIVE_BASE}/${date}/LeafForaging/${sess}"
        log_file=$(find "$archive_dir" -maxdepth 1 -name "*GrassyLandscape*_Cont.log" 2>/dev/null | head -1)
        if [ -z "$log_file" ]; then
            echo "WARNING: No log found for $date/$sess"
            continue
        fi
        RECALL_LIST="${RECALL_LIST} ${log_file}"
        LOG_PATHS[$date]="$log_file"
    done
}

# --- Step 1: Submit recall ---
do_recall() {
    build_file_list
    mkdir -p "$RECALL_DIR"

    echo "=== Submitting recall for ${#LOG_PATHS[@]} log files ==="
    echo "Recall output directory: $RECALL_DIR"
    echo ""

    # Write file list for reference
    printf "%s\n" ${RECALL_LIST} > "${RECALL_DIR}/filelist.txt"
    echo "File list saved to ${RECALL_DIR}/filelist.txt"
    echo ""

    echo "Running: archive_recall.sh -f ${RECALL_DIR}/filelist.txt -o ${RECALL_DIR}"
    archive_recall.sh -f "${RECALL_DIR}/filelist.txt" -o "$RECALL_DIR"

    echo ""
    echo "=== Recall submitted ==="
    echo "Wait for the recall job to finish, then run:"
    echo "  $0 check    # to check if files are ready"
    echo "  $0 copy     # to copy files to raw_data"
}

# --- Check if recall is done ---
do_check() {
    build_file_list
    echo "=== Checking recall status ==="
    ready=0
    total=${#LOG_PATHS[@]}
    for date in $(echo "${!LOG_PATHS[@]}" | tr ' ' '\n' | sort); do
        filename=$(basename "${LOG_PATHS[$date]}")
        found=$(find "$RECALL_DIR" -name "$filename" 2>/dev/null | head -1)
        if [ -n "$found" ] && [ -s "$found" ]; then
            echo "  READY: $date - $filename"
            ((ready++))
        else
            echo "  PENDING: $date - $filename"
        fi
    done
    echo ""
    echo "$ready / $total files ready"
}

# --- Step 2: Copy recalled files to raw_data ---
do_copy() {
    echo "=== Copying recalled log files to raw_data ==="
    copied=0
    total=0

    # Scan the recall directory for all _Cont.log files
    while IFS= read -r recalled_file; do
        ((total++))
        filename=$(basename "$recalled_file")
        # Extract date from recall path: .../Cosmos/YYYYMMDD/LeafForaging/...
        date=$(echo "$recalled_file" | grep -oP 'Cosmos/\K[0-9]{8}')
        if [ -z "$date" ]; then
            echo "  SKIP: Could not extract date from $recalled_file"
            continue
        fi

        dest_dir="${RAW_DATA}/${date}"
        mkdir -p "$dest_dir"
        cp "$recalled_file" "$dest_dir/"
        chmod a-w "$dest_dir/$filename"
        echo "  OK: $filename -> $dest_dir/"
        ((copied++))
    done < <(find "$RECALL_DIR" -name "*_Cont.log" -type f 2>/dev/null)

    echo ""
    echo "Copied $copied / $total files"

    if [ "$copied" -eq "$total" ] && [ "$total" -gt 0 ]; then
        echo ""
        echo "=== All files copied. Cleaning up recall directory ==="
        rm -rf "$RECALL_DIR"
        echo "Removed $RECALL_DIR"
    else
        echo ""
        echo "WARNING: Not all files were copied. Recall dir kept at $RECALL_DIR"
    fi

    echo ""
    echo "=== Verification ==="
    for date in $(echo "${!SESSIONS[@]}" | tr ' ' '\n' | sort); do
        log=$(find "${RAW_DATA}/${date}" -name "*_Cont.log" 2>/dev/null | head -1)
        if [ -n "$log" ]; then
            echo "  OK: $date -> $(basename "$log")"
        else
            echo "  MISSING: $date"
        fi
    done
}

# --- Main ---
case "${1:-}" in
    recall)
        do_recall
        ;;
    check)
        do_check
        ;;
    copy)
        do_copy
        ;;
    *)
        echo "Usage: $0 {recall|check|copy}"
        echo ""
        echo "  recall  - Submit archive recall job for 18 missing log files"
        echo "  check   - Check if recalled files are ready"
        echo "  copy    - Copy recalled files to raw_data and clean up"
        exit 1
        ;;
esac
