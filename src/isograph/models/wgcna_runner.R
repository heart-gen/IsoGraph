#!/usr/bin/env Rscript
# WGCNA runner for IsoGraph Stage 5+ comparison benchmark.
#
# Usage:
#   Rscript wgcna_runner.R \
#     input=<path/to/gene_matrix.csv> \
#     output=<path/to/result.json> \
#     power=auto \
#     power_range=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
#     sft_r2=0.85 \
#     min_module_size=2 \
#     merge_cut_height=0.25 \
#     deep_split=2 \
#     network_type=signed \
#     seed=0
#
# Input CSV: genes x samples, first column is gene_id.
# Output JSON: modules list, edges list, calibration dict.
#
# Scale behaviour:
#   n_genes <= 5000: full adjacency + TOM (exact, edges included)
#   n_genes >  5000: blockwiseModules (memory-safe; edge table omitted)

suppressMessages(library(WGCNA))
options(stringsAsFactors=FALSE)

# ---------------------------------------------------------------------------
# Argument parsing (key=value positional args, no extra packages required)
# ---------------------------------------------------------------------------

parse_args <- function() {
  raw <- commandArgs(trailingOnly=TRUE)
  args <- list(
    input            = NULL,
    output           = NULL,
    power            = "auto",
    power_range      = paste(1:20, collapse=","),
    sft_r2           = "0.85",
    min_module_size  = "2",
    merge_cut_height = "0.25",
    deep_split       = "2",
    network_type     = "signed",
    seed             = "0"
  )
  for (a in raw) {
    kv <- strsplit(a, "=", fixed=TRUE)[[1]]
    if (length(kv) >= 2) {
      key <- kv[1]
      val <- paste(kv[-1], collapse="=")
      args[[key]] <- val
    }
  }
  if (is.null(args$input) || is.null(args$output))
    stop("Required args: input=<path> output=<path>")
  args
}

# ---------------------------------------------------------------------------
# Minimal base-R JSON serialisation (no jsonlite dependency)
# ---------------------------------------------------------------------------

json_str <- function(x) {
  if (is.null(x))       "null"
  else if (is.logical(x)) if (isTRUE(x)) "true" else "false"
  else if (is.numeric(x)) {
    if (is.nan(x) || is.infinite(x)) "null"
    else formatC(x, format="g", digits=8)
  } else paste0('"', gsub('"', '\\\\"', as.character(x)), '"')
}

json_obj <- function(lst) {
  parts <- mapply(function(k, v) paste0('"', k, '": ', json_str(v)),
                  names(lst), lst, SIMPLIFY=TRUE)
  paste0("{", paste(parts, collapse=", "), "}")
}

json_array <- function(df) {
  if (nrow(df) == 0) return("[]")
  rows <- apply(df, 1, function(row) json_obj(as.list(row)))
  paste0("[\n  ", paste(rows, collapse=",\n  "), "\n]")
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BLOCKWISE_THRESHOLD <- 5000L   # switch to blockwiseModules above this

main <- function() {
  args <- parse_args()

  set.seed(as.integer(args$seed))

  sft_r2_threshold <- as.numeric(args$sft_r2)
  min_module_size  <- as.integer(args$min_module_size)
  merge_cut_height <- as.numeric(args$merge_cut_height)
  deep_split       <- as.integer(args$deep_split)
  network_type     <- args$network_type
  power_range      <- as.integer(strsplit(args$power_range, ",")[[1]])

  # --- Read gene matrix (genes x samples, col 1 = gene_id) ---
  mat_df   <- read.csv(args$input, check.names=FALSE)
  gene_ids <- mat_df[, 1]
  mat      <- as.matrix(mat_df[, -1])   # genes x samples
  datExpr  <- t(mat)                     # samples x genes (WGCNA convention)
  colnames(datExpr) <- gene_ids

  n_genes <- ncol(datExpr)

  # Handle degenerate case: too few genes for WGCNA
  if (n_genes < 4) {
    result_json <- paste0(
      '{"modules": [], "edges": [], "calibration": ',
      json_obj(list(
        wgcna_soft_threshold_power = NA,
        wgcna_sft_r2               = NA,
        wgcna_n_modules            = 0L,
        wgcna_n_unassigned_genes   = n_genes,
        wgcna_merge_cut_height     = merge_cut_height,
        wgcna_network_type         = network_type
      )),
      "}"
    )
    writeLines(result_json, args$output)
    return(invisible(NULL))
  }

  # --- Soft-thresholding power selection ---
  if (args$power == "auto") {
    sft <- pickSoftThreshold(
      datExpr,
      powerVector   = power_range,
      networkType   = network_type,
      verbose       = 0
    )
    fit_indices  <- sft$fitIndices
    r2_col       <- fit_indices$SFT.R.sq
    passing      <- which(r2_col >= sft_r2_threshold)
    if (length(passing) > 0) {
      chosen_power <- fit_indices$Power[passing[1]]
      chosen_r2    <- r2_col[passing[1]]
    } else {
      best_idx     <- which.max(r2_col)
      chosen_power <- fit_indices$Power[best_idx]
      chosen_r2    <- r2_col[best_idx]
    }
  } else {
    chosen_power <- as.integer(args$power)
    chosen_r2    <- NA_real_
  }

  # --- Module detection: blockwise for large inputs, full TOM for small ---
  if (n_genes > BLOCKWISE_THRESHOLD) {
    # blockwiseModules avoids holding the full n_genes x n_genes TOM in RAM.
    # maxBlockSize=6000 means at most 2 blocks for a 12k-gene dataset.
    max_block <- min(n_genes, 6000L)
    bwm <- blockwiseModules(
      datExpr,
      power          = chosen_power,
      networkType    = network_type,
      minModuleSize  = min_module_size,
      mergeCutHeight = merge_cut_height,
      deepSplit      = deep_split,
      maxBlockSize   = max_block,
      numericLabels  = FALSE,
      verbose        = 0
    )
    final_colors <- bwm$colors
    # Full adjacency matrix is not available after blockwise — return empty edges.
    edges_df <- data.frame(source=character(0), target=character(0), weight=numeric(0))
  } else {
    # Full TOM: exact but O(n_genes^2) RAM — safe up to ~5000 genes.
    adjacency_mat <- adjacency(
      datExpr,
      power = chosen_power,
      type  = network_type
    )
    dissTOM  <- 1 - TOMsimilarity(adjacency_mat)
    geneTree <- hclust(as.dist(dissTOM), method="average")
    dynamic_labels <- cutreeDynamic(
      dendro         = geneTree,
      distM          = dissTOM,
      deepSplit      = deep_split,
      minClusterSize = min_module_size,
      method         = "hybrid",
      verbose        = 0
    )
    dynamic_colors <- labels2colors(dynamic_labels)
    merged <- mergeCloseModules(
      datExpr,
      dynamic_colors,
      cutHeight = merge_cut_height,
      verbose   = 0
    )
    final_colors <- merged$colors

    # Vectorized upper-triangle edge extraction (replaces O(n^2) R for-loop).
    adj_threshold <- 0.1
    idx <- which(adjacency_mat > adj_threshold, arr.ind=TRUE)
    idx <- idx[idx[, 1] < idx[, 2], , drop=FALSE]
    if (nrow(idx) > 0) {
      edges_df <- data.frame(
        source = gene_ids[idx[, 1]],
        target = gene_ids[idx[, 2]],
        weight = adjacency_mat[idx],
        stringsAsFactors = FALSE
      )
    } else {
      edges_df <- data.frame(source=character(0), target=character(0), weight=numeric(0))
    }
  }

  # --- Build module table (exclude grey = unassigned) ---
  assigned_mask   <- final_colors != "grey"
  assigned_genes  <- gene_ids[assigned_mask]
  assigned_colors <- final_colors[assigned_mask]
  n_unassigned    <- sum(!assigned_mask)

  if (length(assigned_genes) > 0) {
    color_sizes   <- sort(table(assigned_colors), decreasing=TRUE)
    unique_colors <- names(color_sizes)
    color_to_id   <- setNames(
      sprintf("M%03d", seq_along(unique_colors) - 1L),
      unique_colors
    )
    module_ids <- color_to_id[assigned_colors]
    modules_df <- data.frame(gene_id=assigned_genes, module_id=module_ids,
                             stringsAsFactors=FALSE)
  } else {
    modules_df <- data.frame(gene_id=character(0), module_id=character(0))
  }

  # --- Calibration ---
  n_real_modules <- if (length(assigned_genes) > 0) length(unique(module_ids)) else 0L
  cal <- list(
    wgcna_soft_threshold_power = chosen_power,
    wgcna_sft_r2               = chosen_r2,
    wgcna_n_modules            = n_real_modules,
    wgcna_n_unassigned_genes   = n_unassigned,
    wgcna_merge_cut_height     = merge_cut_height,
    wgcna_network_type         = network_type
  )

  # --- Write JSON output ---
  result_json <- paste0(
    '{\n',
    '  "modules": ', json_array(modules_df), ',\n',
    '  "edges": ', json_array(edges_df), ',\n',
    '  "calibration": ', json_obj(cal), '\n',
    '}'
  )
  writeLines(result_json, args$output)
}

tryCatch(
  main(),
  error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n", file=stderr())
    quit(status=1)
  }
)
