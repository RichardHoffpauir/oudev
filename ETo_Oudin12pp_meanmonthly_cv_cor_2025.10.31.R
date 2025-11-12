# ------------------------------------------------------------------------------
# Evapotranspiration Modeling for the Colorado River Basin (CRB) in Texas
#
# Oudin ETo calibration – 12 parameter pairs (a and b) with train/test split, daily time steps
#
# Model form (daily):
#   ETo_mod = pmax(0, a_m * Ra * (Tavg + b_m))
#   where:
#     - Tavg = 0.5*(Tmax + Tmin)
#     - a_m is a calibration parameter for calendar month m (m = 1..12)
#     - b_m is a calibration parameter for calendar month m (m = 1..12)
#     - Ra is extraterrestrial radiation (MJ m^-2 d^-1)
#
# Parameters and bounds:
#   a01..a12 ∈ [0.0001, 0.02] (calibrated)
#   b01..b12 ∈ [-30, 60] (calibrated)
#   Pre-calculated b values (b01_precalc..b12_precalc) are computed from training data
#   and used in Level 1 to seed Level 2 for quadrangle-level optimization.
#
# Calibration approach:
#   Quadrangle-level optimization (performed first for each quadrangle):
#     Pre-Level 1: Calculate monthly b parameters from quadrangle-level training data Tavg.
#     Level 1: One-dimensional Brent optimization of parameter 'a_global_L1' with fixed 
#              pre-calculated monthly b values to maximize KGE on daily TRAINING data.
#              Performed for seven different Tavg quantiles (0.0, 0.003, 0.005, 0.01, 0.025, 0.05, 0.10).
#     Level 2: Differential Evolution on all 24 parameters (a01..a12, b01..b12) to maximize 
#              KGE on daily TRAINING data. The first seven members of the initial population are seeded 
#              with the seven Level 1 results.
#     Level 3: nlminb algorithm refines all 24 parameters starting from Level 2 best solution,
#              maximizing KGE on daily TRAINING data.
#   QDGC-level optimization (performed for each of 16 QDGCs per quadrangle):
#     Level 2: Differential Evolution on all 24 parameters (a01..a12, b01..b12) to maximize 
#              KGE on daily TRAINING data. The first 5 members of the initial population are 
#              seeded as follows: member 1 receives the exact optimal parameters from quadrangle-level 
#              optimization (Level 3 results); members 2-5 receive perturbed versions with ±10% 
#              uniform noise scaled by absolute parameter value, bounded to remain within feasible 
#              ranges. This provides diverse starting points near the high-quality quadrangle solution 
#              while maintaining 97.9% population diversity for exploration. Parameter bounds for 'b' 
#              are dynamically set to ±2.5 around quadrangle calibration values (clamped to [-30, 60]).
#     Level 3: nlminb algorithm refines all 24 parameters starting from Level 2 best solution,
#              maximizing KGE on daily TRAINING data.
#
# Data, periods, objective:
#   - The daily maximum and minimum air temperature data have a period of record
#     from 1915-01-01 through 2018-12-31. These data were derived from the LOCA2
#     training data, also known as Livneh et al. (2015) or simply Livneh.
#     https://loca.ucsd.edu/training-observed-data-sets/
#   - The daily reference grass evapotranspiration (ETo) have period of record
#     from 1979-01-01 through 2024-12-31. These data were derived from the gridMET
#     short grass reference evapotranspiration which was calculated according to
#     ASCE Penman-Monteith. https://www.climatologylab.org/gridmet.html
#   - The common span of the period of records for the Livneh temperature inputs and gridMET ETo
#     observations is 1979-01-01 through 2018-12-31.
#   - The Livneh data have a native resolution of 6km. The gridMET data have a native
#     resolution of 4km. These two datasets were processed into daily time series over
#     a larger spatial resolution, covering 0.25 degree longitude x 0.25 degree latitude,
#     also known as Quarter Degree Grid Cells (QDGC).
#   - The files containing temperature and ETo data are organized by 1 degree x 1 degree "Quadrangles".
#     The Quadrangle files contain the time series for the 16 QDGCs within that Quadrangle.
#   - The ETo model is calibrated by maximizing KGE on daily ETo over the training
#     period of 1999-01-01 through 2018-12-31.
#   - The remaining period of record, 1979-01-01 through 1998-12-31, is used to evaluate the
#     model performance on data outside of the training period; that is, 1979-1998 is the testing set.
#   - If ANY missing (NA) data exists in 1979-2018 from the files containing ETo/Tmax/Tmin for a QDGC,
#     that QDGC is skipped. The output files written to disk will contain NA values and notes
#     that this QDGC was skipped.
#
# Outputs per Quadrangle:
#   - Tab-separated file (TSV) that summarizes the calibration results per QDGC, and includes
#     the final a01..a12 (8 decimals); b01..b12 (8 decimals); TRAIN and TEST metrics such as 
#     KGE, r, mean_monthly, cv_obs, cv_sim, cv_ratio; boundary flags for
#     a and b parameters; and diagnostic metrics. Additionally includes quadrangle-level optimization
#     results for Levels 1, 2, and 3 in the rightmost columns (rows vary by optimization level).
#   - Modeled ETo daily time series file (full common span 1979-2018); columnar formatting mirrors
#     the input files.
#   - Extraterrestrial solar radiation, Ra, daily time series file (full common span), calculated
#     using FAO-56 equations with 365-day trigonometric terms. The Ra output file is provided so
#     that the modeled ETo time series can be fully verified by calculating Oudin using the
#     temperature inputs, Ra, and final model parameters.
#   - The daily time series data in the modeled ETo and Ra files are formatted with 2 and 4 decimals,
#     respectively.
#
# Notes:
#   - Temperature data are in units of Celsius.
#   - Evapotranspiration data are in units of mm per day.
#   - Ra computed by FAO-56 (365-day trig). Leap days are retained; on doy=366,
#     Ra is mapped to doy=1.
#   - The format of the QDGC daily time series input and output files have the following format:
#     column 1: Year, column 2: Month, column 3: Day, column 4: Quadrangle,
#     columns 5 through 20: quarter degree grid cells within the quadrangle.
#   - As an example, for Quadrangle 709, the column 4 through 20 names in the daily time series
#     input and output files are: Q709, Q709.01, Q709.02, Q709.03, ... , Q709.14, Q709.15, Q709.16
#   - For the daily time series output files, the column 4 Quadrangle value is calculated as
#     the mean of the full-precision QDGC columns, then rounded to the specified decimals.
#   - For the daily time series output files, February 29 is retained for leap years.
# ------------------------------------------------------------------------------

# Clear plots, console, and workspace
if (!is.null(dev.list())) dev.off()
cat("\014")
rm(list = ls())

# ------------------------------------------------------------------------------
# Quadrangles to process, qnum.
#
# The quadrangle numbering system follows the format used by the Texas Water Development Board (TWDB).
# https://waterdatafortexas.org/lake-evaporation-rainfall
# The first digit(s) of the TWDB quadrangle numbering system are related to latitude in a north to south order.
# The first row of quadrangles (northernmost across the Texas panhandle) have a leading digit 1.
# The southernmost row of quadrangles (southern border near Brownsville, TX) have leading digits 12.
# The rightmost two digits of the quadrangle numbers are related to longitude in a west to east order.
# The westernmost quadrangles near El Paso have rightmost digits 01. This row of quadrangles increments
# the digits across the state (west to east) and concludes with rightmost digits 14 over the Texas-Louisiana
# state line.
# ------------------------------------------------------------------------------

qnum <- c(406,505,506,507,508,509,605,606,607,608,609,707,708,709,710,711,810,811,812,911,912)

if (length(qnum) == 0) {
  message("No quadrangles to process (qnum is empty).")
  quit(save = "no")
}

# Packages
suppressPackageStartupMessages({
  library(data.table)
})
# Differential Evolution (required for Level 2 optimization)
library(RcppDE)

# Rcpp for monthly mean error and coefficient of variation calculations
library(Rcpp)

# ------------------------------------------------------------------------------
# Rcpp functions for monthly mean error and coefficient of variation
# ------------------------------------------------------------------------------
cppFunction('
double monthly_mean_err(NumericVector sim, IntegerVector month_ix, NumericVector mean_obs_monthly) {
  int j, n = sim.size();
  
  // Initialize arrays for monthly sums and counts (12 months)
  double monthly_sum[12] = {0.0};
  int monthly_count[12] = {0};
  
  // Accumulate sums and counts for each month
  // month_ix is 1-based (R indexing), so subtract 1 for 0-based C++ indexing
  for(j = 0; j < n; j++){
    int month_idx = month_ix[j] - 1;  // Convert to 0-based index
    monthly_sum[month_idx] += sim[j];
    monthly_count[month_idx] += 1;
  }
  
  // Calculate sum of absolute differences between simulated and observed monthly means
  // Normalize by sum of observed monthly means
  double sum_abs_diff = 0.0;
  double sum_obs = 0.0;
  
  for(int m = 0; m < 12; m++){
    double mean_sim_m = monthly_sum[m] / monthly_count[m];
    double abs_diff = fabs(mean_sim_m - mean_obs_monthly[m]);
    sum_abs_diff += abs_diff;
    sum_obs += mean_obs_monthly[m];
  }
  
  // Calculate normalized error metric
  // ratio = total absolute monthly differences / total observed monthly means
  // Use linear ratio calculation
  // Cap at 1.0 to ensure mean_monthly stays in [0,1] range
  double ratio = sum_abs_diff / sum_obs;
  if(ratio > 1.0) ratio = 1.0;
  
  // Return 1 - ratio so that perfect match (ratio=0) returns 1.0
  // This maintains consistency with KGE where perfect performance = 1.0
  return(1.0 - ratio);
}
')

cppFunction('
double cv(NumericVector x) {
  int n = x.size();
  double sum = 0.0;
  double sum_sq = 0.0;
  
  for(int i = 0; i < n; i++){
    sum += x[i];
    sum_sq += x[i] * x[i];
  }
  
  double mean_val = sum / n;
  double variance = (sum_sq / n) - (mean_val * mean_val);
  double sd_val = sqrt(variance);
  
  return sd_val / mean_val;
}
')

# ------------------------------------------------------------------------------
# 0) User input and paths
#
# Pertinent columns of Quads to Extract from LOCA2.txt include:
#   column QID: the integer values of CRB quadrangles.
#   column NWLon: the longitude of the quadrangle northwest corner
#   column NWLat: the latitude of the quadrangle northwest corner
# ------------------------------------------------------------------------------
in_quads <- "G:/Quads to Extract from LOCA2.txt"

# Directory for writing output files
out_dir <- "G:/EvapModels/ETo_Oudin12pp/Training_data/"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# 1) Configuration
# ------------------------------------------------------------------------------
# Common span used for NA checks and outputs
span_start <- as.Date("1979-01-01")
span_end   <- as.Date("2018-12-31")

# Train/test windows (calibration uses TRAIN only; TEST is post-calibration evaluation only)
train_start <- as.Date("1999-01-01")
train_end   <- as.Date("2018-12-31")
test_start  <- as.Date("1979-01-01")
test_end    <- as.Date("1998-12-31")

# Decimal rounding for the summary output file (TSV file)
digits_par    <- 8   # parameters a01..a12, b01..b12, a_global_L1, b01_precalc..b12_precalc
digits_others <- 4   # other numeric diagnostics

# Oudin parameter bounds
lower_a <- 0.0001
upper_a <- 0.02
lower_b <- -30.0
upper_b <- 60.0

# ------------------------------------------------------------------------------
# 2) Establish helper functions
# ------------------------------------------------------------------------------

# FAO-56 extraterrestrial radiation (MJ m^-2 d^-1) with 365-day trigonometric terms.
# During leap years, there will be 366 days. When day of year (doy) equals 366,
# the equations will result in the same Ra as doy 1.
ra_mj <- function(lat_deg, doy) {
  Gsc <- 0.0820
  phi <- lat_deg * pi/180
  # Map leap day to day 1 to respect 365-day trigonometric basis
  doy[doy == 366] <- 1L
  dr <- 1 + 0.033 * cos(2*pi/365 * doy)
  delta <- 0.409 * sin(2*pi/365 * doy - 1.39)
  omega_s <- suppressWarnings(acos(-tan(phi) * tan(delta)))
  (24*60/pi) * Gsc * dr *
    (omega_s * sin(phi) * sin(delta) + cos(phi) * cos(delta) * sin(omega_s))
}

# KGE with three components: correlation, mean_monthly, and CV ratio.
# Accepts pre-calculated cv_obs for efficiency.
kge <- function(obs, sim, cv_obs, month_ix, mean_obs_monthly) {
  mean_obs <- mean(obs)
  mean_sim <- mean(sim)
  r <- cor(obs, sim)
  cv_sim <- cv(sim)
  cv_ratio <- cv_sim / cv_obs
  mean_monthly <- monthly_mean_err(sim, month_ix, mean_obs_monthly)
  
  if (!is.finite(r) || !is.finite(mean_monthly) || !is.finite(cv_ratio)) {
    return(list(KGE = -Inf, r = NA_real_, mean_monthly = NA_real_,
                cv_obs = cv_obs, cv_sim = NA_real_, cv_ratio = NA_real_,
                mean_obs = mean_obs, mean_sim = mean_sim))
  }
  KGE <- 1 - sqrt((r - 1)^2 + (mean_monthly - 1)^2 + (cv_ratio - 1)^2)
  list(KGE = KGE, r = r, mean_monthly = mean_monthly,
       cv_obs = cv_obs, cv_sim = cv_sim, cv_ratio = cv_ratio,
       mean_obs = mean_obs, mean_sim = mean_sim)
}

# KGE objective function for optimization.
# Accepts pre-calculated cv_obs and mean_obs_monthly for efficiency.
# Returns only scalar KGE value.
kge_objfunc <- function(obs, sim, cv_obs, month_ix, mean_obs_monthly) {
  r <- cor(obs, sim)
  mean_monthly <- monthly_mean_err(sim, month_ix, mean_obs_monthly)
  cv_sim <- cv(sim)
  cv_ratio <- cv_sim / cv_obs
  
  KGE <- 1 - sqrt((r - 1)^2 + (mean_monthly - 1)^2 + (cv_ratio - 1)^2)
  return(KGE)
}

# QDGC centers calculated from the Quadrangle northwest corner.
# The QDGC identifiers follow the same numbering system as the TWDB quadrangles,
# i.e., rows north to south, columns west to east; indices 1 through 16.
# There are 4 rows and 4 columns of QDGCs.
# The spatial arrangement of the rows and column QDGC numbers is as follows:
#  1,  2,  3,  4
#  5,  6,  7,  8
#  9, 10, 11, 12
# 13, 14, 15, 16
#
qdgc_centers <- function(nw_lon, nw_lat) {
  idx <- 1:16
  row <- ceiling(idx / 4)
  col <- idx - 4 * (row - 1)
  lon <- nw_lon + (col - 0.5) * 0.25
  lat <- nw_lat  - (row - 0.5) * 0.25
  data.frame(i = idx, row = row, col = col, lon_deg = lon, lat_deg = lat)
}

# Average temperature (Tavg) diagnostics (all days)
# This function is used to identify extreme or anomalous Tavg values.
# Oudin is sensitive to Tavg. The results of this check could be used
# to identify possibly erroneous input data that impact the calibration process.
tavg_diagnostics <- function(tavg_vec) {
  tavg_neg_days <- sum(tavg_vec < 0, na.rm = TRUE)
  q <- stats::quantile(tavg_vec, probs = c(0.25, 0.75, 0.99, 0.999),
                       na.rm = TRUE, names = FALSE, type = 7)
  q1 <- q[1]; q3 <- q[2]; p99 <- q[3]; p999 <- q[4]
  iqr <- q3 - q1; thr <- q3 + 3 * iqr
  tavg_outlier_days_iqr <- sum(tavg_vec > thr, na.rm = TRUE)
  list(
    tavg_neg_days = tavg_neg_days,
    tavg_p99 = as.numeric(p99),
    tavg_p999 = as.numeric(p999),
    tavg_max = suppressWarnings(max(tavg_vec, na.rm = TRUE)),
    tavg_outlier_days_iqr = tavg_outlier_days_iqr,
    tavg_outlier_flag = (tavg_neg_days > 0) || (tavg_outlier_days_iqr > 0)
  )
}

# ------------------------------------------------------------------------------
# Quads metadata (read once)
# ------------------------------------------------------------------------------
if (!file.exists(in_quads)) stop("Quads metadata file not found: ", in_quads)
quads <- fread(in_quads, sep = "\t", na.strings = c("NA"))
if (!all(c("QID","NWLon","NWLat") %in% names(quads))) stop("Quads metadata must have columns: QID, NWLon, NWLat")
if (!is.character(quads$QID)) quads[, QID := sprintf("%03d", as.integer(QID))]

# Make a list of items, created outside of the loop, to preserve from deletion.
.keep_names <- ls(all.names = TRUE)

# ------------------------------------------------------------------------------
# Loop for processing all quadrangles. Variable qid is the quadrangle number.
# ------------------------------------------------------------------------------
for (qid_num in qnum) {
  qid <- sprintf("%03d", as.integer(qid_num))
  cat(sprintf("\n--- Starting Q%s ---\n", qid))
  
  # Per-QID identifiers
  qdgc_ids <- sprintf("Q%s.%02d", qid, 1:16)  # QDGC columns
  col_quad <- sprintf("Q%s", qid)             # Column 4 = quad mean
  
  # Per-QID input files
  in_eto  <- sprintf("G:/gridMET/CRB_etr_grass.DAILY.1979-2024.Q%s.txt", qid)
  in_tmax <- sprintf("G:/LOCA2_QDGC/Training_data/temp_and_wind/livneh_lusu_tmax.1915-2018.Q%s.txt", qid)
  in_tmin <- sprintf("G:/LOCA2_QDGC/Training_data/temp_and_wind/livneh_lusu_tmin.1915-2018.Q%s.txt", qid)
  
  # Per-QID output files
  out_diag <- file.path(out_dir, sprintf("Oudin12pp_KGE_calibration_1979-2018_Q%s.tsv", qid))
  out_mod  <- file.path(out_dir, sprintf("Oudin12pp_KGE_calibrated.modeled_ETo.1979-2018.Q%s.txt", qid))
  out_ra   <- file.path(out_dir, sprintf("Oudin12pp_KGE_calibrated.Ra.1979-2018.Q%s.txt", qid))
  
  # ------------------------------------------------------------------------------
  # 3) Read and preprocess input files (ETo observations, Tmax, Tmin)
  # ------------------------------------------------------------------------------
  if (!file.exists(in_eto))  stop("ETo file not found: ", in_eto)
  if (!file.exists(in_tmax)) stop("Tmax file not found: ", in_tmax)
  if (!file.exists(in_tmin)) stop("Tmin file not found: ", in_tmin)
  
  # Read ETo (1979-2024)
  eto_full <- fread(in_eto, sep = "\t", na.strings = c("NA"))
  eto_full[, Date := as.Date(sprintf("%04d-%02d-%02d", Year, Month, Day))]
  eto <- eto_full[Date >= span_start & Date <= span_end]
  rm(eto_full)
  
  # Read Tmax, Tmin (1915-2018); subset to common span
  tmax_full <- fread(in_tmax, sep = "\t", na.strings = c("NA"))
  tmax_full[, Date := as.Date(sprintf("%04d-%02d-%02d", Year, Month, Day))]
  tmax <- tmax_full[Date >= span_start & Date <= span_end]
  rm(tmax_full)
  
  tmin_full <- fread(in_tmin, sep = "\t", na.strings = c("NA"))
  tmin_full[, Date := as.Date(sprintf("%04d-%02d-%02d", Year, Month, Day))]
  tmin <- tmin_full[Date >= span_start & Date <= span_end]
  rm(tmin_full)
  
  # Check that ETo, Tmax, and Tmin all have the same dates
  if (nrow(eto) != nrow(tmax) || nrow(eto) != nrow(tmin)) {
    stop("ETo, Tmax, and Tmin files have different number of rows for Q", qid)
  }
  if (!all(eto$Date == tmax$Date) || !all(eto$Date == tmin$Date)) {
    stop("ETo, Tmax, and Tmin dates do not match for Q", qid)
  }
  
  # ------------------------------------------------------------------------------
  # 4) Build common matrices and temporal indices
  # ------------------------------------------------------------------------------
  n_cells <- 16L
  n_days  <- nrow(eto)
  
  # Build matrices: each column is a QDGC (n_days x n_cells)
  obs_mat  <- as.matrix(eto[, ..qdgc_ids])
  tmax_mat <- as.matrix(tmax[, ..qdgc_ids])
  tmin_mat <- as.matrix(tmin[, ..qdgc_ids])
  
  # Tavg (daily)
  Tavg_mat <- 0.5 * (tmax_mat + tmin_mat)
  
  # Calculate Ra for each QDGC
  qrow <- quads[QID == qid]
  if (nrow(qrow) != 1) stop("Quadrangle ", qid, " not found or duplicated in quads metadata")
  centers <- qdgc_centers(qrow$NWLon, qrow$NWLat)
  
  doy_vec <- as.integer(format(eto$Date, "%j"))
  Ra_mat <- matrix(NA_real_, nrow = n_days, ncol = n_cells)
  for (j in 1:n_cells) {
    Ra_mat[, j] <- ra_mj(centers$lat_deg[j], doy_vec)
  }
  
  # Monthly indices (1..12) for the full span, TRAIN, and TEST
  month_ix <- as.integer(format(eto$Date, "%m"))
  is_train <- (eto$Date >= train_start & eto$Date <= train_end)
  is_test  <- (eto$Date >= test_start  & eto$Date <= test_end)
  month_ix_train <- month_ix[is_train]
  month_ix_test  <- month_ix[is_test]
  
  # ------------------------------------------------------------------------------
  # 5) Quadrangle-level aggregation for TRAIN period
  # ------------------------------------------------------------------------------
  obs_quad_train  <- rowMeans(obs_mat[is_train, , drop = FALSE], na.rm = TRUE)
  Tavg_quad_train <- rowMeans(Tavg_mat[is_train, , drop = FALSE], na.rm = TRUE)
  Ra_quad_train   <- rowMeans(Ra_mat[is_train, , drop = FALSE], na.rm = TRUE)
  
  # Pre-calculate observed monthly mean ETo and CV for quadrangle TRAIN period
  mean_obs_monthly_quad <- numeric(12)
  for (m in 1:12) {
    mean_obs_monthly_quad[m] <- mean(obs_quad_train[month_ix_train == m])
  }
  cv_obs_quad_train <- cv(obs_quad_train)
  
  # ------------------------------------------------------------------------------
  # 6) Quadrangle-level optimization (Levels 1–3)
  # ------------------------------------------------------------------------------
  quad_start_time <- Sys.time()
  skip_all_qdgcs <- FALSE
  
  # Initialize quadrangle results storage
  par_hat_quad <- rep(NA_real_, 24)
  names(par_hat_quad) <- c(sprintf("a%02d", 1:12), sprintf("b%02d", 1:12))
  quad_KGE <- NA_real_
  quad_r <- NA_real_
  quad_mean_monthly <- NA_real_
  quad_cv_obs <- NA_real_
  quad_cv_sim <- NA_real_
  quad_cv_ratio <- NA_real_
  
  # Initialize storage for all quadrangle optimization levels
  quad_L1_results <- vector("list", 7)  # 7 Level 1 optimizations
  quad_L2_results <- list(params = rep(NA_real_, 24), KGE = NA_real_, 
                          r = NA_real_, mean_monthly = NA_real_,
                          cv_obs = NA_real_, cv_sim = NA_real_, cv_ratio = NA_real_)
  quad_L3_results <- list(params = rep(NA_real_, 24), KGE = NA_real_, 
                          r = NA_real_, mean_monthly = NA_real_,
                          cv_obs = NA_real_, cv_sim = NA_real_, cv_ratio = NA_real_)
  
  # Parameter names
  a_names <- sprintf("a%02d", 1:12)
  b_names <- sprintf("b%02d", 1:12)
  
  # Check for any NA in quadrangle training data
  has_na_quad <- anyNA(obs_quad_train) || anyNA(Tavg_quad_train) || anyNA(Ra_quad_train)
  if (has_na_quad) {
    message("Quadrangle-level optimization skipped: NA present in aggregated quadrangle training data.")
    message("All 16 QDGCs will be skipped for this quadrangle.")
    skip_all_qdgcs <- TRUE
  } else {
    # -----------------------
    # Level 1: Seven quantile-based calibrations
    # Calculate monthly b parameters and optimize single 'a' parameter for each quantile
    # -----------------------
    prob_values <- c(0.0, 0.003, 0.005, 0.01, 0.025, 0.05, 0.10)
    par0_quad_L1_list <- vector("list", 7)
    names(par0_quad_L1_list) <- paste0("L1_", 1:7)
    
    for (idx in 1:7) {
      prob_val <- prob_values[idx]
      
      # Calculate b_precalc_quad for this probability
      b_precalc_quad <- rep(NA_real_, 12)
      names(b_precalc_quad) <- b_names
      
      for (m in 1:12) {
        Tavg_m <- Tavg_quad_train[month_ix_train == m]
        if (length(Tavg_m) == 0 || all(is.na(Tavg_m))) {
          b_precalc_quad[m] <- NA_real_
        } else {
          q_val <- quantile(Tavg_m, probs = prob_val, type = 7, na.rm = TRUE)
          b_precalc_quad[m] = -1.0 * q_val
        }
      }
      
      # Level 1: Optimize a_global with fixed b_precalc_quad
      if (!anyNA(b_precalc_quad)) {
        kge_train_quad_L1 <- function(a) {
          b_day <- b_precalc_quad[month_ix_train]
          sim_daily <- pmax(0, a * Ra_quad_train * (Tavg_quad_train + b_day))
          kge_objfunc(obs_quad_train, sim_daily, cv_obs_quad_train, month_ix_train, mean_obs_monthly_quad)
        }
        opt_quad_L1 <- try(optimize(f = kge_train_quad_L1, interval = c(lower_a, upper_a), maximum = TRUE),
                           silent = TRUE)
        
        if (!inherits(opt_quad_L1, "try-error") && is.finite(opt_quad_L1$maximum)) {
          a_L1_quad <- opt_quad_L1$maximum
          par0_quad_L1_list[[idx]] <- c(rep(a_L1_quad, 12), b_precalc_quad)
          names(par0_quad_L1_list[[idx]]) <- c(a_names, b_names)
          
          # Calculate KGE and components for this Level 1 result
          a_day_L1 <- rep(a_L1_quad, length(month_ix_train))
          b_day_L1 <- b_precalc_quad[month_ix_train]
          sim_daily_L1 <- pmax(0, a_day_L1 * Ra_quad_train * (Tavg_quad_train + b_day_L1))
          kge_L1_result <- kge(obs_quad_train, sim_daily_L1, cv_obs_quad_train, month_ix_train, mean_obs_monthly_quad)
          
          quad_L1_results[[idx]] <- list(
            params = par0_quad_L1_list[[idx]],
            KGE = kge_L1_result$KGE,
            r = kge_L1_result$r,
            mean_monthly = kge_L1_result$mean_monthly,
            cv_obs = kge_L1_result$cv_obs,
            cv_sim = kge_L1_result$cv_sim,
            cv_ratio = kge_L1_result$cv_ratio,
            tavg_quant = prob_val
          )
        } else {
          par0_quad_L1_list[[idx]] <- NULL  # Failed optimization
          quad_L1_results[[idx]] <- list(
            params = rep(NA_real_, 24),
            KGE = NA_real_,
            r = NA_real_,
            mean_monthly = NA_real_,
            cv_obs = NA_real_,
            cv_sim = NA_real_,
            cv_ratio = NA_real_,
            tavg_quant = prob_val
          )
        }
      } else {
        par0_quad_L1_list[[idx]] <- NULL  # Failed b_precalc calculation
        quad_L1_results[[idx]] <- list(
          params = rep(NA_real_, 24),
          KGE = NA_real_,
          r = NA_real_,
          mean_monthly = NA_real_,
          cv_obs = NA_real_,
          cv_sim = NA_real_,
          cv_ratio = NA_real_,
          tavg_quant = prob_val
        )
      }
    }
    
    # Check if all Level 1 calibrations failed
    if (all(sapply(par0_quad_L1_list, is.null))) {
      message("Quadrangle-level optimization skipped: all seven Level 1 calibrations failed.")
      message("All 16 QDGCs will be skipped for this quadrangle.")
      skip_all_qdgcs <- TRUE
    } else {
      # Bounds for 24 parameters
      lower_a_vec <- rep(lower_a, 12)
      upper_a_vec <- rep(upper_a, 12)
      lower_b_vec <- rep(lower_b, 12)
      upper_b_vec <- rep(upper_b, 12)
      names(lower_a_vec) <- a_names
      names(upper_a_vec) <- a_names
      names(lower_b_vec) <- b_names
      names(upper_b_vec) <- b_names
      lower_vec_24 <- c(lower_a_vec, lower_b_vec)
      upper_vec_24 <- c(upper_a_vec, upper_b_vec)
      
      kge_train_quad_24 <- function(par) {
        a_day <- par[1:12][month_ix_train]
        b_day <- par[13:24][month_ix_train]
        sim_daily <- pmax(0, a_day * Ra_quad_train * (Tavg_quad_train + b_day))
        -(kge_objfunc(obs_quad_train, sim_daily, cv_obs_quad_train, month_ix_train, mean_obs_monthly_quad))
      }
      
      # Build initial population for quadrangle Level 2
      set.seed(1234 + which(qnum == qid_num))  # Quadrangle-specific seed
      NP <- 240
      initialpop_quad <- matrix(NA_real_, nrow = NP, ncol = 24)
      # First 12 columns are a parameters
      for (kk in 1:12) {
        initialpop_quad[, kk] <- runif(NP, min = 0.001, max = 0.02)
      }
      # Next 12 columns are b parameters
      for (kk in 13:24) {
        initialpop_quad[, kk] <- runif(NP, min = -30.0, max = 30.0)
      }
      # Seed first 7 members with seven Level 1 solutions (or leave as random if failed)
      for (idx in 1:7) {
        if (!is.null(par0_quad_L1_list[[idx]])) {
          initialpop_quad[idx, ] <- par0_quad_L1_list[[idx]]
        }
        # If NULL (failed), leave as random values (already set above)
      }
      
      de_ctrl_quad <- RcppDE::DEoptim.control(
        strategy = 5, NP = NP, CR = 0.5, F = 0.1,
        itermax = 8000, reltol = 1e-6, steptol = 1000,
        trace = FALSE, initialpop = initialpop_quad
      )
      
      de_fit_quad <- try(RcppDE::DEoptim(kge_train_quad_24, lower = lower_vec_24, upper = upper_vec_24, control = de_ctrl_quad),
                         silent = TRUE)
      
      # Select best Level 1 result as initial starting point for Level 3
      par0_quad_24 <- NULL
      best_val <- -Inf
      for (idx in 1:7) {
        if (!is.null(par0_quad_L1_list[[idx]])) {
          val <- -kge_train_quad_24(par0_quad_L1_list[[idx]])  # negative because kge_train_quad_24 minimizes negative KGE
          if (val > best_val) {
            best_val <- val
            par0_quad_24 <- par0_quad_L1_list[[idx]]
          }
        }
      }
      
      # Compare Level 1 best with DE best and keep better for Level 3
      if (!inherits(de_fit_quad, "try-error") && is.finite(de_fit_quad$optim$bestval)) {
        val_par0 <- kge_train_quad_24(par0_quad_24)
        val_de   <- de_fit_quad$optim$bestval
        if (val_de <= val_par0) {
          par0_quad_24 <- setNames(as.numeric(de_fit_quad$optim$bestmem), c(a_names, b_names))
        }
        
        # Store Level 2 results
        a_day_L2 <- par0_quad_24[1:12][month_ix_train]
        b_day_L2 <- par0_quad_24[13:24][month_ix_train]
        sim_daily_L2 <- pmax(0, a_day_L2 * Ra_quad_train * (Tavg_quad_train + b_day_L2))
        kge_L2_result <- kge(obs_quad_train, sim_daily_L2, cv_obs_quad_train, month_ix_train, mean_obs_monthly_quad)
        
        quad_L2_results <- list(
          params = par0_quad_24,
          KGE = kge_L2_result$KGE,
          r = kge_L2_result$r,
          mean_monthly = kge_L2_result$mean_monthly,
          cv_obs = kge_L2_result$cv_obs,
          cv_sim = kge_L2_result$cv_sim,
          cv_ratio = kge_L2_result$cv_ratio
        )
      } else {
        warning("Quadrangle DEoptim failed. Proceeding to Level 3 with best Level 1 seed.")
        # Level 2 results remain NA (initialized above)
      }
      
      # Level 3: nlminb refinement
      safe_obj_quad <- function(p) {
        a_day <- p[1:12][month_ix_train]
        b_day <- p[13:24][month_ix_train]
        sim_daily <- pmax(0, a_day * Ra_quad_train * (Tavg_quad_train + b_day))
        v <- -(kge_objfunc(obs_quad_train, sim_daily, cv_obs_quad_train, month_ix_train, mean_obs_monthly_quad))
        if (!is.finite(v)) 1e6 else v
      }
      
      fit_nb_quad <- try(
        nlminb(
          start     = par0_quad_24,
          objective = safe_obj_quad,
          lower     = lower_vec_24,
          upper     = upper_vec_24,
          control   = list(
            eval.max = 4000,
            iter.max = 2000,
            rel.tol  = 1e-10,
            x.tol    = 1e-12
          )
        ),
        silent = TRUE
      )
      
      if (inherits(fit_nb_quad, "try-error") || !is.finite(fit_nb_quad$objective) || any(is.na(fit_nb_quad$par))) {
        par_hat_quad <- par0_quad_24
        # Level 3 results remain NA (initialized above)
      } else {
        par_hat_quad <- fit_nb_quad$par
        
        # Store Level 3 results
        a_day_L3 <- par_hat_quad[1:12][month_ix_train]
        b_day_L3 <- par_hat_quad[13:24][month_ix_train]
        sim_daily_L3 <- pmax(0, a_day_L3 * Ra_quad_train * (Tavg_quad_train + b_day_L3))
        kge_L3_result <- kge(obs_quad_train, sim_daily_L3, cv_obs_quad_train, month_ix_train, mean_obs_monthly_quad)
        
        quad_L3_results <- list(
          params = par_hat_quad,
          KGE = kge_L3_result$KGE,
          r = kge_L3_result$r,
          mean_monthly = kge_L3_result$mean_monthly,
          cv_obs = kge_L3_result$cv_obs,
          cv_sim = kge_L3_result$cv_sim,
          cv_ratio = kge_L3_result$cv_ratio
        )
      }
      names(par_hat_quad) <- c(a_names, b_names)
      
      # Calculate quadrangle-level KGE and components for diagnostics (using final result)
      a_day_quad <- par_hat_quad[1:12][month_ix_train]
      b_day_quad <- par_hat_quad[13:24][month_ix_train]
      sim_daily_quad <- pmax(0, a_day_quad * Ra_quad_train * (Tavg_quad_train + b_day_quad))
      kge_quad_result <- kge(obs_quad_train, sim_daily_quad, cv_obs_quad_train, month_ix_train, mean_obs_monthly_quad)
      # Store quadrangle KGE components
      quad_KGE <- kge_quad_result$KGE
      quad_r <- kge_quad_result$r
      quad_mean_monthly <- kge_quad_result$mean_monthly
      quad_cv_obs <- kge_quad_result$cv_obs
      quad_cv_sim <- kge_quad_result$cv_sim
      quad_cv_ratio <- kge_quad_result$cv_ratio
    }
  }
  
  quad_elapsed_mins <- round(as.numeric(difftime(Sys.time(), quad_start_time, units = "secs")) / 60, 2)
  message("Quadrangle-level calibration complete.   Elapsed minutes: ", quad_elapsed_mins)
  
  # ------------------------------------------------------------------------------
  # 7) Calibration loop (Levels 2–3) for each of the 16 QDGCs (n_cells)
  # ------------------------------------------------------------------------------
  diag_list   <- vector("list", n_cells)
  sim_mat_all <- matrix(NA_real_, nrow = n_days, ncol = n_cells, dimnames = list(NULL, qdgc_ids))
  
  # Parameter names
  a_quad_names <- sprintf("a%02d_quad", 1:12)
  b_quad_names <- sprintf("b%02d_quad", 1:12)
  
  for (j in 1:n_cells) {
    start_time <- Sys.time()
    qid_j <- qdgc_ids[j]
    
    # Initialize L2_iter to NA (will be set if QDGC Level 2 DE optimization succeeds)
    L2_iter <- NA_integer_
    
    # If quadrangle-level optimization failed, skip all QDGCs
    if (skip_all_qdgcs) {
      diag_list[[j]] <- do.call(
        data.table,
        c(
          list(qdgc_id = qid_j),
          # 12 a parameters as atomic columns
          setNames(as.list(rep(NA_real_, 12)), a_names),
          # 12 b parameters as atomic columns
          setNames(as.list(rep(NA_real_, 12)), b_names),
          # metrics
          list(
            # TRAIN
            KGE_cal = NA_real_, mean_monthly_cal = NA_real_,
            r_cal = NA_real_, cv_obs_cal = NA_real_, cv_sim_cal = NA_real_, cv_ratio_cal = NA_real_,
            # L2 iteration count
            L2_iter = NA_integer_,
            # TEST
            KGE_test = NA_real_, mean_monthly_test = NA_real_,
            r_test = NA_real_, cv_obs_test = NA_real_, cv_sim_test = NA_real_, cv_ratio_test = NA_real_
          ),
          # 12 boundary flags for a parameters
          setNames(as.list(rep(NA, 12)), paste0(a_names, "_at_boundary")),
          # 12 boundary flags for b parameters
          setNames(as.list(rep(NA, 12)), paste0(b_names, "_at_boundary")),
          # location & Tavg diagnostics
          list(
            lat_deg = centers$lat_deg[j], lon_deg = centers$lon_deg[j],
            tavg_neg_days = NA_integer_, tavg_p99 = NA_real_, tavg_p999 = NA_real_, tavg_max = NA_real_,
            tavg_outlier_days_iqr = NA_integer_, tavg_outlier_flag = NA,
            # reason
            note = "skipped: quadrangle-level optimization failed"
          ),
          # Quadrangle-level parameters (NA if quadrangle optimization failed)
          setNames(as.list(rep(NA_real_, 12)), a_quad_names),
          setNames(as.list(rep(NA_real_, 12)), b_quad_names),
          # Quadrangle-level KGE diagnostics (NA if quadrangle optimization failed)
          list(
            KGE_quad = NA_real_, r_quad = NA_real_, mean_monthly_quad = NA_real_,
            cv_obs_quad = NA_real_, cv_sim_quad = NA_real_, cv_ratio_quad = NA_real_
          ),
          # New columns for optimization level tracking
          list(
            Level = NA_integer_,
            tavg_quant = NA_real_
          )
        )
      )
      next
    }
    
    # Strict NA policy over full common span 1979-2018.
    # Skip a QDGC if it contains any input data with NA values.
    has_na_full <- anyNA(obs_mat[, j]) || anyNA(tmax_mat[, j]) || anyNA(tmin_mat[, j])
    if (has_na_full) {
      diag_list[[j]] <- do.call(
        data.table,
        c(
          list(qdgc_id = qid_j),
          # 12 a parameters as atomic columns
          setNames(as.list(rep(NA_real_, 12)), a_names),
          # 12 b parameters as atomic columns
          setNames(as.list(rep(NA_real_, 12)), b_names),
          # metrics
          list(
            # TRAIN
            KGE_cal = NA_real_, mean_monthly_cal = NA_real_,
            r_cal = NA_real_, cv_obs_cal = NA_real_, cv_sim_cal = NA_real_, cv_ratio_cal = NA_real_,
            # L2 iteration count
            L2_iter = NA_integer_,
            # TEST
            KGE_test = NA_real_, mean_monthly_test = NA_real_,
            r_test = NA_real_, cv_obs_test = NA_real_, cv_sim_test = NA_real_, cv_ratio_test = NA_real_
          ),
          # 12 boundary flags for a parameters
          setNames(as.list(rep(NA, 12)), paste0(a_names, "_at_boundary")),
          # 12 boundary flags for b parameters
          setNames(as.list(rep(NA, 12)), paste0(b_names, "_at_boundary")),
          # location & Tavg diagnostics
          list(
            lat_deg = centers$lat_deg[j], lon_deg = centers$lon_deg[j],
            tavg_neg_days = NA_integer_, tavg_p99 = NA_real_, tavg_p999 = NA_real_, tavg_max = NA_real_,
            tavg_outlier_days_iqr = NA_integer_, tavg_outlier_flag = NA,
            # reason
            note = "skipped: NA present in full common span 1979-2018"
          ),
          # Quadrangle-level parameters (show actual values since quadrangle optimization succeeded)
          setNames(as.list(par_hat_quad[1:12]), a_quad_names),
          setNames(as.list(par_hat_quad[13:24]), b_quad_names),
          # Quadrangle-level KGE diagnostics (show actual values since quadrangle optimization succeeded)
          list(
            KGE_quad = quad_KGE, r_quad = quad_r, mean_monthly_quad = quad_mean_monthly,
            cv_obs_quad = quad_cv_obs, cv_sim_quad = quad_cv_sim, cv_ratio_quad = quad_cv_ratio
          ),
          # New columns for optimization level tracking
          list(
            Level = NA_integer_,
            tavg_quant = NA_real_
          )
        )
      )
      next
    }
    
    # Extract data for QDGC j
    obs_j  <- obs_mat[, j]
    Tavg_j <- Tavg_mat[, j]
    Ra_j   <- Ra_mat[, j]
    
    # TRAIN and TEST subsets for QDGC j
    obs_train  <- obs_j[is_train]
    Tavg_train <- Tavg_j[is_train]
    Ra_train   <- Ra_j[is_train]
    obs_test   <- obs_j[is_test]
    Tavg_test  <- Tavg_j[is_test]
    Ra_test    <- Ra_j[is_test]
    
    # Pre-calculate observed monthly mean ETo and CV for TRAIN period
    mean_obs_monthly_train <- numeric(12)
    for (m in 1:12) {
      mean_obs_monthly_train[m] <- mean(obs_train[month_ix_train == m])
    }
    cv_obs_train <- cv(obs_train)
    
    # Pre-calculate observed monthly mean ETo and CV for TEST period
    mean_obs_monthly_test <- numeric(12)
    for (m in 1:12) {
      mean_obs_monthly_test[m] <- mean(obs_test[month_ix_test == m])
    }
    cv_obs_test <- cv(obs_test)
    
    # Checks for erroneous TRAIN data: zero mean on daily observed series
    # If detected, skip this QDGC because calibration cannot proceed meaningfully
    if (!is.finite(mean(obs_train)) || mean(obs_train) == 0) {
      diag_list[[j]] <- do.call(
        data.table,
        c(
          list(qdgc_id = qid_j),
          # 12 a parameters as atomic columns
          setNames(as.list(rep(NA_real_, 12)), a_names),
          # 12 b parameters as atomic columns
          setNames(as.list(rep(NA_real_, 12)), b_names),
          # metrics
          list(
            # TRAIN
            KGE_cal = NA_real_, mean_monthly_cal = NA_real_,
            r_cal = NA_real_, cv_obs_cal = NA_real_, cv_sim_cal = NA_real_, cv_ratio_cal = NA_real_,
            # L2 iteration count
            L2_iter = NA_integer_,
            # TEST
            KGE_test = NA_real_, mean_monthly_test = NA_real_,
            r_test = NA_real_, cv_obs_test = NA_real_, cv_sim_test = NA_real_, cv_ratio_test = NA_real_
          ),
          # 12 boundary flags for a parameters
          setNames(as.list(rep(NA, 12)), paste0(a_names, "_at_boundary")),
          # 12 boundary flags for b parameters
          setNames(as.list(rep(NA, 12)), paste0(b_names, "_at_boundary")),
          # location & Tavg diagnostics
          list(
            lat_deg = centers$lat_deg[j], lon_deg = centers$lon_deg[j],
            tavg_neg_days = NA_integer_, tavg_p99 = NA_real_, tavg_p999 = NA_real_, tavg_max = NA_real_,
            tavg_outlier_days_iqr = NA_integer_, tavg_outlier_flag = NA,
            # reason
            note = "skipped: zero or non-finite mean(obs_train)"
          ),
          # Quadrangle-level parameters
          setNames(as.list(par_hat_quad[1:12]), a_quad_names),
          setNames(as.list(par_hat_quad[13:24]), b_quad_names),
          # Quadrangle-level KGE diagnostics
          list(
            KGE_quad = quad_KGE, r_quad = quad_r, mean_monthly_quad = quad_mean_monthly,
            cv_obs_quad = quad_cv_obs, cv_sim_quad = quad_cv_sim, cv_ratio_quad = quad_cv_ratio
          ),
          # New columns for optimization level tracking
          list(
            Level = NA_integer_,
            tavg_quant = NA_real_
          )
        )
      )
      next
    }
    
    # Tavg diagnostics for this QDGC (over full common span)
    tavg_diag <- tavg_diagnostics(Tavg_j)
    
    # Use quadrangle-level parameters to seed Level 2 (first member of initialpop)
    par0_24 <- par_hat_quad
    
    # -----------------------
    # Level 2 (Differential Evolution): Global stochastic search on KGE surface within TRAINING data,
    # optimizing all 24 parameters (a01..a12, b01..b12). The first member of the initial population
    # is seeded with quadrangle-level optimal parameters (Level 3 results).
    # -----------------------
    
    # Set dynamic bounds for b parameters based on quadrangle results
    lower_b_vec <- pmax(-30.0, par_hat_quad[13:24] - 2.5)
    upper_b_vec <- pmin(60.0, par_hat_quad[13:24] + 2.5)
    names(lower_b_vec) <- b_names
    names(upper_b_vec) <- b_names
    
    # Set bounds for a parameters (unchanged)
    lower_a_vec <- rep(lower_a, 12)
    upper_a_vec <- rep(upper_a, 12)
    names(lower_a_vec) <- a_names
    names(upper_a_vec) <- a_names
    
    # Combined bounds for 24 parameters (a01..a12, b01..b12)
    lower_vec_24 <- c(lower_a_vec, lower_b_vec)
    upper_vec_24 <- c(upper_a_vec, upper_b_vec)
    
    kge_train_obj_24par <- function(par) {
      # par: a01..a12, b01..b12 (24 parameters)
      a_day <- par[1:12][month_ix_train]   # a for each TRAIN day
      b_day <- par[13:24][month_ix_train]  # b for each TRAIN day
      sim_daily <- pmax(0, a_day * Ra_train * (Tavg_train + b_day))
      -(kge_objfunc(obs_train, sim_daily, cv_obs_train, month_ix_train, mean_obs_monthly_train))  # Minimize negative KGE to maximize KGE
    }
    
    # Build initial random population and set the first 5 members with seeded values
    set.seed(1234 + j)  # Reproducible random number generator seed but distinct per QDGC
    NP <- 240  # 24 parameters × 10
    initialpop <- matrix(NA_real_, nrow = NP, ncol = 24)
    # Use narrower random initialization ranges for QDGC optimization to provide
    # localized adjustments around the quadrangle-level seed (first 5 members).
    # First 12 columns are a parameters
    for (kk in 1:12) {
      initialpop[, kk] <- runif(NP, min = 0.001, max = 0.02)
    }
    # Next 12 columns are b parameters
    for (kk in 13:24) {
      initialpop[, kk] <- runif(NP, min = -30.0, max = 30.0)
    }
    
    # Seed member 1 with exact quadrangle-level optimal parameters (Level 3 final)
    initialpop[1, ] <- par0_24
    
    # Seed members 2-5 with small proportional perturbations around quadrangle solution
    # Perturbations are ±10% of absolute parameter value, uniformly distributed
    # This provides diverse starting points near the high-quality quadrangle solution
    # while maintaining sufficient population diversity for exploration
    for (seed_idx in 2:5) {
      # Generate proportional noise: ±10% of absolute parameter value
      # Uniform random in [-0.1, 0.1] scaled by absolute value
      noise_factor <- runif(24, min = -0.1, max = 0.1)
      perturbation <- noise_factor * abs(par0_24)
      
      # Apply perturbation
      perturbed_params <- par0_24 + perturbation
      
      # Ensure perturbed values stay within dynamic bounds
      perturbed_params <- pmax(lower_vec_24, pmin(upper_vec_24, perturbed_params))
      
      initialpop[seed_idx, ] <- perturbed_params
    }
    
    # Set the controls for differential evolution. Use strategy 5: rand/1/bin with per generation dither.
    de_ctrl <- RcppDE::DEoptim.control(
      strategy = 5, NP = NP, CR = 0.5, F = 0.1,
      itermax = 8000, reltol = 1e-6, steptol = 1000,
      trace = FALSE, initialpop = initialpop
    )
    
    de_fit <- try(RcppDE::DEoptim(kge_train_obj_24par, lower = lower_vec_24, upper = upper_vec_24, control = de_ctrl),
                  silent = TRUE)
    
    # Extract iteration count from QDGC Level 2 DE optimization (if successful)
    if (!inherits(de_fit, "try-error") && !is.null(de_fit$optim$iter)) {
      L2_iter <- as.integer(de_fit$optim$iter)
    }
    
    # Prepare seed for Level 3 (use par0_24 or DE best, whichever is better)
    if (!inherits(de_fit, "try-error") && is.finite(de_fit$optim$bestval)) {
      # Keep better of (par0_24) and DE-best as start for Level 3
      val_par0 <- kge_train_obj_24par(par0_24)
      val_de   <- de_fit$optim$bestval
      if (val_de <= val_par0) {
        par0_24 <- setNames(as.numeric(de_fit$optim$bestmem), c(a_names, b_names))
      }
    } else {
      warning("DEoptim failed. Proceeding to Level 3 with initial seed.")
    }
    
    # -----------------------
    # Level 3 (nlminb): Local refinement of all 24 parameters; uses KGE calculated from daily
    # series over TRAINING data. Uses a "safe" wrapper so any non-finite objective returns a large penalty.
    # -----------------------
    safe_obj <- function(p) {
      a_day <- p[1:12][month_ix_train]
      b_day <- p[13:24][month_ix_train]
      sim_daily <- pmax(0, a_day * Ra_train * (Tavg_train + b_day))
      v <- -(kge_objfunc(obs_train, sim_daily, cv_obs_train, month_ix_train, mean_obs_monthly_train))
      if (!is.finite(v)) 1e6 else v
    }
    
    fit_nb <- try(
      nlminb(
        start     = par0_24,
        objective = safe_obj,
        lower     = lower_vec_24,
        upper     = upper_vec_24,
        control   = list(
          eval.max = 4000,
          iter.max = 2000,
          rel.tol  = 1e-10,
          x.tol    = 1e-12
        )
      ),
      silent = TRUE
    )
    
    if (inherits(fit_nb, "try-error") || !is.finite(fit_nb$objective) || any(is.na(fit_nb$par))) {
      a_hat <- setNames(as.list(par0_24[1:12]), a_names)
      b_hat <- setNames(as.list(par0_24[13:24]), b_names)
      note_txt <- "Level 3 failed; using Level 2 solution"
    } else {
      a_hat <- setNames(as.list(fit_nb$par[1:12]), a_names)
      b_hat <- setNames(as.list(fit_nb$par[13:24]), b_names)
      note_txt <- ""
    }
    
    # Boundary checks: a parameter at boundary if within 0.1% of range
    tol_a <- 0.001 * (upper_a - lower_a)
    at_bd_a <- setNames(as.list(abs(unlist(a_hat) - lower_a) < tol_a | abs(unlist(a_hat) - upper_a) < tol_a), paste0(a_names, "_at_boundary"))
    
    # Boundary checks: b parameter at boundary if within 0.1% of range (using dynamic bounds)
    tol_b_vec <- 0.001 * (upper_b_vec - lower_b_vec)
    at_bd_b <- setNames(as.list(abs(unlist(b_hat) - lower_b_vec) < tol_b_vec | abs(unlist(b_hat) - upper_b_vec) < tol_b_vec), paste0(b_names, "_at_boundary"))
    
    # -----------------------
    # Evaluate performance on TRAINING data
    # -----------------------
    a_day_train <- unlist(a_hat)[month_ix_train]
    b_day_train <- unlist(b_hat)[month_ix_train]
    sim_train <- pmax(0, a_day_train * Ra_train * (Tavg_train + b_day_train))
    met_cal <- kge(obs_train, sim_train, cv_obs_train, month_ix_train, mean_obs_monthly_train)
    
    # -----------------------
    # Evaluate performance on TEST data
    # -----------------------
    a_day_test <- unlist(a_hat)[month_ix_test]
    b_day_test <- unlist(b_hat)[month_ix_test]
    sim_test <- pmax(0, a_day_test * Ra_test * (Tavg_test + b_day_test))
    met_tst <- kge(obs_test, sim_test, cv_obs_test, month_ix_test, mean_obs_monthly_test)
    
    # -----------------------
    # Generate modeled ETo for full span (1979-2018)
    # -----------------------
    a_day_all <- unlist(a_hat)[month_ix]
    b_day_all <- unlist(b_hat)[month_ix]
    sim_all <- pmax(0, a_day_all * Ra_j * (Tavg_j + b_day_all))
    sim_mat_all[, j] <- sim_all
    
    # Collect diagnostics
    # Column order: qdgc_id, a01..a12, b01..b12, KGE metrics, boundary flags (a then b), diagnostics,
    # quadrangle-level parameters (a01_quad..a12_quad, b01_quad..b12_quad)
    diag_row <- data.table(qdgc_id = qid_j)
    
    # Add a01..a12
    for (nm in a_names) diag_row[, (nm) := a_hat[[nm]] ]
    # Add b01..b12
    for (nm in b_names) diag_row[, (nm) := b_hat[[nm]] ]
    
    # Add TRAIN metrics
    diag_row[, `:=`(
      KGE_cal = met_cal$KGE, mean_monthly_cal = met_cal$mean_monthly,
      r_cal = met_cal$r, cv_obs_cal = met_cal$cv_obs, 
      cv_sim_cal = met_cal$cv_sim, cv_ratio_cal = met_cal$cv_ratio
    )]
    # Add L2 iteration count
    diag_row[, L2_iter := L2_iter]
    # Add TEST metrics
    diag_row[, `:=`(
      KGE_test = met_tst$KGE, mean_monthly_test = met_tst$mean_monthly,
      r_test = met_tst$r, cv_obs_test = met_tst$cv_obs,
      cv_sim_test = met_tst$cv_sim, cv_ratio_test = met_tst$cv_ratio
    )]
    
    # Add boundary flags for a parameters
    for (nm in names(at_bd_a)) diag_row[, (nm) := at_bd_a[[nm]] ]
    # Add boundary flags for b parameters
    for (nm in names(at_bd_b)) diag_row[, (nm) := at_bd_b[[nm]] ]
    
    # Add location and Tavg diagnostics
    diag_row[, `:=`(
      lat_deg = centers$lat_deg[j], lon_deg = centers$lon_deg[j],
      tavg_neg_days = tavg_diag$tavg_neg_days,
      tavg_p99 = tavg_diag$tavg_p99, tavg_p999 = tavg_diag$tavg_p999,
      tavg_max = tavg_diag$tavg_max, tavg_outlier_days_iqr = tavg_diag$tavg_outlier_days_iqr,
      tavg_outlier_flag = tavg_diag$tavg_outlier_flag,
      note = note_txt
    )]
    
    # Add quadrangle-level parameters
    # Add all quadrangle 'a' parameters first
    for (m in 1:12) {
      diag_row[, (a_quad_names[m]) := par_hat_quad[m]]
    }
    # Then add all quadrangle 'b' parameters
    for (m in 1:12) {
      diag_row[, (b_quad_names[m]) := par_hat_quad[m + 12]]
    }
    # Add quadrangle-level KGE diagnostics
    diag_row[, `:=`(
      KGE_quad = quad_KGE,
      r_quad = quad_r,
      mean_monthly_quad = quad_mean_monthly,
      cv_obs_quad = quad_cv_obs,
      cv_sim_quad = quad_cv_sim,
      cv_ratio_quad = quad_cv_ratio
    )]
    
    # Add new columns for optimization level tracking
    diag_row[, `:=`(
      Level = NA_integer_,
      tavg_quant = NA_real_
    )]
    
    diag_list[[j]] <- diag_row
    
    elapsed_mins <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")) / 60, 2)
    cat("Calibration complete for ", qid_j, ".   Elapsed minutes: ", elapsed_mins, "\n", sep = "")
  }
  
  # ------------------------------------------------------------------------------
  # 8) Modify diagnostics table to include quadrangle optimization level results
  # ------------------------------------------------------------------------------
  diag_dt <- rbindlist(diag_list, use.names = TRUE, fill = TRUE)
  
  # Replace quadrangle-specific columns in rows 1-9 with optimization level results
  # Rows 1-7: Level 1 results (7 quantiles)
  # Row 8: Level 2 results
  # Row 9: Level 3 results
  # Rows 10-16: NA values
  
  for (row_idx in 1:7) {
    # Level 1 results
    L1_res <- quad_L1_results[[row_idx]]
    if (!is.null(L1_res$params) && !all(is.na(L1_res$params))) {
      for (m in 1:12) {
        set(diag_dt, i = row_idx, j = a_quad_names[m], value = L1_res$params[m])
        set(diag_dt, i = row_idx, j = b_quad_names[m], value = L1_res$params[m + 12])
      }
      set(diag_dt, i = row_idx, j = "KGE_quad", value = L1_res$KGE)
      set(diag_dt, i = row_idx, j = "r_quad", value = L1_res$r)
      set(diag_dt, i = row_idx, j = "mean_monthly_quad", value = L1_res$mean_monthly)
      set(diag_dt, i = row_idx, j = "cv_obs_quad", value = L1_res$cv_obs)
      set(diag_dt, i = row_idx, j = "cv_sim_quad", value = L1_res$cv_sim)
      set(diag_dt, i = row_idx, j = "cv_ratio_quad", value = L1_res$cv_ratio)
    }
    set(diag_dt, i = row_idx, j = "Level", value = 1L)
    set(diag_dt, i = row_idx, j = "tavg_quant", value = L1_res$tavg_quant)
  }
  
  # Row 8: Level 2 results
  if (!all(is.na(quad_L2_results$params))) {
    for (m in 1:12) {
      set(diag_dt, i = 8L, j = a_quad_names[m], value = quad_L2_results$params[m])
      set(diag_dt, i = 8L, j = b_quad_names[m], value = quad_L2_results$params[m + 12])
    }
    set(diag_dt, i = 8L, j = "KGE_quad", value = quad_L2_results$KGE)
    set(diag_dt, i = 8L, j = "r_quad", value = quad_L2_results$r)
    set(diag_dt, i = 8L, j = "mean_monthly_quad", value = quad_L2_results$mean_monthly)
    set(diag_dt, i = 8L, j = "cv_obs_quad", value = quad_L2_results$cv_obs)
    set(diag_dt, i = 8L, j = "cv_sim_quad", value = quad_L2_results$cv_sim)
    set(diag_dt, i = 8L, j = "cv_ratio_quad", value = quad_L2_results$cv_ratio)
  }
  set(diag_dt, i = 8L, j = "Level", value = 2L)
  set(diag_dt, i = 8L, j = "tavg_quant", value = NA_real_)
  
  # Row 9: Level 3 results
  if (!all(is.na(quad_L3_results$params))) {
    for (m in 1:12) {
      set(diag_dt, i = 9L, j = a_quad_names[m], value = quad_L3_results$params[m])
      set(diag_dt, i = 9L, j = b_quad_names[m], value = quad_L3_results$params[m + 12])
    }
    set(diag_dt, i = 9L, j = "KGE_quad", value = quad_L3_results$KGE)
    set(diag_dt, i = 9L, j = "r_quad", value = quad_L3_results$r)
    set(diag_dt, i = 9L, j = "mean_monthly_quad", value = quad_L3_results$mean_monthly)
    set(diag_dt, i = 9L, j = "cv_obs_quad", value = quad_L3_results$cv_obs)
    set(diag_dt, i = 9L, j = "cv_sim_quad", value = quad_L3_results$cv_sim)
    set(diag_dt, i = 9L, j = "cv_ratio_quad", value = quad_L3_results$cv_ratio)
  }
  set(diag_dt, i = 9L, j = "Level", value = 3L)
  set(diag_dt, i = 9L, j = "tavg_quant", value = NA_real_)
  
  # Rows 10-16: Set quadrangle columns to NA
  for (row_idx in 10:16) {
    for (m in 1:12) {
      set(diag_dt, i = row_idx, j = a_quad_names[m], value = NA_real_)
      set(diag_dt, i = row_idx, j = b_quad_names[m], value = NA_real_)
    }
    set(diag_dt, i = row_idx, j = "KGE_quad", value = NA_real_)
    set(diag_dt, i = row_idx, j = "r_quad", value = NA_real_)
    set(diag_dt, i = row_idx, j = "mean_monthly_quad", value = NA_real_)
    set(diag_dt, i = row_idx, j = "cv_obs_quad", value = NA_real_)
    set(diag_dt, i = row_idx, j = "cv_sim_quad", value = NA_real_)
    set(diag_dt, i = row_idx, j = "cv_ratio_quad", value = NA_real_)
    set(diag_dt, i = row_idx, j = "Level", value = NA_integer_)
    set(diag_dt, i = row_idx, j = "tavg_quant", value = NA_real_)
  }
  
  # Round a and b parameters (both QDGC and quadrangle-level) to 8 decimals
  par_cols <- c(a_names, b_names, a_quad_names, b_quad_names)
  for (p in par_cols) {
    if (p %in% names(diag_dt)) diag_dt[[p]] <- ifelse(is.na(diag_dt[[p]]), NA_real_, round(diag_dt[[p]], digits_par))
  }
  
  # Round other numeric metrics to 4 decimals
  no_round <- c("qdgc_id", paste0(a_names, "_at_boundary"), paste0(b_names, "_at_boundary"),
                "tavg_neg_days","tavg_outlier_days_iqr","tavg_outlier_flag","note","Level")
  for (cc in names(diag_dt)) {
    if (cc %in% c(no_round, par_cols)) next
    if (is.numeric(diag_dt[[cc]])) {
      diag_dt[[cc]] <- ifelse(is.na(diag_dt[[cc]]), NA_real_, round(diag_dt[[cc]], digits_others))
    }
  }
  # Logical to character for portability
  for (lc in c(paste0(a_names, "_at_boundary"), paste0(b_names, "_at_boundary"))) {
    if (lc %in% names(diag_dt)) diag_dt[, (lc) := as.character(get(lc))]
  }
  if ("tavg_outlier_flag" %in% names(diag_dt)) {
    diag_dt[, tavg_outlier_flag := as.character(tavg_outlier_flag)]
  }
  
  fwrite(diag_dt, out_diag, sep = "\t", na = "NA", quote = FALSE)
  
  # ------------------------------------------------------------------------------
  # 9) Write modeled ETo time series (full common span 1979-2018, daily time step)
  # ------------------------------------------------------------------------------
  mod_dt <- data.table(
    Year  = as.integer(format(eto$Date, "%Y")),
    Month = as.integer(format(eto$Date, "%m")),
    Day   = as.integer(format(eto$Date, "%d"))
  )
  for (j in 1:n_cells) mod_dt[, (qdgc_ids[j]) := sim_mat_all[, j]]
  
  # Column 4 = daily quad mean across available QDGCs (computed at full precision)
  quad_mean <- rowMeans(as.matrix(mod_dt[, ..qdgc_ids]), na.rm = TRUE)
  quad_mean[is.nan(quad_mean)] <- NA_real_
  mod_dt[, (col_quad) := quad_mean]
  setcolorder(mod_dt, c("Year","Month","Day", col_quad, qdgc_ids))
  
  # Round modeled ETo columns to 2 decimals (after means)
  num_cols_mod <- setdiff(names(mod_dt), c("Year","Month","Day"))
  mod_dt[, (num_cols_mod) := lapply(.SD, function(z) ifelse(is.na(z), NA_real_, round(z, 2))), .SDcols = num_cols_mod]
  
  fwrite(mod_dt, out_mod, sep = "\t", na = "NA", quote = FALSE)
  
  # ------------------------------------------------------------------------------
  # 10) Write Ra time series (full common span; 4 decimals)
  # ------------------------------------------------------------------------------
  ra_dt <- data.table(
    Year  = as.integer(format(eto$Date, "%Y")),
    Month = as.integer(format(eto$Date, "%m")),
    Day   = as.integer(format(eto$Date, "%d"))
  )
  for (j in 1:n_cells) ra_dt[, (qdgc_ids[j]) := Ra_mat[, j]]
  ra_qmean <- rowMeans(as.matrix(ra_dt[, ..qdgc_ids]), na.rm = TRUE)
  ra_dt[, (col_quad) := ra_qmean]
  setcolorder(ra_dt, c("Year","Month","Day", col_quad, qdgc_ids))
  
  # Round numeric columns (except date parts) to 4 decimals
  num_cols_ra <- setdiff(names(ra_dt), c("Year","Month","Day"))
  for (cc in num_cols_ra) {
    ra_dt[[cc]] <- ifelse(is.na(ra_dt[[cc]]), NA_real_, round(ra_dt[[cc]], 4))
  }
  
  fwrite(ra_dt, out_ra, sep = "\t", na = "NA", quote = FALSE)
  
  # ------------------------------------------------------------------------------
  # 11) Summary message
  # ------------------------------------------------------------------------------
  message("Done for Q", qid, ".")
  message("Diagnostics (Oudin 12 parameter pairs; train/test with daily KGE): ", out_diag)
  message("Modeled ETo (Oudin 12 parameter pairs; 1979-2018, daily time step): ", out_mod)
  message("Ra series (1979-2018, daily time step): ", out_ra)
  
  # Memory hygiene between quadrangles.
  # Keep the information that was developed outside of the loop.
  rm(list = setdiff(ls(all.names = TRUE), c(.keep_names, "qid_num")))
  gc()
}






