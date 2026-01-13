# ---- libraries ----
suppressPackageStartupMessages({
  library(MASS)      # mvrnorm
  library(survival)  # (not strictly necessary here, but kept for consistency)
})

# -------------------------
# global configs
# -------------------------
N <- 1000            # number of subjects
K <- 20              # repeated measures per subject
cens_horiz <- 50     # administrative censoring horizon
insp.rate <- 2       # inspection time rate (exponential increments)

set.seed(123)

# ---- longitudinal model true params ----
sigma.y <- 0.6
# betas: intercept, time, X1, X2, X3 (we will use nonlinear transforms for X1-3)
betas <- c("(Intercept)" = 5.6, "time" = -0.45, "X1" = -0.25, "X2" = -0.11, "X3" = -0.3)

# random effects covariance (intercept + slope)
D <- matrix(c(1, 0.5,
              0.5, 1), 2, 2)
D <- (D + t(D)) / 2

# ---- survival model true params ----
gammas <- c("(Intercept)" = -8, "X1" = 1.90, "X2" = 2.15, "X3" = 2.65)
phi <- 2        # Weibull shape
alpha0 <- 0.07  # association

# -------------------------
# baseline covariates (binary as in your script)
# -------------------------
X1 <- rbinom(N, 1, 0.60)
X2 <- rbinom(N, 1, 0.50)
X3 <- rbinom(N, 1, 0.55)
X4 <- rbinom(N, 1, 0.50)
X5 <- rbinom(N, 1, 0.65)
X6 <- rbinom(N, 1, 0.45)
X7 <- rbinom(N, 1, 0.55)
X8 <- rbinom(N, 1, 0.70)
X9 <- rbinom(N, 1, 0.60)
X10 <- rbinom(N, 1, 0.50)

base_dat <- data.frame(
  id = 1:N,
  X1 = X1, X2 = X2, X3 = X3, X4 = X4, X5 = X5,
  X6 = X6, X7 = X7, X8 = X8, X9 = X9, X10 = X10
)

# -------------------------
# random effects
# b[i,1] = b0 (random intercept), b[i,2] = b1 (random slope)
# -------------------------
b <- mvrnorm(N, mu = rep(0, 2), Sigma = D)
colnames(b) <- c("b0", "b1")

# -------------------------
# generate inspection times for each subject
# -------------------------
times_mat <- replicate(N, cumsum(c(0, rexp(n = K - 1, rate = insp.rate))))
times_dat <- data.frame(
  id = rep(1:N, each = K),
  time = as.vector(times_mat)
)

# max.FUtime for root-finding upper bound
max.FUtime <- max(times_mat) + 2 * IQR(as.vector(times_mat))
if (!is.finite(max.FUtime) || max.FUtime <= 0) max.FUtime <- cens_horiz

# merge to form full longitudinal design grid
DF <- merge(times_dat, base_dat, by = "id", all.x = TRUE)
DF <- DF[order(DF$id, DF$time), ]

# -------------------------
# nonlinear transforms used consistently
# (binary covariates -> still nonlinear but mild; keeps your design)
# -------------------------
nonlin_X1 <- function(x) x^2
nonlin_X2 <- function(x) log(x + 1)   # log(1) or log(2)
nonlin_X3 <- function(x) sin(x)       # sin(0) or sin(1)

# -------------------------
# longitudinal mean function mu_y(i, t)
# -------------------------
mu_y <- function(i, t) {
  x1n <- nonlin_X1(X1[i])
  x2n <- nonlin_X2(X2[i])
  x3n <- nonlin_X3(X3[i])
  
  # fixed + random
  fixed <- betas["(Intercept)"] +
    betas["time"] * t +
    betas["X1"] * x1n +
    betas["X2"] * x2n +
    betas["X3"] * x3n
  
  rand <- b[i, "b0"] + b[i, "b1"] * t
  fixed + rand
}

# -------------------------
# survival part: inverse survival equation root
# invS(t) = \int_0^t hazard(s) ds + log(1-u) = 0
# hazard uses current longitudinal latent f1(s) = mu_y(i,s) WITHOUT measurement noise
# -------------------------
invS <- function(t, u, i) {
  h <- function(s) {
    # subject-specific survival linear predictor
    eta_t_i <- gammas["(Intercept)"] +
      gammas["X1"] * X1[i] +
      gammas["X2"] * X2[i] +
      gammas["X3"] * X3[i]
    
    # longitudinal latent trajectory (no noise)
    f1 <- mu_y(i, s)
    
    # Weibull baseline hazard with association
    haz <- exp(log(phi) + (phi - 1) * log(s) + eta_t_i + f1 * alpha0)
    haz
  }
  
  # numerical integration
  val <- integrate(h, lower = 0, upper = t, subdivisions = 200L, rel.tol = 1e-6)$value
  val + log(1 - u)
}

# -------------------------
# simulate survival times via root finding
# -------------------------
u <- runif(N)
trueTimes <- rep(NA_real_, N)

for (i in 1:N) {
  Root <- try(
    uniroot(invS, interval = c(1e-5, max.FUtime), u = u[i], i = i, extendInt = "upX")$root,
    silent = TRUE
  )
  trueTimes[i] <- if (inherits(Root, "try-error")) Inf else Root
}

cat("\n[Debug] trueTimes summary:\n")
print(summary(trueTimes))
cat("\n[Debug] event time generation u summary:\n")
print(summary(u))

# censoring
Ctimes <- runif(N, 0, cens_horiz)
Time <- pmin(trueTimes, Ctimes, rep(cens_horiz, N))
event <- ifelse(trueTimes <= Time, 1, 0)

cat("\n[Debug] event table:\n")
print(table(event))

# -------------------------
# construct final observed longitudinal dataset:
# keep rows with time <= Time[i], and add an event-time row if event==1
# -------------------------
DF$Time <- Time[DF$id]
DF$event <- event[DF$id]

# keep observed times within follow-up
keep <- DF$time <= DF$Time
dat <- DF[keep, ]

# simulate longitudinal measurement y on kept rows
dat$y <- NA_real_
for (r in 1:nrow(dat)) {
  i <- dat$id[r]
  t <- dat$time[r]
  dat$y[r] <- rnorm(1, mean = mu_y(i, t), sd = sigma.y)
}

# add event-time observation (exactly at Time) for those with event==1
# to avoid duplicates: only add if event time not already present
tol <- 1e-10
add_rows <- list()
cnt <- 0

for (i in 1:N) {
  if (event[i] == 1) {
    ti <- Time[i]
    # check if already have time == event time
    has_exact <- any(abs(dat$time[dat$id == i] - ti) < tol)
    if (!has_exact) {
      # take last observed row before event time to copy baseline covariates, then set time=event time
      cand <- dat[dat$id == i & dat$time <= ti, ]
      if (nrow(cand) == 0) next
      new_row <- cand[which.max(cand$time), , drop = FALSE]
      new_row$time <- ti
      new_row$Time <- ti
      new_row$event <- 1
      new_row$y <- rnorm(1, mean = mu_y(i, ti), sd = sigma.y)
      
      cnt <- cnt + 1
      add_rows[[cnt]] <- new_row
    }
  }
}

if (length(add_rows) > 0) {
  dat <- rbind(dat, do.call(rbind, add_rows))
}

# sort
dat <- dat[order(dat$id, dat$time), ]
row.names(dat) <- NULL

names(dat)[names(dat) == "time"] <- "times"
names(dat)[names(dat) == "Time"] <- "tte"
names(dat)[names(dat) == "event"] <- "label"

cat("\n[Debug] final dat head:\n")
print(head(dat))

# -------------------------
# save complete dataset (no missingness)
# -------------------------
write.csv(dat, file = "nonlin_long_term_complete.csv", row.names = FALSE)
cat("\nSaved complete data: nonlin_long_term_complete.csv\n")

# ============================================================
# Part 2: MCAR missingness generation
# ============================================================

# ---- MCAR function ----
add_mcar_missing <- function(dat, miss_rate, cols_miss, seed = 1) {
  set.seed(seed)
  out <- dat
  
  # sanity: only allow existing columns
  cols_miss <- intersect(cols_miss, colnames(out))
  if (length(cols_miss) == 0) stop("cols_miss not found in dat.")
  
  n <- nrow(out)
  d <- length(cols_miss)
  
  total_cells <- n * d
  n_miss <- floor(total_cells * miss_rate)
  
  idx <- sample.int(total_cells, size = n_miss, replace = FALSE)
  row_idx <- ((idx - 1) %% n) + 1
  col_idx <- ((idx - 1) %/% n) + 1
  cols <- cols_miss[col_idx]
  
  for (k in seq_along(idx)) {
    out[row_idx[k], cols[k]] <- NA
  }
  out
}

# ---- choose columns to be missing (MCAR) ----
cols_miss <- c("y")

# ---- generate datasets for different missing rates ----
rates <- c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

for (p in rates) {
  dat_miss <- if (p == 0) dat else add_mcar_missing(dat, miss_rate = p, cols_miss = cols_miss, seed = 123 + round(p * 100))
  out_name <- sprintf("nonlin_long_term_miss%02d.csv", as.integer(p * 100))
  write.csv(dat_miss, file = out_name, row.names = FALSE)
  
  # quick check: realized missing rate on selected columns
  realized <- mean(is.na(as.matrix(dat_miss[, cols_miss, drop = FALSE])))
  cat(sprintf("Saved: %-28s | target=%.2f  realized=%.4f | cols=%s\n",
              out_name, p, realized, paste(cols_miss, collapse = ",")))
}

cat("\nDone.\n")
