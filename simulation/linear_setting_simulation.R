# Load required libraries
library(MASS)      # For multivariate normal generation
library(survival)  # For survival analysis
library(writexl)   # For saving Excel files

# Define inverse survival function invS
invS <- function(t, u, i) {
  # Inverse survival function used to generate survival times
  # t: time
  # u: value drawn from Uniform[0, 1]
  # i: patient id
  
  h <- function(s) {
    # Hazard function of the survival model
    X1_i <- X1[i]
    X2_i <- X2[i]
    X3_i <- X3[i]
    XX <- cbind(1, s, X1_i, X2_i, X3_i)
    ZZ <- cbind(1, s)
    f1 <- as.vector(XX %*% betas + rowSums(ZZ * b[rep(i, nrow(ZZ)), ]))
    haz <- exp(log(phi) + (phi - 1) * log(s) + eta.t[i] + f1 * alpha0)
    return(haz)
  }
  
  integrate(h, lower = 0, upper = t)$value + log(1 - u)
}

# Simulation parameter settings
N <- 1000               # Number of individuals
K <- 20                 # Number of repeated measurements per individual
cens_horiz <- 20        # Maximum follow-up time
insp.rate <- 2          # Rate for inspection times
max_time_length <- 50   # Maximum length for longitudinal time generation

# True parameters for the longitudinal model
sigma.y <- 0.6  # Measurement error standard deviation
betas <- c("(Intercept)" = 5.6, "time" = -0.45, "X1" = -0.25, "X2" = -0.11, "X3" = -0.3)
D <- matrix(c(1, 0.5, 0.5, 1), 2, 2)  # Covariance matrix of random effects
D <- (D + t(D)) / 2

# True parameters for the survival model
gammas <- c("(Intercept)" = -8, "X1" = 1.90, "X2" = 2.15, "X3" = 2.65)
phi <- 2          # Shape parameter of the Weibull distribution
alpha0 <- 0.07    # Association parameter

# Simulate baseline covariates
set.seed(123)  # Set random seed for reproducibility
X1 <- rbinom(N, size = 1, prob = 0.60)
X2 <- rbinom(N, size = 1, prob = 0.50)
X3 <- rbinom(N, size = 1, prob = 0.55)
X4 <- rbinom(N, size = 1, prob = 0.50)
X5 <- rbinom(N, size = 1, prob = 0.65)
X6 <- rbinom(N, size = 1, prob = 0.45)
X7 <- rbinom(N, size = 1, prob = 0.55)
X8 <- rbinom(N, size = 1, prob = 0.70)
X9 <- rbinom(N, size = 1, prob = 0.60)
X10 <- rbinom(N, size = 1, prob = 0.50)

# Simulate random effects
b <- mvrnorm(N, rep(0, nrow(D)), D)

# Generate longitudinal time points and restrict to maximum length
times <- replicate(
  N,
  {
    temp_times <- cumsum(c(0, rexp(n = K - 1, rate = insp.rate)))
    temp_times[temp_times <= max_time_length]  # Restrict time points to be within max_time_length
  },
  simplify = FALSE
)

# Convert times to a data frame
times_dat <- data.frame(id = rep(1:N, sapply(times, length)), time = unlist(times))

# Baseline covariates part
base_dat <- data.frame(id = 1:N, X1, X2, X3)
DF <- merge(times_dat, base_dat, by = "id", all.x = TRUE)

# Longitudinal model design matrices
X <- model.matrix(~ time + X1 + X2 + X3, data = DF)
Z <- model.matrix(~ time, data = DF)

id <- rep(1:N, sapply(times, length))
eta.y <- as.vector(X %*% betas + rowSums(Z * b[id, ]))

# Simulate survival data
W <- cbind("(Intercept)" = 1, "X1" = X1, "X2" = X2, "X3" = X3)
eta.t <- as.vector(W %*% gammas)

y <- rnorm(nrow(DF), eta.y, sigma.y)

# Generate survival times
u <- runif(N)
trueTimes <- numeric(N)
for (i in 1:N) {
  Root <- try(
    uniroot(invS, interval = c(1e-05, max_time_length), u = u[i], i = i, extendInt = "upX")$root,
    TRUE
  )
  trueTimes[i] <- ifelse(inherits(Root, "try-error"), Inf, Root)
}

# Print debugging information
print(summary(trueTimes))
print(summary(u))

# Generate censoring times and event indicators
Ctimes <- runif(N, 0, cens_horiz)
Time <- pmin(trueTimes, Ctimes, rep(cens_horiz, N))
event <- ifelse(trueTimes <= Time, 1, 0)

# Print distribution of event indicators
print(table(event))

# Create final dataset
Time_dat <- data.frame(id = rep(1:N, sapply(times, length)), Time = rep(Time, sapply(times, length)))
ind <- times_dat$time <= Time_dat$Time
y <- y[ind]
X <- X[ind, , drop = FALSE]
Z <- Z[ind, , drop = FALSE]
id <- id[ind]
id <- match(id, unique(id))

dat <- DF[ind, ]
dat$id <- id
dat$y <- y
dat$Time <- Time[id]
dat$event <- event[id]

# Add an observation at the event time for individuals with an event
for (i in 1:N) {
  if (event[i] == 1) {
    event_time <- Time[i]
    # Find the last measurement time before the event
    last_time <- max(DF$time[DF$id == i & DF$time <= event_time])
    new_row <- DF[DF$id == i & DF$time == last_time, ]
    new_row$time <- event_time
    new_row$id <- i
    
    # Update linear predictor new_eta_y
    new_eta_y <- betas["(Intercept)"] +
      new_row$time * betas["time"] +
      new_row$X1 * betas["X1"] +
      new_row$X2 * betas["X2"] +
      new_row$X3 * betas["X3"] +
      new_row$time * b[i, 1] +
      b[i, 2] * new_row$time
    
    # Simulate a new longitudinal measurement at event time
    new_row$y <- rnorm(1, new_eta_y, sigma.y)
    
    # Set Time and event for this row
    new_row$Time <- event_time
    new_row$event <- 1
    
    # Ensure all columns are consistent and bind to existing data
    dat <- rbind(dat, new_row)
  }
}

# Sort final dataset by id and time
dat <- dat[order(dat$id, dat$time), ]

# Inspect the final dataset
print(head(dat))

# Save final data frame as CSV
write.csv(dat, file = "long_term.csv", row.names = FALSE)

##################simulation_completed##############################
