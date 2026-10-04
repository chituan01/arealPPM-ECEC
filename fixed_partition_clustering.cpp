#include "fixed_partition_clustering.h"
#include <algorithm>
#include <cmath>
#include <vector>

// [[Rcpp::export]]
List Gibbs_FixedPartition(arma::mat y, arma::cube X_O, arma::cube X_E,
                          arma::sp_mat W, arma::uvec optimal_partition,
                          List hyper_param, List mcmc_param, List flags) {
  // 1. Unpack MCMC params
  int burn_in = mcmc_param["burn_in"];
  int n_save = mcmc_param["n_save"];
  int thin = mcmc_param["thin"];
  int n_tot = burn_in + n_save * thin;

  // 2. Unpack Flags
  bool use_fixed_E = flags["use_fixed_E"];
  bool use_spatial = flags["use_spatial"];
  bool random_xi = flags["random_xi"];

  // 3. Unpack Data dimensions
  int I = y.n_rows;
  int T = y.n_cols;
  int p1 = X_O.n_cols;
  int p2 = X_E.n_cols;

  // 4. Unpack Hyperparameters
  double a_sigma2 = hyper_param["a_sigma2"];
  double b_sigma2 = hyper_param["b_sigma2"];
  arma::vec mu_O = hyper_param["mu_O"];
  arma::mat Sigma_O = hyper_param["Sigma_O"];
  arma::mat Sigma_O_inv;
  if (p1 > 0) {
    Sigma_O_inv = arma::inv_sympd(Sigma_O);
  }

  arma::vec m_E(p2, arma::fill::zeros);
  double a_Sigma_E = 0.0, b_Sigma_E = 0.0;
  arma::vec mu_E(p2, arma::fill::zeros);
  arma::vec sigma2_E(p2, arma::fill::ones);
  arma::mat Sigma_E_inv_fixed; // for log-like computation
  if (use_fixed_E) {
    mu_E = as<arma::vec>(hyper_param["mu_E"]);
    arma::mat Sigma_E = hyper_param["Sigma_E"];
    sigma2_E = Sigma_E.diag();
    if (p2 > 0) {
      Sigma_E_inv_fixed = arma::inv_sympd(Sigma_E);
    }
  } else {
    m_E = as<arma::vec>(hyper_param["m_E"]);
    a_Sigma_E = hyper_param["a_Sigma_E"];
    b_Sigma_E = hyper_param["b_Sigma_E"];
    mu_E = m_E;
    sigma2_E.fill(b_Sigma_E / (a_Sigma_E - 1.0));
  }

  double a_tau2 = 0.0, b_tau2 = 0.0, alpha_xi = 0.0, beta_xi = 0.0, xi = 0.0;
  if (use_spatial) {
    a_tau2 = hyper_param["a_tau2"];
    b_tau2 = hyper_param["b_tau2"];
    if (random_xi) {
      alpha_xi = hyper_param["alpha_xi"];
      beta_xi = hyper_param["beta_xi"];
      xi = 0.5; // initialize
    } else {
      xi = hyper_param["xi"];
    }
  }

  // 5. Initialize Parameters
  double sigma2 = b_sigma2 / (a_sigma2 - 1);
  double temp = sigma2;

  double tau2 = 0.1;
  if (!use_spatial) {
    tau2 = 0;
  }

  arma::vec beta_O = mu_O;
  if (p1 > 0) {
    arma::mat L_O = arma::chol(Sigma_O, "lower");
    beta_O = mu_O + L_O * arma::randn(p1);
  }

  int K = optimal_partition.max() + 1; // 0-indexed optimal_partition
  arma::uvec s = optimal_partition;

  // Initialize beta_star with K rows
  arma::mat beta_star = arma::randn(K, p2);
  if (!use_fixed_E) {
    arma::mat L_E = arma::chol(arma::diagmat(sigma2_E), "lower");
    for (int j = 0; j < K; j++) {
      beta_star.row(j) = (mu_E + L_E * arma::randn(p2)).t();
    }
  } else {
    arma::mat Sigma_E = hyper_param["Sigma_E"];
    arma::mat L_E = arma::chol(Sigma_E, "lower");
    for (int j = 0; j < K; j++) {
      beta_star.row(j) = (mu_E + L_E * arma::randn(p2)).t();
    }
  }

  arma::vec u(I, arma::fill::zeros);

  // Prepare Q matrices for spatial
  arma::sp_mat diagI(I, I);
  diagI.eye();
  arma::vec aux_vec(I, arma::fill::ones);
  arma::sp_mat sumW(I, I);
  sumW.diag() = W * aux_vec;

  arma::sp_mat Q_xi = xi * (sumW - W) + (1.0 - xi) * diagI;

  double current_log_det_Q = 0.0;
  if (use_spatial) {
    double sign;
    arma::log_det(current_log_det_Q, sign, arma::mat(Q_xi));
  }

  // MH adaptive parameters for xi
  double s_xi = 0.01;
  double xi_accept = 0;
  int xi_count = 0;

  // Prepare outputs
  arma::mat beta_O_out(n_save, p1);
  List beta_star_out(n_save);
  arma::mat u_out(n_save, I);
  arma::vec sigma2_out(n_save);
  arma::vec tau2_out(n_save);
  arma::vec xi_out(n_save);
  arma::mat mu_E_out(n_save, p2);
  arma::mat sigma2_E_out(n_save, p2);
  arma::vec log_lik_out(n_save);
  arma::mat log_lik_ind_out(n_save, I, arma::fill::zeros);
  arma::vec log_post_out(n_save);

  int iter = 0;

  // Main Loop
  for (int g = 0; g < n_tot; g++) {

    // 1. Update beta_O
    if (p1 > 0) {
      arma::mat S_beta_prec_O = Sigma_O_inv;
      arma::vec m_beta_O = Sigma_O_inv * mu_O;

      for (int t = 0; t < T; t++) {
        arma::mat X_O_t = X_O.slice(t);
        arma::mat X_E_t = X_E.slice(t);
        arma::vec y_col_t = y.col(t);

        arma::vec y_resid_O_t = y_col_t - u;
        for (int i = 0; i < I; i++) {
          y_resid_O_t(i) -= arma::dot(X_E_t.row(i), beta_star.row(s(i)));
        }

        S_beta_prec_O += (X_O_t.t() * X_O_t) / sigma2;
        m_beta_O += (X_O_t.t() * y_resid_O_t) / sigma2;
      }

      arma::mat L_upper_O = arma::chol(S_beta_prec_O);
      arma::mat L_lower_O = L_upper_O.t();
      m_beta_O = arma::solve(arma::trimatl(L_lower_O), m_beta_O);
      beta_O =
          arma::solve(arma::trimatu(L_upper_O), arma::randn(p1) + m_beta_O);
    }

    // 2. Update beta_star_j
    arma::mat Sigma_E_inv = arma::diagmat(1.0 / sigma2_E);

    for (int j = 0; j < K; j++) {
      arma::uvec idx = arma::find(s == j);
      if (idx.n_elem > 0) {
        arma::mat S_beta_prec_j = Sigma_E_inv;
        arma::vec m_beta_j = Sigma_E_inv * mu_E;

        for (int t = 0; t < T; t++) {
          arma::mat X_O_j_t = X_O.slice(t).rows(idx);
          arma::mat X_E_j_t = X_E.slice(t).rows(idx);
          arma::vec y_col_t = y.col(t);
          arma::vec y_j_t = y_col_t.elem(idx);
          arma::vec u_j = u.elem(idx);

          arma::vec y_resid_j_t = y_j_t - X_O_j_t * beta_O - u_j;

          S_beta_prec_j += (X_E_j_t.t() * X_E_j_t) / sigma2;
          m_beta_j += (X_E_j_t.t() * y_resid_j_t) / sigma2;
        }

        arma::mat L_upper_j = arma::chol(S_beta_prec_j);
        arma::mat L_lower_j = L_upper_j.t();

        m_beta_j = arma::solve(arma::trimatl(L_lower_j), m_beta_j);
        beta_star.row(j) =
            arma::solve(arma::trimatu(L_upper_j), arma::randn(p2) + m_beta_j)
                .t();
      }
    }

    // Step 3 (Update s - Allocation vector) has been removed because the
    // partition is fixed.

    // 4. Update u
    if (use_spatial) {
      arma::vec y_resid_u(I, arma::fill::zeros);
      for (int t = 0; t < T; t++) {
        arma::mat X_O_t = X_O.slice(t);
        arma::mat X_E_t = X_E.slice(t);
        arma::vec y_col_t = y.col(t);

        arma::vec y_resid_u_t = y_col_t - X_O_t * beta_O;
        for (int i = 0; i < I; i++) {
          y_resid_u_t(i) -= arma::dot(X_E_t.row(i), beta_star.row(s(i)));
        }
        y_resid_u += y_resid_u_t;
      }

      // Precision matrix = Q_xi/tau2 + T*I/sigma2
      arma::mat Prec_u = arma::mat((Q_xi / tau2) + (T * diagI / sigma2));
      arma::vec rhs = y_resid_u / sigma2;

      arma::mat L_upper_u = arma::chol(Prec_u);
      arma::mat L_lower_u = L_upper_u.t();
      arma::vec rhs_solved = arma::solve(arma::trimatl(L_lower_u), rhs);
      u = arma::solve(arma::trimatu(L_upper_u), arma::randn(I) + rhs_solved);
    }

    // 5. Update mu_E, Sigma_E
    if (!use_fixed_E) {
      // Update sigma2_E (Inv-Gamma)
      double a_tilde = a_Sigma_E + (K + 1.0) / 2.0;
      for (int l = 0; l < p2; l++) {
        double b_tilde = b_Sigma_E + pow(mu_E(l) - m_E(l), 2.0) / 2.0;
        for (int j = 0; j < K; j++) {
          b_tilde += pow(beta_star(j, l) - mu_E(l), 2.0) / 2.0;
        }
        sigma2_E(l) =
            1.0 / arma::randg(arma::distr_param(a_tilde, 1.0 / b_tilde));
      }

      // Update mu_E
      arma::vec sum_beta_star = arma::sum(beta_star, 0).t();
      arma::vec mean_mu_E = (m_E + sum_beta_star) / (K + 1.0);
      arma::mat cov_mu_E = arma::diagmat(sigma2_E) / (K + 1.0);
      arma::mat L_mu_E = arma::chol(cov_mu_E, "lower");
      mu_E = mean_mu_E + L_mu_E * arma::randn(p2);
    }

    // 6. Update sigma2
    double a_sig_tilde = a_sigma2 + (I * T) / 2.0;
    double b_sig_tilde = b_sigma2;
    for (int t = 0; t < T; t++) {
      arma::mat X_O_t = X_O.slice(t);
      arma::mat X_E_t = X_E.slice(t);
      arma::vec y_col_t = y.col(t);

      for (int i = 0; i < I; i++) {
        double res = y_col_t(i) - arma::dot(X_O_t.row(i), beta_O) -
                     arma::dot(X_E_t.row(i), beta_star.row(s(i))) - u(i);
        b_sig_tilde += pow(res, 2.0) / 2.0;
      }
    }
    sigma2 =
        1.0 / arma::randg(arma::distr_param(a_sig_tilde, 1.0 / b_sig_tilde));

    // 7. Update tau2
    if (use_spatial) {
      double a_tau_tilde = a_tau2 + I / 2.0;
      double b_tau_tilde = b_tau2 + 0.5 * arma::as_scalar(u.t() * Q_xi * u);
      tau2 =
          1.0 / arma::randg(arma::distr_param(a_tau_tilde, 1.0 / b_tau_tilde));
    }

    // 8. Update xi
    if (use_spatial && random_xi) {
      double xi_now = xi;
      double xi_new = log(xi_now / (1.0 - xi_now)) + sqrt(s_xi) * arma::randn();
      xi_new = 1.0 / (1.0 + exp(-xi_new));

      arma::sp_mat Q_xi_new = xi_new * (sumW - W) + (1.0 - xi_new) * diagI;
      double log_det_Q_now = current_log_det_Q;
      double log_det_Q_new;
      double sign;
      arma::log_det(log_det_Q_new, sign, arma::mat(Q_xi_new));

      double log_accept = alpha_xi * (log(xi_new) - log(xi_now)) +
                          beta_xi * (log(1.0 - xi_new) - log(1.0 - xi_now));
      log_accept += 0.5 * (log_det_Q_new - log_det_Q_now) -
                    0.5 / tau2 * arma::as_scalar(u.t() * (Q_xi_new - Q_xi) * u);

      double accept_prob =
          std::isnan(log_accept) ? 0.0 : std::min(1.0, exp(log_accept));

      xi_accept += accept_prob;
      xi_count++;

      if (R::runif(0, 1) < accept_prob) {
        xi = xi_new;
        Q_xi = Q_xi_new;
        current_log_det_Q = log_det_Q_new;
      }

      double adapt_val = pow(g + 1.0, -0.7) * (accept_prob - 0.234);
      double log_s_xi = log(s_xi) + adapt_val;
      if (log_s_xi > 50.0)
        log_s_xi = 50.0;
      if (log_s_xi < -50.0)
        log_s_xi = -50.0;
      s_xi = exp(log_s_xi);
    }

    // Save samples
    if (g >= burn_in && (g - burn_in) % thin == 0) {
      double log_lik = 0.0;
      for (int i = 0; i < I; i++) {
        double ll_i = 0.0;
        for (int t = 0; t < T; t++) {
          arma::mat X_O_t = X_O.slice(t);
          arma::mat X_E_t = X_E.slice(t);
          arma::vec y_col_t = y.col(t);

          double mean_it = arma::dot(X_O_t.row(i), beta_O) +
                           arma::dot(X_E_t.row(i), beta_star.row(s(i))) + u(i);
          ll_i += R::dnorm(y_col_t(i), mean_it, sqrt(sigma2), 1);
        }
        log_lik += ll_i;
        log_lik_ind_out(iter, i) = ll_i;
      }
      log_lik_out(iter) = log_lik;

      double log_prior = 0.0;
      if (p1 > 0) {
        arma::vec diff_O = beta_O - mu_O;
        double val_O, sign_O;
        arma::log_det(val_O, sign_O, Sigma_O_inv);
        log_prior += -0.5 * p1 * log(2.0 * M_PI) + 0.5 * val_O -
                     0.5 * arma::as_scalar(diff_O.t() * Sigma_O_inv * diff_O);
      }

      if (use_fixed_E && p2 > 0) {
        double val_E, sign_E;
        arma::log_det(val_E, sign_E, Sigma_E_inv_fixed);
        for (int j = 0; j < K; j++) {
          arma::vec diff_E = beta_star.row(j).t() - mu_E;
          log_prior +=
              -0.5 * p2 * log(2.0 * M_PI) + 0.5 * val_E -
              0.5 * arma::as_scalar(diff_E.t() * Sigma_E_inv_fixed * diff_E);
        }
      } else if (!use_fixed_E) {
        for (int j = 0; j < K; j++) {
          for (int l = 0; l < p2; l++) {
            log_prior +=
                R::dnorm(beta_star(j, l), mu_E(l), sqrt(sigma2_E(l)), 1);
          }
        }
        for (int l = 0; l < p2; l++) {
          log_prior += R::dnorm(mu_E(l), m_E(l), sqrt(sigma2_E(l)), 1);
          log_prior += a_Sigma_E * log(b_Sigma_E) - R::lgammafn(a_Sigma_E) -
                       (a_Sigma_E + 1.0) * log(sigma2_E(l)) -
                       b_Sigma_E / sigma2_E(l);
        }
      }

      log_prior += a_sigma2 * log(b_sigma2) - R::lgammafn(a_sigma2) -
                   (a_sigma2 + 1.0) * log(sigma2) - b_sigma2 / sigma2;

      if (use_spatial) {
        log_prior += a_tau2 * log(b_tau2) - R::lgammafn(a_tau2) -
                     (a_tau2 + 1.0) * log(tau2) - b_tau2 / tau2;
        log_prior += -0.5 * I * log(2.0 * M_PI) + 0.5 * current_log_det_Q -
                     (I / 2.0) * log(tau2) -
                     0.5 / tau2 * arma::as_scalar(u.t() * Q_xi * u);

        if (random_xi) {
          log_prior += -R::lbeta(alpha_xi, beta_xi) +
                       (alpha_xi - 1.0) * log(xi) +
                       (beta_xi - 1.0) * log(1.0 - xi);
        }
      }

      log_post_out(iter) = log_lik + log_prior;

      beta_O_out.row(iter) = beta_O.t();
      beta_star_out[iter] = beta_star;
      u_out.row(iter) = u.t();
      sigma2_out(iter) = sigma2;
      tau2_out(iter) = tau2;
      xi_out(iter) = xi;
      mu_E_out.row(iter) = mu_E.t();
      sigma2_E_out.row(iter) = sigma2_E.t();

      iter++;
    }

    // Print progress
    if ((g + 1) % 1000 == 0) {
      Rcpp::Rcout << "Iteration " << g + 1 << " / " << n_tot << std::endl;
    }
  }

  return List::create(
      Named("beta_O") = beta_O_out, Named("beta_star") = beta_star_out,
      Named("u") = u_out, Named("sigma2") = sigma2_out,
      Named("tau2") = tau2_out, Named("xi") = xi_out, Named("mu_E") = mu_E_out,
      Named("sigma2_E") = sigma2_E_out, Named("log_lik") = log_lik_out,
      Named("log_lik_ind") = log_lik_ind_out, Named("log_post") = log_post_out);
}
