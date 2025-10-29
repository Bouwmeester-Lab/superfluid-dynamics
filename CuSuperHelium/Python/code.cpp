// rho_new = dot(r0, r)
    cublasDdot(handle, N, r0, 1, r, 1, &rho_new);

    if (rho_new == 0.0) break; // breakdown

    beta = (rho_new / rho_old) * (alpha / omega);

    // p = r + beta * (p - omega * v)
    // p = p - omega * v
    const double minus_omega = -omega;
    cublasDaxpy(handle, N, &minus_omega, v, 1, p, 1);
    // p = beta * p
    cublasDscal(handle, N, &beta, p, 1);
    // p = p + r
    cublasDaxpy(handle, N, &alpha, r, 1, p, 1);

    // v = A * p
    A.apply(p, v, stream);

    // alpha = rho_new / dot(r0, v)
    double temp_dot;
    cublasDdot(handle, N, r0, 1, v, 1, &temp_dot);
    alpha = rho_new / temp_dot;

    // s = r - alpha * v
    cublasDcopy(handle, N, r, 1, s, 1);
    double neg_alpha = -alpha;
    cublasDaxpy(handle, N, &neg_alpha, v, 1, s, 1);

    // Check ||s|| < tolerance
    cublasDnrm2(handle, N, s, 1, &resid);
    if (resid < tolerance) {
        // x += alpha * p
        cublasDaxpy(handle, N, &alpha, p, 1, x, 1);
        break;
    }

    // t = A * s
    A.apply(s, t, stream);

    // omega = dot(t, s) / dot(t, t)
    double ts, tt;
    cublasDdot(handle, N, t, 1, s, 1, &ts);
    cublasDdot(handle, N, t, 1, t, 1, &tt);
    omega = ts / tt;

    // x += alpha * p + omega * s
    cublasDaxpy(handle, N, &alpha, p, 1, x, 1);
    cublasDaxpy(handle, N, &omega, s, 1, x, 1);

    // r = s - omega * t
    cublasDcopy(handle, N, s, 1, r, 1);
    double neg_omega = -omega;
    cublasDaxpy(handle, N, &neg_omega, t, 1, r, 1);

    // Check ||r|| < tolerance
    cublasDnrm2(handle, N, r, 1, &resid);
    if (resid < tolerance) {
        break;
    }

    rho_old = rho_new;