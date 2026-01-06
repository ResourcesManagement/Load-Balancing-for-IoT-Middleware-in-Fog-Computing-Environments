def GOA(initsol, fname, xmin, xmax, Max_iter):
    FT = 0.5  # Threshold for exploring a random direction
    RT = 0.3  # Threshold for moving towards ripe position
    epsilon = 0.1  # Step size (perturbation rate)
    # Initialize flowers and plums
    N, D = positions.shape[0], positions.shape[1]
    plums = np.copy(positions)  # Initialize plums to the positions of the flowers
    pgbest = positions[np.argmin(obj_fitness(positions))]  # Initialize pgbest
    Convergence_curve = np.zeros(maxiter)
    ct = time.time()
    # Main loop
    for iteration in range(maxiter):
        # Update each flower and plum
        for i in range(N):
            r = np.random.rand()
            if r >= FT:
                # Update the flower using formula (3)
                positions[i] = update_flower(positions[i], r, FT, RT, epsilon, X_min, X_max)
            elif r >= RT:
                # Update the flower using formula (4)
                positions[i] = update_flower(positions[i], r, FT, RT, epsilon, X_min, X_max)
            else:
                # Update the flower using formulas (5)-(6)
                positions[i] = update_flower(positions[i], r, FT, RT, epsilon, X_min, X_max)

            # Ensure flowers remain within the range [X_min, X_max]
            positions[i] = np.clip(positions[i], X_min, X_max)

            # Update the plum using formula (7)
            plums[i] = update_plum(plums[i], positions, alpha=0.5)

        # Update the global best position (pgbest)
        current_best = positions[np.argmin(obj_fitness(positions))]
        if obj_fitness(current_best.reshape(1, -1)) < obj_fitness(pgbest.reshape(1, -1)):
            pgbest = current_best
        Convergence_curve[iteration] = current_best
    ct = time.time() - ct
    return pgbest, Convergence_curve, plums, ct


