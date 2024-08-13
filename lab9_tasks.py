def min_max_cached(game, player, depth=0):
    """
    :param game: an instance of MathTicTacToe
    :param player: whether maximizing player(1) or minimizing player(-1)
    :param depth: current search depth
    :return: best_value, best_move
    """

    # cache_key is a tuple of the board status and player
    cache_key = (tuple(game.board.flatten().tolist()), player)

    # if the board status has been calculated before, return the score from the cache
    if cache_key in game.score_cache:
        game.hit += 1
        return game.score_cache[cache_key]

    # boundary case check: we use decayed score for the leaf node
    if game.check_win():
        previous_player = -player
        score = 100 if previous_player == 1 else -100
        decay_factor = 0.9
        score *= decay_factor ** (depth - 1)

        game.score_cache[cache_key] = (score, None)
        return score, None

    if game.is_full:
        score = 0
        game.score_cache[cache_key] = (score, None)
        return score, None  # This is a draw

    best_value = float('-inf') * player  # inf for player -1, -inf for player 1
    best_move = None

    # scan all cases by min-max alpha-beta pruning
    if player == 1:
        # Maximizing player
        ######################## TODO 1.1 ############################
        # In this task, you search all possible moves and scores, and find the move with the best score in maximizer's view.
        # possible usage: game.get_available_moves(), game.make_move(), game.undo_move()
        # target variables to get: best_value, best_move
        # hint: you may use min_max_cached function recursively
        for move in game.get_available_moves():
            game.make_move(move)
            value, _ = min_max_cached(game, -player, depth + 1)
            game.undo_move(move)

            if value > best_value:
                best_value = value
                best_move = move

        ######################## END of TODO 1.1 ############################

        game.score_cache[cache_key] = (best_value, best_move)# cache record
        return best_value, best_move
    else:
        # Minimizing player
        ######################## TODO 1.2 ############################
        # In this task, you search all possible moves and scores, and find the move with the best score in minimizer's view.
        # possible usage: game.get_available_moves(), game.make_move(), game.undo_move()
        # target variables: best_value, best_move
        # hint: you may use min_max_cached function recursively
        for move in game.get_available_moves():
            game.make_move(move)
            value, _ = min_max_cached(game, -player, depth + 1)
            game.undo_move(move)

            if value < best_value:
                best_value = value
                best_move = move

        ######################## END of TODO 1.2 ############################

        game.score_cache[cache_key] = (best_value, best_move) # cache record
        return best_value, best_move



def min_max_alpha_beta_cached(game, player, alpha=float('-inf'), beta=float('inf'), depth=0):

    # cache_key is a tuple of the board status and player
    cache_key = (tuple(game.board.flatten().tolist()), player)

    if cache_key in game.score_cache:
        game.hit += 1
        return game.score_cache[cache_key]

    # boundary case check: we use decayed score for the leaf node
    if game.check_win():
        previous_player = -player
        score = 100 if previous_player == 1 else -100
        decay_factor = 0.9
        score *= decay_factor ** (depth - 1)

        game.score_cache[cache_key] = (score, None)
        return score, None

    if game.is_full:
        score = 0
        game.score_cache[cache_key] = (score, None)
        return score, None  # This is a draw

    best_value = float('-inf') * player  # inf for player -1, -inf for player 1
    best_move = None
    alpha_beta_break = False

    # scan all cases by min-max alpha-beta pruning
    if player == 1:

        # Maximizing player
        ######################## TODO 2.1 ############################
        # In this task, you search all possible moves and scores, and find the move with the best score in maximizer's view with alpha-beta pruning.
        # possible usage: game.get_available_moves(), game.make_move(), game.undo_move()
        # target variables to get: best_value, best_move, alpha_beta_break
        # hint: you may use min_max_alpha_beta_cached function recursively;
        # hint: you need to change alpha_beta_break flag when alpha-beta pruning is triggered.
        for move in game.get_available_moves():
            game.make_move(move)
            value, _ = min_max_alpha_beta_cached(game, -player, alpha, beta, depth + 1)
            game.undo_move(move)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if alpha >= beta:
                alpha_beta_break = True
                break

        ######################## END of TODO 2.1 ############################

        if not alpha_beta_break:
            game.score_cache[cache_key] = (best_value, best_move)

        return best_value, best_move

    else:
        # Minimizing player

        ######################## TODO 2.2 ############################
       # In this task, you search all possible moves and scores, and find the move with the best score in minimizer's view with alpha-beta pruning.
       # possible usage: game.get_available_moves(), game.make_move(), game.undo_move()
        # target variables to get: best_value, best_move, alpha_beta_break
        # hint: you may use min_max_alpha_beta_cached function recursively; you need to change alpha_beta_break flag when alpha-beta pruning is triggered.
        for move in game.get_available_moves():
            game.make_move(move)
            value, _ = min_max_alpha_beta_cached(game, -player, alpha, beta, depth + 1)
            game.undo_move(move)

            if value < best_value:
                best_value = value
                best_move = move

            beta = min(beta, best_value)
            if alpha >= beta:
                alpha_beta_break = True
                break

        ######################## END of TODO 2.2 ############################
        if not alpha_beta_break:
            game.score_cache[cache_key] = (best_value, best_move)
        return best_value, best_move