import random
import numpy as np
from sklearn.preprocessing import normalize
import copy
from collections import OrderedDict

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Retrieve the adjacency matrix, update it and calculate cell scores
    adjacent_score = copy.deepcopy(the_adjacency_matrix)

    adjacent_score = copy.deepcopy(the_adjacency_matrix)
    adjacent_score[[pos for pos,val in enumerate(game._board_state[0:-3]) if val>0],:]=0
    adjacent_score = np.dot(np.dot(np.dot(adjacent_score,adjacent_score),adjacent_score),adjacent_score)
    score_vector = np.sum(adjacent_score,axis=1)
    
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
  
    own_score = sum([score_vector[x+y*game.height] for x,y in game.get_legal_moves(player)])
    opp_score = sum([score_vector[x+y*game.height] for x,y in game.get_legal_moves(game.get_opponent(player))])
    return float(own_score - opp_score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Retrieve the adjacency matrix, update it and calculate cell scores
    adjacent_score = copy.deepcopy(the_adjacency_matrix)
    used_positions_list = [pos for pos,val in enumerate(game._board_state[0:-3]) if val>0]
    adjacent_score[used_positions_list,:]=0
    adjacent_score[:,used_positions_list]=0
    adjacent_score = np.dot(np.dot(np.dot(adjacent_score,adjacent_score),adjacent_score),adjacent_score)
    score_vector = normalize(np.sum(adjacent_score,axis=1)[:,None],norm='max',axis=0)
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
  
    own_score = sum([score_vector[x+y*game.height] for x,y in game.get_legal_moves(player)])
    opp_score = sum([score_vector[x+y*game.height] for x,y in game.get_legal_moves(game.get_opponent(player))])
    return float(own_score - opp_score)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Retrieve the adjacency matrix, update it and calculate cell scores
    adjacent_score = copy.deepcopy(the_adjacency_matrix)
    used_positions_list = [pos for pos,val in enumerate(game._board_state[0:-3]) if val>0]
    adjacent_score[used_positions_list,:]=0
    adjacent_score[:,used_positions_list]=0
    adjacent_score = np.dot(np.dot(np.dot(adjacent_score,adjacent_score),adjacent_score),adjacent_score)
    score_vector = np.sum(adjacent_score>0,axis=1)
    
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
  
    own_score = sum([score_vector[x+y*game.height] for x,y in game.get_legal_moves(player)])
    opp_score = sum([score_vector[x+y*game.height] for x,y in game.get_legal_moves(game.get_opponent(player))])
    return float(own_score - opp_score)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def min_value(self, game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if depth <= 0:
                #return min([self.score(game.forecast_move(m), self) for m in legal_moves])
                return self.score(game,self)
            else:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    return float('-inf')
                
                value_m = float('inf')
                for m in legal_moves:
                    value_m = min(value_m, max_value(self, game.forecast_move(m), depth-1))
                return value_m
            
        def max_value(self, game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if depth <= 0:
                #return max([self.score(game.forecast_move(m), self) for m in legal_moves])
                return self.score(game, self)
            else:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    return float('-inf')

                value_m = float('-inf')
                for m in legal_moves:
                    value_m = max(value_m, min_value(self, game.forecast_move(m), depth-1))
                return value_m
        
        legal_moves_original = game.get_legal_moves()
        if legal_moves_original:
            return max(legal_moves_original, key=lambda m: min_value(self, game.forecast_move(m), depth-1))
        else:
            return (-1,-1)




class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
 
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        
        # Create an adjacency matrix for the graph to use for scoring
        global the_adjacency_matrix
        try:
            the_adjacency_matrix
        except:
            directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            ad=np.zeros((game.width*game.height,game.width*game.height), float)
            for idx in [((x+game.height*y),(x+dx+(game.height*(y+dy)))) for x in range(game.height) for y in range(game.width) for dx,dy in directions if 0 <= x+dx < game.height and 0 <= y+dy < game.width]:
                ad[idx]=1
            the_adjacency_matrix = ad
         
        depth = 1
        while depth>=0:
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move= self.alphabeta(game, depth)
                depth += 1

            except SearchTimeout:
                return best_move
                depth=-1
        # Return the best move from the last completed search iteration
        return best_move

    


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def max_value(self, game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth <= 0:
                return self.score(game, self)
            else:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    return float('-inf')

                value_m = float('-inf')
                for m in legal_moves:
                    value_m = max(value_m, min_value(self, game.forecast_move(m), depth-1, alpha, beta))
                    if value_m >= beta:
                        return value_m
                    alpha = max(alpha, value_m)
                return value_m
         
         
        def min_value(self, game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth <= 0:
                return self.score(game,self)
            else:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    return float('inf')
                
                value_m = float('inf')
                for m in legal_moves:
                    value_m = min(value_m, max_value(self, game.forecast_move(m), depth-1, alpha, beta))
                    if value_m <= alpha:
                        return value_m
                    beta=min(beta,value_m)
                return value_m
        
        legal_moves_original = game.get_legal_moves()
        if legal_moves_original:
            bestmove = legal_moves_original[0]
            value_m = float('-inf')
            for m in legal_moves_original:
                nvm = min_value(self, game.forecast_move(m), depth-1, alpha, beta)
                if nvm > value_m:
                    value_m = nvm
                    bestmove = m
                    if value_m > beta:
                        return m
                alpha = max(alpha, value_m)
            return bestmove
        else:
            return (-1,-1)
