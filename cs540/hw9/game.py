import random
import copy
import math

# Minimax Algorithm (make_move): model: ChatGPT: I copied the algorithm from class, but then some small things were wrong so I typed the prompt into chatGPT and it told me what to fix. It also helped me with the minimax_decision function, I asked it to write that after I was done writing the min_value and max_value functions
# heuristic_game_value: model: ChatGPT: I gave it my current heuristic (that considered only the manhattan distance to the center) and then asked ChatGPT to help me make it more complex since I wasn't passing all of the tests on gradescope.
# make_move end part: model: ChatGPT: I gave ChatGPT my current code and asked it to help me make a move based on my heuristic and minimax algorithm.


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def run_challenge_test(self):
        # Set to True if you would like to run gradescope against the challenge AI!
        # Leave as False if you would like to run the gradescope tests faster for debugging.
        # You can still get full credit with this set to False
        return True

    # a heuristic to evaluate non-terminal states
    def heuristic_game_value(self, state):
        total_distance = 0
        c = 0
        pieces = []
        
        # find all my pieces
        for row in range(5):
            for col in range(5):
                if state[row][col] == self.my_piece:
                    # Compute Euclidian distance to the center. (2, 2) is the center
                    total_distance += math.sqrt((row - 2) ** 2 + (col - 2) ** 2)
                    pieces.append((row, col))
                    
        # Calculate clustering: sum of Euclidian distances between each pair of pieces
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                r1, c1 = pieces[i]
                r2, c2 = pieces[j]
                c += math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
                
        # Normalize the proximity score (distance to the center)
        max_distance_to_center = len(pieces) * math.sqrt(2 * 4)  # max Euclidean distance per piece to center
        if pieces:
            proximity_score = 1 - (total_distance / max_distance_to_center)
        else:
            proximity_score = 0

        # Normalize the clustering score
        if len(pieces) > 1:
            max_clustering_penalty = len(pieces) * (len(pieces) - 1) / 2 * math.sqrt(2 * 4)
        else:
            max_clustering_penalty = 1
        if max_clustering_penalty > 0:
            clustering_score = 1 - (c / max_clustering_penalty)
        else:
            clustering_score = 0

        # Combine the two scores: proximity is weighted higher
        scaled_score = 2 * (0.7 * proximity_score + 0.3 * clustering_score) - 1

        return scaled_score
            
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        # drop phase means that we are just placing pieces, not relocating pieces (start of game)
        drop_phase = True   # TODO: detect drop phase
        count_pieces = 0
        for row in range(5):
            for col in range(5):
                # if it's not blank, we have a piece there
                if state[row][col] != ' ':
                    count_pieces += 1
        if count_pieces >= 8:
            drop_phase = False

        # TODO: implement a minimax algorithm to play better
        max_depth = 3
        
        # minimax implementing
        def max_value(state, depth, alpha, beta):
            if self.game_value(state) != 0:
                return self.game_value(state)
            if depth == 0:
                return self.heuristic_game_value(state)
            value = -float('inf')
            for s in self.succ(state):
                value = max(value, min_value(s, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
    
        def min_value(state, depth, alpha, beta):
            if self.game_value(state) != 0:
                return self.game_value(state)
            if depth == 0:
                return self.heuristic_game_value(state)
            value = float('inf')
            for s in self.succ(state):
                value = min(value, max_value(s, depth - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:  # Alpha cutoff
                    break
            return value
        
        def minimax_decision(state, max_depth):
            best_move = None
            best_value = -float('inf')  # Maximizing player
            alpha = -float('inf')
            beta = float('inf')
            
            for succ_state in self.succ(state):
                value = min_value(succ_state, max_depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_move = succ_state
                alpha = max(alpha, best_value)
            return best_move

        # Get the best successor state
        best_state = minimax_decision(state, max_depth)
        
        move = []
        
        if drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ' and best_state[row][col] == self.my_piece:
                        move =  [(row, col)]
        else:
            source = None
            destination = None
            for row in range(5):
                for col in range(5):
                    if state[row][col] == self.my_piece and best_state[row][col] == ' ':
                        source = (row, col)
                    if state[row][col] == ' ' and best_state[row][col] == self.my_piece:
                        destination = (row, col)
            if source and destination:
                move = [destination, source]
        
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins - there are 4 options
        if state[1][0] != ' ' and state[1][0] == state[2][1] == state[3][2] == state[4][3]:
            return 1 if state[1][0] == self.my_piece else -1
        if state[0][0] != ' ' and state[0][0] == state[1][1] == state[2][2] == state[3][3]:
            return 1 if state[0][0] == self.my_piece else -1
        if state[1][1] != ' ' and state[1][1] == state[2][2] == state[3][3] == state[4][4]:
            return 1 if state[1][1] == self.my_piece else -1
        if state[0][1] != ' ' and state[0][1] == state[1][2] == state[2][3] == state[3][4]:
            return 1 if state[0][1] == self.my_piece else -1
        
        # TODO: check / diagonal wins - there are 4 options
        if state[3][0] != ' ' and state[3][0] == state[2][1] == state[1][2] == state[0][3]:
            return 1 if state[3][0] == self.my_piece else -1
        if state[4][0] != ' ' and state[4][0] == state[3][1] == state[2][2] == state[1][3]:
            return 1 if state[4][0] == self.my_piece else -1
        if state[3][1] != ' ' and state[3][1] == state[2][2] == state[1][3] == state[0][4]:
            return 1 if state[3][1] == self.my_piece else -1
        if state[4][1] != ' ' and state[4][1] == state[3][2] == state[2][3] == state[1][4]:
            return 1 if state[4][1] == self.my_piece else -1
        
        
        # TODO: check box wins
        for r in range(4):
            for c in range(4):
                if state[r][c] != ' ' and state[r][c] == state[r + 1][c] == state[r][c + 1] == state[r + 1][c + 1]:
                    return 1 if state[r][c] == self.my_piece else -1

        return 0 # no winner yet

    def succ(self, state):
        """state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character)."""
        
        successors = []
        
        # first check if we are in a drop phase
        count_pieces = 0
        for row in range(5):
            for col in range(5):
                # if it's not blank, we have a piece there
                if state[row][col] != ' ':
                    count_pieces += 1
        
        # now, we are in a drop phase if count_pieces < 8
        if count_pieces < 8:
            for row in range(5):
                for col in range(5):
                    # if it's blank, we can place there
                    if state[row][col] == ' ':
                        new_succ = copy.deepcopy(state)
                        new_succ[row][col] = self.my_piece # place the current player's color here
                        successors.append(new_succ)
        else:
            # not a drop phase. This is more complicated, we need to go through every one of our pieces
            for row in range(5):
                for col in range(5):
                    # if our piece is here, we can work with it and move it.
                    if state[row][col] == self.my_piece:
                        # try moving left
                        if (col - 1) >= 0 and state[row][col - 1] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row][col - 1] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
                        # try moving up left
                        if (col - 1) >= 0 and (row - 1) >= 0 and state[row - 1][col - 1] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row - 1][col - 1] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
                        # try moving up
                        if (row - 1) >= 0 and state[row - 1][col] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row - 1][col] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
                        
                        # try moving up right
                        if (row - 1) >= 0 and (col + 1) <= 4 and state[row - 1][col + 1] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row - 1][col + 1] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
                        
                        # try moving right
                        if (col + 1) <= 4 and state[row][col + 1] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row][col + 1] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
                        
                        # try moving down right
                        if (col + 1) <= 4 and (row + 1) <= 4 and state[row + 1][col + 1] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row + 1][col + 1] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
                        
                        # try moving down
                        if (row + 1) <= 4 and state[row + 1][col] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row + 1][col] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
                        
                        # try moving down left
                        if (row + 1) <= 4 and (col - 1) >= 0 and state[row + 1][col - 1] == ' ':
                            new_succ = copy.deepcopy(state)
                            new_succ[row + 1][col - 1] = self.my_piece
                            new_succ[row][col] = ' '
                            successors.append(new_succ)
        
        return successors
    
############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
