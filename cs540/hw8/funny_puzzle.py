import heapq
import copy

# solve method: ChatGPT, Prompt: I entered the description for the solve method, and compared it with my method (which contained errors)
# heuristic method: ChatGPT, Prompt: I asked how to get the manhattan distance in a 3x3 grid.
# solvable method: ChatGPT, Prompt: "How would you check if an 8-puzzle is solvable?" I then used those ideas in my implementation
# get_succ method: ChatGPT, Prompt: I typed in the description of the method, and then used the ideas in my method.


# return the manhattan distance heuristic
def get_manhattan_distance(from_state, to_state):
    manhattan = 0
    state = from_state
    goal_state = to_state
    for i in range(len(state)):
        number = state[i]
        
        # if the tile is empty we do not count it
        if number != 0:
            goal_i = goal_state.index(number)
            curr_x = i // 3
            curr_y = i % 3
            goal_x = goal_i // 3
            goal_y = goal_i % 3
            manhattan += abs(goal_x - curr_x) + abs(goal_y - curr_y)
    return manhattan

# check the format of state, and return corresponding goal state.
def state_check(state):
    """check the format of state, and return corresponding goal state.
    Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')     
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                         'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

# given a state of the puzzle, represented as a single list of integers with a 0 in the empty spaces, print 
# to the console all of the possible successor states.
def print_succ(state):
    goal_state = state_check(state)
    succ_list = []
    zero_indices = []
    
    # we need to get the indices of the zero positions so we can know what tiles we can move
    for i in range(len(state)):
        if state[i] == 0:
            zero_indices.append(i)
    
    # now, go through each index that has nothing in it and get the successors
    for j in zero_indices:
        # doing the index divided by 3 with the floor function gives us the row
        x = j // 3
        
        # doing the index mod with 3 gives us the column
        y = j % 3
        
        # we can move left, right, down, and up
        next_moves = []
        next_moves.append([x - 1, y])
        next_moves.append([x + 1, y])
        next_moves.append([x, y - 1])
        next_moves.append([x, y + 1])
        
        # now we need to go through each of those moves and see if they are valid and get their successor
        for move in next_moves:
            curr_x = move[0]
            curr_y = move[1]
            
            # the next x and y need to be between zero and two
            if (curr_x >= 0 and curr_x <= 2 and curr_y >= 0 and curr_y <= 2):
                next_position = curr_x * 3 + curr_y
                
                # create a new state and update it based on the new move
                next_state = copy.copy(state)
                
                # we don't want to try to swap 0 and 0, that will give us extra successors we don't want
                if not (next_state[j] == next_state[next_position] == 0):
                    save = next_state[j]
                    next_state[j] = next_state[next_position]
                    next_state[next_position] = save
                    succ_list.append(next_state)
    succ_list.sort()
    
    # now to print out the successors and the manhattan distance
    for s in succ_list:
        h = get_manhattan_distance(s, goal_state)
        print(f"{s} h={h}")

# get the successors but do not print them
def get_succ(state):
    goal_state = state_check(state)
    succ_list = []
    zero_indices = []
    
    # we need to get the indices of the zero positions so we can know what tiles we can move
    for i in range(len(state)):
        if state[i] == 0:
            zero_indices.append(i)
    
    # now, go through each index that has nothing in it and get the successors
    for j in zero_indices:
        # doing the index divided by 3 with the floor function gives us the row
        x = j // 3
        
        # doing the index mod with 3 gives us the column
        y = j % 3
        
        # we can move left, right, down, and up
        next_moves = []
        next_moves.append([x - 1, y])
        next_moves.append([x + 1, y])
        next_moves.append([x, y - 1])
        next_moves.append([x, y + 1])
        
        # now we need to go through each of those moves and see if they are valid and get their successor
        for move in next_moves:
            curr_x = move[0]
            curr_y = move[1]
            
            # the next x and y need to be between zero and two
            if (curr_x >= 0 and curr_x <= 2 and curr_y >= 0 and curr_y <= 2):
                next_position = curr_x * 3 + curr_y
                
                # create a new state and update it based on the new move
                next_state = copy.copy(state)
                
                # we don't want to try to swap 0 and 0, that will give us extra successors we don't want
                if not (next_state[j] == next_state[next_position] == 0):
                    save = next_state[j]
                    next_state[j] = next_state[next_position]
                    next_state[next_position] = save
                    succ_list.append(next_state)
    succ_list.sort()
    return succ_list

# see if a state is solvable or not. It is solvable when the number of inversions is even, or if there are multiple
# zeros
def solvable_condition(state):
    # get a list of the non-zero numbers
    nums_not_zero = []
    for num in state:
        if num != 0:
            nums_not_zero.append(num)
    
    # if there were multiple zeros, then it is automatically solvable.
    if len(nums_not_zero) <= 7:
        return True
    
    num_inversions = 0
    for i in range(len(nums_not_zero)):
        for j in range(i + 1, len(nums_not_zero)):
            if nums_not_zero[i] > nums_not_zero[j]:
                num_inversions = num_inversions + 1
    if num_inversions % 2 == 0:
        return True
    else:
        return False
    
# given a state of the puzzle, perform the A* search algorithm and print the path from the current state 
# to the goal state.
def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    # first, check if the state is solvable
    if solvable_condition(state) == False:
        print(False)
        return
    
    # if we get here, it's solvable
    goal_state = list(state_check(state))
    pq = []
    visited = {}
    g = 0
    h = get_manhattan_distance(state, goal_state)
    
    # push initial state to the pq
    heapq.heappush(pq, (g + h, state, (g, h, -1)))
    
    while pq:
        # pop the first element off
        popped = heapq.heappop(pq)
        curr_state = popped[1]
        g = popped[2][0]
        h = popped[2][1]
        parent_index = popped[2][2]
        
        # if the popped state is the goal state, we're done
        if curr_state == goal_state:
            # get the path from start to goal
            path = []
            print(True)
            
            # it won't work if we try to do this same code with the initial state, so we just add that last
            while parent_index != -1:
                append_this = (curr_state, g)
                path.append(append_this)
                
                # update current item
                curr_state = tuple(parent_index)
                (g, h, parent_index) = visited[curr_state]
            
            # we also need the parent node in there. Default g is zero
            path.append((state, 0))
            
            # it's backwards, so reverse it
            path = path[::-1]
            
            # printing
            for state, g in path:
                print(f"{list(state)} h={get_manhattan_distance(state, goal_state)} moves: {g}")
            return
        
        # if this state has not already been visited, we need to visit it and update successors
        if tuple(curr_state) not in visited:
            visited[tuple(curr_state)] = (g, h, parent_index)
            # if the popped state was not the goal state, we need to expand the successors
            for successor in get_succ(curr_state):
                if tuple(successor) not in visited:
                    # recalculate the heuristic
                    h1 = get_manhattan_distance(successor, goal_state)
                    
                    # the parent of each successor is the current state. Add one to each g.
                    heapq.heappush(pq, (g + 1 + h1, successor, (g + 1, h1, curr_state)))