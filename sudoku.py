"""
Sudoku Solver using Constraint Satisfaction Problem (CSP) Formulation

This module defines a SudokuCSP class that models a Sudoku puzzle as a CSP,
and provides both a basic backtracking solver and a smart backtracking solver
that incorporates heuristics like Minimum Remaining Values (MRV), Degree Heuristic,
Least Constraining Value (LCV), and Forward Checking (FC).

The module also includes functions to solve a Sudoku puzzle, print the board,
and run a main function that tests four puzzles (Easy, Medium, Hard, Evil) using
both solving strategies. 
"""

import time
import random
import copy
from collections import defaultdict

class SudokuCSP:
    """
    A class to represent a Sudoku puzzle as a Constraint Satisfaction Problem (CSP).
    
    Attributes:
        board (list of list of int): 9x9 Sudoku board where 0 represents empty cells.
        size (int): The number of rows/columns (9).
        box_size (int): The size of each 3x3 subgrid (3).
        empty_cell (int): The value representing an empty cell (0).
        domains (dict): Dictionary mapping each cell (tuple) to its possible values.
        trial_count (int): Counter for the number of assignment trials.
    """
    
    def __init__(self, board):
        """
        Initialize the SudokuCSP with a given board.
        
        Args:
            board (list of list of int): A 9x9 list representing the Sudoku puzzle,
                where 0 indicates an empty cell.
        """
        self.board = copy.deepcopy(board)
        self.size = 9
        self.box_size = 3
        self.empty_cell = 0  # Value representing empty cells
        
        # Maintain a dictionary of domains for each variable (cell)
        self.domains = {}
        self.initialize_domains()
        
        # Counter for the number of assignment trials
        self.trial_count = 0

    def initialize_domains(self):
        """
        Initialize domains for all variables (cells) in the Sudoku board.
        
        For each empty cell, the domain is set to {1,...,9}. For pre-filled cells,
        the domain is the singleton set containing the given value.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == self.empty_cell:
                    self.domains[(i, j)] = set(range(1, self.size + 1))
                else:
                    self.domains[(i, j)] = {self.board[i][j]}
                    
    def print_board(self):
        """
        Print the current state of the Sudoku board in a formatted layout.
        """
        for i in range(self.size):
            if i % self.box_size == 0 and i > 0:
                print("-" * 21)
            row = ""
            for j in range(self.size):
                if j % self.box_size == 0 and j > 0:
                    row += "| "
                row += str(self.board[i][j]) + " "
            print(row)
    
    def is_complete(self):
        """
        Check if the Sudoku board assignment is complete.
        
        Returns:
            bool: True if all cells are filled (non-zero), False otherwise.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == self.empty_cell:
                    return False
        return True
    
    def is_consistent(self, var, value):
        """
        Check if assigning a given value to a cell (var) is consistent with the current assignment.
        
        The assignment is consistent if the value does not appear in the same row, column, or 3x3 box.
        
        Args:
            var (tuple): The cell coordinates as (row, col).
            value (int): The value to assign.
        
        Returns:
            bool: True if the assignment is consistent, False otherwise.
        """
        row, col = var
        
        # Check row constraint
        for j in range(self.size):
            if self.board[row][j] == value:
                return False
        
        # Check column constraint
        for i in range(self.size):
            if self.board[i][col] == value:
                return False
        
        # Check 3x3 box constraint
        box_row, box_col = (row // self.box_size) * self.box_size, (col // self.box_size) * self.box_size
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if self.board[i][j] == value:
                    return False
        
        return True
    
    def get_unassigned_variable_in_order(self):
        """
        Get the next unassigned cell in a left-to-right, top-to-bottom order.
        
        Returns:
            tuple or None: The cell coordinates (row, col) if an unassigned cell is found;
                           None if the board is complete.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == self.empty_cell:
                    return (i, j)
        return None
    
    def select_unassigned_variable_mrv(self):
        """
        Select an unassigned cell using the Minimum Remaining Values (MRV) heuristic.
        
        In case of a tie, the degree heuristic is used as a tie-breaker.
        
        Returns:
            tuple: The selected cell coordinates (row, col).
        """
        min_remaining_values = float('inf')
        mrv_var = None
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == self.empty_cell:
                    remaining_values = len(self.domains[(i, j)])
                    if remaining_values < min_remaining_values:
                        min_remaining_values = remaining_values
                        mrv_var = (i, j)
                    elif remaining_values == min_remaining_values and self.degree_heuristic((i, j)) > self.degree_heuristic(mrv_var):
                        mrv_var = (i, j)
        
        return mrv_var
    
    def degree_heuristic(self, var):
        """
        Calculate the degree of a cell as the number of constraints (unassigned neighboring cells).
        
        This is used to break ties when using the MRV heuristic.
        
        Args:
            var (tuple): The cell coordinates (row, col).
        
        Returns:
            int: The degree (number of neighboring unassigned cells).
        """
        if var is None:
            return 0
        
        row, col = var
        count = 0
        
        # Count unassigned cells in the same row
        for j in range(self.size):
            if j != col and self.board[row][j] == self.empty_cell:
                count += 1
        
        # Count unassigned cells in the same column
        for i in range(self.size):
            if i != row and self.board[i][col] == self.empty_cell:
                count += 1
        
        # Count unassigned cells in the same 3x3 box
        box_row, box_col = (row // self.box_size) * self.box_size, (col // self.box_size) * self.box_size
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if i != row and j != col and self.board[i][j] == self.empty_cell:
                    count += 1
        
        return count
    
    def order_domain_values(self, var, lcv=False):
        """
        Return the list of possible values for a cell, optionally ordered by the
        Least Constraining Value (LCV) heuristic.
        
        Args:
            var (tuple): The cell coordinates (row, col).
            lcv (bool): If True, sort the values by how few constraints they impose on neighbors.
        
        Returns:
            list: A list of possible values for the cell.
        """
        if not lcv:
            return list(self.domains[var])
        
        # Compute constraining count for each value
        values = list(self.domains[var])
        constraining_count = {}
        row, col = var
        
        for value in values:
            count = 0
            # Temporarily assign the value
            self.board[row][col] = value
            
            # Count constraints on the row
            for j in range(self.size):
                if j != col and self.board[row][j] == self.empty_cell and value in self.domains[(row, j)]:
                    count += 1
            
            # Count constraints on the column
            for i in range(self.size):
                if i != row and self.board[i][col] == self.empty_cell and value in self.domains[(i, col)]:
                    count += 1
            
            # Count constraints on the 3x3 box
            box_row, box_col = (row // self.box_size) * self.box_size, (col // self.box_size) * self.box_size
            for i in range(box_row, box_row + self.box_size):
                for j in range(box_col, box_col + self.box_size):
                    if (i != row or j != col) and self.board[i][j] == self.empty_cell and value in self.domains[(i, j)]:
                        count += 1
            
            # Reset the cell value
            self.board[row][col] = self.empty_cell
            constraining_count[value] = count
        
        # Return values sorted by their constraining effect (least constraining first)
        return sorted(values, key=lambda x: constraining_count[x])
    
    def forward_checking(self, var, value):
        """
        Perform forward checking after assigning a value to a cell.
        
        This method removes the assigned value from the domains of all neighboring
        unassigned cells. If any neighbor's domain becomes empty, forward checking fails.
        
        Args:
            var (tuple): The cell coordinates (row, col) where the value is assigned.
            value (int): The value assigned to the cell.
        
        Returns:
            tuple or bool:
                - (True, reduced_domains) if forward checking succeeds, where reduced_domains
                  is a dict mapping affected cells to the removed value.
                - False if any neighbor's domain becomes empty.
        """
        row, col = var
        affected_vars = []
        
        # Collect neighbors in the same row, column, and box
        for j in range(self.size):
            if j != col and self.board[row][j] == self.empty_cell:
                affected_vars.append((row, j))
        for i in range(self.size):
            if i != row and self.board[i][col] == self.empty_cell:
                affected_vars.append((i, col))
        box_row, box_col = (row // self.box_size) * self.box_size, (col // self.box_size) * self.box_size
        for i in range(box_row, box_row + self.box_size):
            for j in range(box_col, box_col + self.box_size):
                if (i != row or j != col) and self.board[i][j] == self.empty_cell:
                    affected_vars.append((i, j))
        
        reduced_domains = {}
        for affected_var in affected_vars:
            if value in self.domains[affected_var]:
                reduced_domains[affected_var] = value
                self.domains[affected_var].remove(value)
                # If domain becomes empty, forward checking fails
                if len(self.domains[affected_var]) == 0:
                    # Restore the domains that were reduced
                    for var_restore, val in reduced_domains.items():
                        self.domains[var_restore].add(val)
                    return False
        return True, reduced_domains
    
    def restore_domains(self, reduced_domains):
        """
        Restore the domains of cells after backtracking.
        
        Args:
            reduced_domains (dict): Dictionary mapping cells to the value that was removed.
        """
        for var, val in reduced_domains.items():
            self.domains[var].add(val)
    
    def backtrack(self):
        """
        Solve the Sudoku puzzle using basic backtracking.
        
        This method selects the next unassigned cell (in order) and tries each value
        in its domain, incrementing the trial counter for each attempt.
        
        Returns:
            bool: True if the puzzle is solved, False otherwise.
        """
        if self.is_complete():
            return True
        
        var = self.get_unassigned_variable_in_order()
        
        for value in self.domains[var]:
            self.trial_count += 1  # Increment trial counter for each attempt
            if self.is_consistent(var, value):
                self.board[var[0]][var[1]] = value
                
                if self.backtrack():
                    return True
                
                # Undo assignment
                self.board[var[0]][var[1]] = self.empty_cell
        
        return False
    
    def smart_backtrack(self, use_mrv=True, use_lcv=True, use_fc=True):
        """
        Solve the Sudoku puzzle using smart backtracking with heuristics.
        
        The heuristics include:
          - Minimum Remaining Values (MRV) for variable selection.
          - Least Constraining Value (LCV) for ordering domain values.
          - Forward Checking (FC) to prune domains.
        
        Args:
            use_mrv (bool): If True, use the MRV heuristic.
            use_lcv (bool): If True, use the LCV heuristic.
            use_fc (bool): If True, use forward checking.
        
        Returns:
            bool: True if the puzzle is solved, False otherwise.
        """
        if self.is_complete():
            return True
        
        # Select unassigned variable using MRV or simple order
        var = self.select_unassigned_variable_mrv() if use_mrv else self.get_unassigned_variable_in_order()
        
        # Order the domain values, optionally using LCV
        values = self.order_domain_values(var, lcv=use_lcv)
        
        for value in values:
            self.trial_count += 1  # Increment trial counter for each attempt
            if self.is_consistent(var, value):
                self.board[var[0]][var[1]] = value
                
                fc_success = True
                reduced_domains = {}
                if use_fc:
                    fc_result = self.forward_checking(var, value)
                    if isinstance(fc_result, tuple):
                        fc_success, reduced_domains = fc_result
                    else:
                        fc_success = fc_result
                
                if fc_success:
                    if self.smart_backtrack(use_mrv, use_lcv, use_fc):
                        return True
                
                # Undo assignment and restore domains if necessary
                self.board[var[0]][var[1]] = self.empty_cell
                if use_fc:
                    self.restore_domains(reduced_domains)
        
        return False

def solve_sudoku(board, algorithm="smart", use_mrv=True, use_lcv=True, use_fc=True):
    """
    Solve the given Sudoku puzzle using CSP-based backtracking.
    
    Args:
        board (list of list of int): A 9x9 Sudoku board where 0 represents empty cells.
        algorithm (str): Either "simple" for basic backtracking or "smart" for improved backtracking.
        use_mrv (bool): Use the Minimum Remaining Values heuristic if True.
        use_lcv (bool): Use the Least Constraining Value heuristic if True.
        use_fc (bool): Use forward checking if True.
    
    Returns:
        tuple: A tuple containing:
            - solved_board (list of list of int): The solved Sudoku board.
            - execution_time (float): The running time in seconds.
            - trial_count (int): The number of assignment trials performed.
            - success (bool): True if the puzzle was solved, False otherwise.
    """
    sudoku = SudokuCSP(board)
    
    start_time = time.time()
    
    if algorithm == "simple":
        success = sudoku.backtrack()
    else:
        success = sudoku.smart_backtrack(use_mrv, use_lcv, use_fc)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return sudoku.board, execution_time, sudoku.trial_count, success

def print_board(board):
    """
    Print the formatted Sudoku board.
    
    Args:
        board (list of list of int): The 9x9 Sudoku board.
    """
    for i in range(9):
        if i % 3 == 0 and i > 0:
            print("-" * 21)
        row = ""
        for j in range(9):
            if j % 3 == 0 and j > 0:
                row += "| "
            row += str(board[i][j]) + " "
        print(row)

def main():
    """
    Main function to test the Sudoku solvers on four puzzles:
    Easy, Medium, Hard, and Evil. For each puzzle, both simple and smart
    backtracking methods are used, and performance metrics (execution time
    and trial count) are printed.
    """
    # Test puzzles
    easy = [
        [0, 3, 0, 0, 8, 0, 0, 0, 6],
        [5, 0, 0, 2, 9, 4, 7, 1, 0],
        [0, 0, 0, 3, 0, 0, 5, 0, 0],
        [0, 0, 5, 0, 1, 0, 8, 0, 4],
        [4, 2, 0, 8, 0, 5, 0, 3, 9],
        [1, 0, 8, 0, 3, 0, 6, 0, 0],
        [0, 0, 3, 0, 0, 7, 0, 0, 0],
        [0, 4, 1, 6, 5, 3, 0, 0, 2],
        [2, 0, 0, 0, 4, 0, 0, 6, 0]
    ]
    
    medium = [
        [3, 0, 8, 2, 9, 6, 0, 0, 0],
        [0, 4, 0, 0, 0, 8, 0, 0, 0],
        [5, 0, 2, 1, 0, 0, 0, 8, 7],
        [0, 1, 3, 0, 0, 0, 0, 0, 0],
        [7, 8, 0, 0, 0, 0, 0, 3, 5],
        [0, 0, 0, 0, 0, 0, 4, 1, 0],
        [1, 2, 0, 0, 0, 7, 8, 0, 3],
        [0, 0, 0, 8, 0, 0, 0, 2, 0],
        [0, 0, 0, 5, 4, 2, 1, 0, 6]
    ]
    
    hard = [
        [7, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 4, 1, 0, 2, 5, 0],
        [0, 1, 3, 0, 9, 5, 0, 0, 0],
        [8, 6, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 1, 0, 0, 0, 4, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 8, 6],
        [0, 0, 0, 8, 4, 0, 5, 3, 0],
        [0, 4, 2, 0, 3, 6, 0, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 9]
    ]
    
    evil = [
        [0, 6, 0, 8, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 6, 0, 0, 0, 9],
        [1, 0, 0, 0, 4, 3, 0, 6, 0],
        [0, 5, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 6, 0, 9, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 7, 0],
        [0, 1, 0, 4, 8, 0, 0, 0, 5],
        [8, 0, 0, 0, 1, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 4, 0]
    ]
    
    puzzles = {
        "Easy": easy,
        "Medium": medium,
        "Hard": hard,
        "Evil": evil
    }
    
    print("\n=== Simple Backtracking ===")
    for name, puzzle in puzzles.items():
        print(f"\nSolving {name} puzzle with simple backtracking:")
        print("\nOriginal board:")
        print_board(puzzle)
        
        solved_board, execution_time, trial_count, success = solve_sudoku(puzzle, algorithm="simple")
        
        print("\nSolved board:")
        print_board(solved_board)
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"Trial count: {trial_count}")
        print(f"Success: {success}")
    
    print("\n=== Smart Backtracking ===")
    for name, puzzle in puzzles.items():
        print(f"\nSolving {name} puzzle with smart backtracking (MRV, LCV, FC):")
        print("\nOriginal board:")
        print_board(puzzle)
        
        solved_board, execution_time, trial_count, success = solve_sudoku(puzzle, algorithm="smart")
        
        print("\nSolved board:")
        print_board(solved_board)
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"Trial count: {trial_count}")
        print(f"Success: {success}")

if __name__ == "__main__":
    main()
