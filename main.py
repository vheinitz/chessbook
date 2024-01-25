from chessboard import display
import chess
import time
import random

board = chess.Board()

valid_fen = board.fen() #'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'

game_board = display.start()

while board.legal_moves.count() > 0 :

    #moves = list(board.legal_moves)
    #n = random.randrange(len(moves))
    #m = moves[n]
    #board.push_san(str(m))
    #print ( board.fen() )
    #valid_fen = board.fen()

    display.check_for_quit()
    display.update(valid_fen, game_board)

    # board flip interface
    if not game_board.flipped:
        display.flip(game_board)

    if board.is_repetition():
        break

    time.sleep(.5)