
from PyQt5 import QtWidgets, uic, QtSvg, QtGui
import sys
from chessboard import display
import chess
import chess.svg
import time
import random
from stockfish import Stockfish
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QEvent, QSize, QTimer
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import QTimer

import cv2
import numpy as np
import pyautogui

from getboard import get_chessboard
from init_figures import extract_piece_images
from init_figures import extract_fen_from_image

screenshot_path = "/tmp/screenshot.png"
template_path = "./templ1.png"
startimg = "./startboard.png"
output_path = "./extracted_chessboard.png"
startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


# Extract piece images based on the FEN string
board_size, piece_images = extract_piece_images(startimg, startfen)

###############################################
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from anytree import NodeMixin


class ChessMoveNode(NodeMixin):  # Inherits from NodeMixin
    def __init__(self, ntype, name, info=None, data=None, parent=None, children=None):
        super().__init__()
        self.ntype = ntype
        self.name = name
        self.info = info
        self.data = data
        self.parent = parent
        if children:  # Children can be a list of nodes
            self.children = children



# Create Tree
root = ChessMoveNode("start","scandinavian", data="d2d4")
child1 = Node("a7a6", parent=root)
child2 = Node("h7h6", parent=root)

# Export to JSON
exporter = JsonExporter(indent=2, sort_keys=True)
with open('tree.json', 'w') as outfile:
    outfile.write(exporter.export(root))

# Import from JSON
importer = JsonImporter()
#with open('tree.json', 'r') as infile:
#    root = importer.import_(infile)

# Print Tree
for pre, _, node in RenderTree(root):
    print("%s%s" % (pre, node.name))
############################################

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('gui.ui', self)
        self.selected_square = None
        self.show()

        self.bBack.clicked.connect(self.backClicked )
        self.bForward.clicked.connect(self.forwardClicked)
        self.bSetFen.clicked.connect(self.processFen)
        self.bMove.clicked.connect(self.move)
        self.bGetPos.clicked.connect(self.getPos)

        self.stockfish = Stockfish(path="stockfish/stockfish",
                      depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 3})

        self.board = chess.Board()
        self.svgWidget = QtSvg.QSvgWidget('')
        self.svgWidget.setMinimumSize(QSize(400,400))
        self.svgWidget.setMaximumSize(QSize(400, 400))
        self.svgWidget.installEventFilter(self)
        self.ltBoard.addWidget(  self.svgWidget )
        self.cbTurnBoard.clicked.connect(self.processTurnBoard)
        self.show()
        self.lastMove = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateBoard)
        #self.timer.start(10)

    def processTurnBoard(self, ornt):
        self.updateBoard()
        QTimer.singleShot(1000, self.move)

    def backClicked(self):
        self.lastMove = None
        if len( self.board.move_stack ) > 0 and self.board.peek():  # Check if there is a move to undo
            self.board.pop()  # Undo the last move
        self.updateBoard()


    def forwardClicked(self):
        self.lastMove = None
        pass

    def processFen(self):
        self.lastMove = None
        fen = self.tFen.toPlainText()
        if len(fen) == 0:
            self.board.reset()
        else:
            self.board.set_fen(fen)
        self.updateBoard()

    def getPos(self):
        move_uci = self.eMove.text()
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(screenshot_path, screenshot_cv)
        chessboard_image, coordinates = get_chessboard( screenshot_path, template_path, output_path )
        player = "W"
        if self.cbTurnBoard.isChecked():
            player = "B"
        fen = extract_fen_from_image( output_path, board_size, piece_images, player )
        self.tFen.setPlainText( fen )

        self.updateBoard()

    def move(self):
        move_uci = self.eMove.text()

        if len(move_uci)>0 :
            move = chess.Move.from_uci(move_uci)
            self.makeMove(move)
        else:
            valid_fen = self.board.fen()  # 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'
            self.stockfish.set_fen_position(valid_fen)
            bestMove = self.stockfish.get_best_move()
            if bestMove is not None:
                self.tBookText.setText("Best {0}".format(bestMove))
                move = chess.Move.from_uci(bestMove)
                self.makeMove(move)
        self.updateBoard()

    def eventFilter(self, watched, event):
        if watched == self.svgWidget and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                x = event.x()  # x coordinate relative to svgWidget
                y = event.y()  # y coordinate relative to svgWidget
                # handle your logic here, like converting coordinates to square and making a move, etc.
                square = self.get_square_from_coordinates(x, y)
                self.handle_square_selection(square)
        return False

    def get_square_from_coordinates(self, x, y):
        w = self.svgWidget.width()
        square_size =  w // 8  # assuming square board and widget
        row = y // square_size
        col = x // square_size
        square = chess.square(col, 7 - row)  # 7 - row because chess squares are counted from the bottom
        return square

    def makeMove(self, move ):
        self.board.push(move)
        self.lastMove = move

    def handle_square_selection(self, square):
        if self.cbTurnBoard.isChecked():
            # Adjust the square if the board is flipped.
            rank = 7 - (square // 8)
            file = 7 - (square % 8)
            square = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and (piece.color == self.board.turn):
                # Select the square if it contains a piece of the current player.
                self.selected_square = square
                print(square)
        else:
            move = chess.Move(self.selected_square, square)
            print(move)
            if move in self.board.legal_moves:
                self.makeMove(move)
                self.updateBoard(True)
            self.selected_square = None  # reset selected square after move
            QTimer.singleShot(100, self.move)
            #self.move()

    def updateBoard(self, autoMove=False):
        arrows = []

        if self.lastMove is not None:
            arrows.append(chess.svg.Arrow(tail=self.lastMove.from_square, head=self.lastMove.to_square, color='yellow'))

        ornt = chess.WHITE
        if self.cbTurnBoard.isChecked():
            ornt = chess.BLACK


        valid_fen = self.board.fen()  # 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'
        self.stockfish.set_fen_position(valid_fen)
        best_move = self.stockfish.get_best_move()
        evaluation = self.stockfish.get_evaluation()

        # Check for mate
        if evaluation["type"] == "mate":
            moves_to_mate = [best_move]
            mate_in_n = evaluation["value"]
            for _ in range(abs(mate_in_n) - 1):
                self.stockfish.make_moves_from_current_position([best_move])
                best_move = self.stockfish.get_best_move()
                moves_to_mate.append(best_move)

            mate_in_n = evaluation["value"]
            if mate_in_n > 0:
                self.tMovesToMat.setText( f"Mate for White in {mate_in_n} moves." )
                self.tMovesToMat.append( ", ".join(moves_to_mate) )
            elif mate_in_n < 0:
                self.tMovesToMat.setText(f"Mate for Black in {-mate_in_n} moves.")
                self.tMovesToMat.append( ", ".join(moves_to_mate) )
        else:
            self.tMovesToMat.setText(f"")

        arrows = self.get_attack_arrows()

        valid_fen = self.board.fen()  # 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'
        self.stockfish.set_fen_position(valid_fen)
        bestMove = self.stockfish.get_best_move()
        if bestMove is not None:
            self.tBookText.setText("Best {0}".format(bestMove))
            move = chess.Move.from_uci(bestMove)

            if self.cbOppAutoMove.isChecked():
                move = chess.Move.from_uci(bestMove)

            # Extracting the from_square and to_square
            square = move.from_square
            target_square = move.to_square
            arrows.append(chess.svg.Arrow(square, target_square, color='lightblue'))

        board_svg = chess.svg.board(board=self.board, arrows=arrows, orientation=ornt, lastmove=self.lastMove)
        self.svgWidget.load(board_svg.encode('UTF-8'))




    def is_piece_unprotected(self, square):
        piece = self.board.piece_at(square)
        if piece is None:
            return False

        attackers = self.board.attackers(piece.color, square)
        return not any(self.board.piece_at(attacker_square).color == piece.color for attacker_square in attackers)

    def get_piece_value(self, piece):
        values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        return values[piece.symbol().upper()]

    def get_attack_arrows1(self):
        arrows = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color = piece.color

                for target_square in self.board.attacks(square):
                    target_piece = self.board.piece_at(target_square)
                    if target_piece and target_piece.color != color:
                        # Check if the target piece is unprotected or of lesser value
                        if self.is_piece_unprotected(target_square) or self.get_piece_value(
                                piece) < self.get_piece_value(target_piece):
                            arrow_color = 'red' if color == self.board.turn else 'blue'
                            arrows.append(chess.svg.Arrow(square, target_square, color=arrow_color))

        return arrows

    def get_attack_arrows(self):
        arrows = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                color = piece.color

                # Arrows for attacks by the player whose turn it is, to opponent's pieces.
                if color == self.board.turn:
                    for target_square in self.board.attacks(square):
                        target_piece = self.board.piece_at(target_square)
                        if self.is_piece_unprotected(target_square):
                            if target_piece and target_piece.color != color:  # Only if there is an opponent's piece at the target square
                                arrow = chess.svg.Arrow(square, target_square, color='red')
                                arrows.append(arrow)
                else:
                    # Arrows for opponent's pieces that are attacking the player whose turn it is.
                    for target_square in self.board.attacks(square):
                        target_piece = self.board.piece_at(target_square)
                        if self.is_piece_unprotected(target_square):
                            if target_piece and target_piece.color != color:  # Only if there is an opponent's piece at the target square
                                arrow = chess.svg.Arrow(square, target_square, color='blue')
                                arrows.append(arrow)

        return arrows


app = QtWidgets.QApplication(sys.argv)
window = Ui()

window.updateBoard()
app.exec_()