import sys
import os
import traceback

# Workaround: cv2 sets QT_QPA_PLATFORM_PLUGIN_PATH to its own bundled Qt plugins
# which are incompatible with PyQt5. Unset it so PyQt5 uses its own.
import cv2
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

from PyQt5 import QtWidgets, uic, QtSvg, QtCore
from PyQt5.QtCore import Qt, QEvent, QSize, QTimer, QSettings
from PyQt5.QtSvg import QSvgWidget
import chess
import chess.svg
from stockfish import Stockfish

from getboard import get_chessboard
from init_figures import extract_piece_images
from init_figures import extract_fen_from_image
from screengrab import grab_screen

screenshot_path = "/tmp/screenshot.png"
template_path = "./templ1.png"
startimg = "./startboard.png"
output_path = "./extracted_chessboard.png"
calib_path = "./calibration_board.png"

# Position shown in startboard.png.  It is NOT the initial position: the extra
# queens and kings on c6/d6 and c3/d3 are there so that every piece type has a
# template on a light *and* on a dark square.  This string has to describe
# startboard.png exactly, otherwise the templates end up with wrong labels.
startfen = "rnbqkbnr/pppppppp/2qk4/8/8/2QK4/PPPPPPPP/RNBQKBNR"

# The real initial position - used to check whether re-calibrating on the
# captured board is safe.
initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


# Extract piece images from the reference board (used as fallback)
board_size, piece_images = extract_piece_images(startimg, startfen)

# Calibrated templates extracted from the actual captured board
calibrated_templates = None

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


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('gui.ui', self)
        self.selected_square = None

        # Window position and size of the last session (~/.config/chessbook)
        self.settings = QSettings("chessbook", "chessbook")
        self.restoreWindowGeometry()

        self.show()

        self.bBack.clicked.connect(self.backClicked )
        self.bForward.clicked.connect(self.forwardClicked)
        self.bSetFen.clicked.connect(self.processFen)
        self.bMove.clicked.connect(self.onMove)
        self.bAddVar.clicked.connect(self.addVar)
        
        self.bGetPos.clicked.connect(self.getPos)

        self.startEngine()

        self.board = chess.Board()
        self.svgWidget = QtSvg.QSvgWidget()
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

    def startEngine(self):
        """Starts (or restarts) the engine process."""
        self.stockfish = Stockfish(path="stockfish/stockfish",
                      depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 1})

    def analysePosition(self, board):
        """
        Asks the engine for the best move and the evaluation of `board`.

        Returns (None, None) when the position cannot be analysed.  Illegal
        positions are normal here - the recognition can miss a king, and
        flipping the side to move leaves the opponent in check - and Stockfish
        does not survive them: it dies on the spot and takes every following
        request with it.  So the position is validated first, and if the engine
        crashes anyway it is restarted instead of bringing the app down.
        """
        if not board.is_valid():
            return None, None

        try:
            self.stockfish.set_fen_position(board.fen())
            return self.stockfish.get_best_move(), self.stockfish.get_evaluation()
        except Exception as exc:
            print(f"engine: {exc} - restarting")
            self.startEngine()
            return None, None

    def mateSequence(self, best_move, mate_in_n):
        """Follows the engine's mate line, starting from its current position."""
        moves = [best_move]
        try:
            for _ in range(abs(mate_in_n) - 1):
                self.stockfish.make_moves_from_current_position([moves[-1]])
                next_move = self.stockfish.get_best_move()
                if next_move is None:
                    break
                moves.append(next_move)
        except Exception as exc:
            print(f"engine: {exc} - restarting")
            self.startEngine()
        return moves

    def describePosition(self, board):
        """Explains why a position could not be analysed."""
        if board.is_valid():
            return "No move (mate, stalemate or engine unavailable)"

        status = board.status()
        problems = [flag.name.replace("_", " ").lower()
                    for flag in chess.Status
                    if flag.value and status & flag.value]
        return "Position not playable: " + ", ".join(problems)

    def restoreWindowGeometry(self):
        """Puts the window back where it was when it was closed last time."""
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def saveWindowGeometry(self):
        """Stores position, size and maximized state of the window."""
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.sync()

    def closeEvent(self, event):
        self.saveWindowGeometry()
        super(Ui, self).closeEvent(event)

    def processTurnBoard(self, ornt):
        self.updateBoard()
        QTimer.singleShot(1000, self.onMove)

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
        fen = self.tFen.toPlainText().strip()
        if len(fen) == 0:
            self.board.reset()
        else:
            try:
                self.board.set_fen(fen)
            except ValueError as exc:
                # Hand-typed FENs and, in rare cases, the recognition can
                # produce something python-chess refuses to parse.
                self.tBookText.setText(f"Invalid FEN: {exc}")
                return
        self.updateBoard()

    def getPos(self):
        global piece_images, calibrated_templates
        
        move_uci = self.eMove.text()

        # Take a screenshot (silently - no flash, no shutter sound)
        grab_screen(screenshot_path)

        chessboard_image, coordinates = get_chessboard(board_size, screenshot_path, template_path, output_path )
        player = "W"
        if self.cbTurnBoard.isChecked():
            player = "B"

        # Recognize with the templates from the reference board, plus the
        # calibrated ones once we have them.
        templates = calibrated_templates if calibrated_templates is not None else piece_images
        fen = extract_fen_from_image( output_path, templates, player )

        # Calibration: templates taken from the board on screen match its
        # rendering exactly.  It is only safe when we know which piece sits on
        # which square, i.e. when the captured board shows the initial position
        # with white at the bottom.  Calibrating on any other position labels
        # the templates wrongly - a knight would land in the pawn template list
        # and every knight from then on would be read as a pawn.
        if calibrated_templates is None and player == "W" and fen == initial_fen:
            calib_size, calib = extract_piece_images(output_path, fen)
            merged = {name: list(images) for name, images in piece_images.items()}
            for name, images in calib.items():
                merged[name].extend(images)
            calibrated_templates = merged
            cv2.imwrite(calib_path, cv2.imread(output_path))
            print(f"Calibrated templates from captured board ({calib_size})")

        self.tFen.setPlainText( fen )
        self.processFen()

        #self.updateBoard()

    def onMove(self):
        move_uci = self.eMove.text().strip()

        if len(move_uci)>0 :
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                self.tBookText.setText(f"Not a move: {move_uci}")
                return
            if move not in self.board.legal_moves:
                self.tBookText.setText(f"Illegal move: {move_uci}")
                return
            self.makeMove(move)
        else:
            bestMove, _ = self.analysePosition(self.board)
            if bestMove is not None:
                self.tBookText.setText("Best {0}".format(bestMove))
                move = chess.Move.from_uci(bestMove)
                self.makeMove(move)
            else:
                self.tBookText.setText(self.describePosition(self.board))
    
    def addVar(self):
        valid_fen = self.board.fen()  # 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2'
        print (valid_fen)
 

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
        #if self.cbOppAutoMove.isChecked():
            self.board.push(move)
            self.lastMove = move
            self.updateBoard(True)

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
            self.selected_square = None  # reset selected square after move
            QTimer.singleShot(100, self.onMove)
            #self.onMove()

    def updateBoard(self, autoMove=False):
        arrows = []

        if self.lastMove is not None:
            arrows.append(chess.svg.Arrow(tail=self.lastMove.from_square, head=self.lastMove.to_square, color='yellow'))

        ornt = chess.WHITE
        if self.cbTurnBoard.isChecked():
            ornt = chess.BLACK


        # When the board is turned, analyse from Black's point of view.  Doing
        # that by flipping the side to move can leave the opponent in check,
        # which is an illegal position - fall back to the real one then.
        analysis_board = self.board.copy()
        if self.cbTurnBoard.isChecked():
            analysis_board.turn = chess.BLACK
        if not analysis_board.is_valid():
            analysis_board = self.board

        best_move, evaluation = self.analysePosition(analysis_board)

        # Check for mate
        if best_move is not None and evaluation["type"] == "mate":
            mate_in_n = evaluation["value"]
            moves_to_mate = self.mateSequence(best_move, mate_in_n)

            if mate_in_n > 0:
                self.tMovesToMat.setText( f"Mate for White in {mate_in_n} moves." )
                self.tMovesToMat.append( ", ".join(moves_to_mate) )
            elif mate_in_n < 0:
                self.tMovesToMat.setText(f"Mate for Black in {-mate_in_n} moves.")
                self.tMovesToMat.append( ", ".join(moves_to_mate) )
        else:
            self.tMovesToMat.setText(f"")

        arrows = self.get_attack_arrows()

        if best_move is not None:
            self.tBookText.setText("Best {0}".format(best_move))
            move = chess.Move.from_uci(best_move)

            # Extracting the from_square and to_square
            square = move.from_square
            target_square = move.to_square
            arrows.append(chess.svg.Arrow(square, target_square, color='lightblue'))
        else:
            self.tBookText.setText(self.describePosition(analysis_board))

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


def excepthook(exc_type, exc_value, exc_traceback):
    """
    Prints unhandled exceptions instead of letting the application die.

    Since PyQt 5.5 an unhandled exception inside a slot ends in qFatal(), which
    aborts the process ("Aborted (core dumped)") - one bad move or FEN would
    close the window.  An own hook takes precedence over that.
    """
    traceback.print_exception(exc_type, exc_value, exc_traceback)


def qtMessageHandler(mode, context, message):
    """
    Qt's own log output.

    On a GNOME Wayland session Qt prints "Ignoring XDG_SESSION_TYPE=wayland on
    Gnome" on every start, although it then simply runs on XWayland as it
    should.  Drop that one line and pass everything else through.
    """
    if "Ignoring XDG_SESSION_TYPE" in message:
        return
    sys.stderr.write(message + "\n")


sys.excepthook = excepthook
QtCore.qInstallMessageHandler(qtMessageHandler)

app = QtWidgets.QApplication(sys.argv)
window = Ui()

window.updateBoard()
app.exec_()