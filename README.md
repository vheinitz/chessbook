# Chessbook

Reads a chess position **directly off your screen** and analyses it with a bundled
Stockfish 16.

Open a game in your browser, press **GetPos**, and Chessbook grabs a screenshot,
finds the board in it, recognises every piece, turns the position into a FEN and
shows you the engine's best move - plus arrows for hanging pieces and a mate-in-N
readout.

```
browser board -> screenshot -> board detection -> piece recognition -> FEN -> Stockfish
```

Screenshots are taken silently: no white flash, no camera-shutter sound. The piece
recognition is tuned for the green/beige board theme in `startboard.png` (lichess'
default); other themes need a new reference image or a calibration run, see below.

## How it works

| Step | File | What happens |
| --- | --- | --- |
| 1. Capture | `screengrab.py` | Whole-screen grab through GNOME Shell's D-Bus API (~0.17 s, no flash, no sound), with X11 and `gnome-screenshot` fallbacks. |
| 2. Locate board | `getboard.py` | HSV histogram back-projection against `templ1.png` (a small sample of the board colours), largest contour wins, cropped and resized to the reference size. |
| 3. Recognise pieces | `init_figures.py` | Every square is cropped, the figure is separated from the square background by adaptive thresholding, then matched against template figures by Hu moments plus a brightness check for the piece colour. |
| 4. Analyse | `gui.py` | `python-chess` for the board model and SVG rendering, Stockfish for best move, evaluation and mate detection. |

### Where the piece templates come from

`startboard.png` is the reference board. It is deliberately **not** the initial
position: it carries two extra queens and kings on c6/d6 and c3/d3 so that every
piece type has a template on a light *and* on a dark square. The FEN describing it
lives in `gui.py` as `startfen` - **if you replace `startboard.png`, that string has
to match the new image exactly**, otherwise the templates get the wrong labels and
recognition silently degrades (knights read as pawns, and so on).

On top of that, Chessbook re-calibrates from your own screen: the first time
**GetPos** sees the *initial position* with white at the bottom, it takes fresh
templates from that capture (saved as `calibration_board.png`) and merges them with
the reference ones, so they match your board theme's exact rendering. Calibration is
skipped for any other position - templates taken from an unknown position would be
mislabeled.

## Requirements

- Linux with a GNOME session (Wayland or X11). Other desktops work, but the
  screenshot falls back to a method that flashes and beeps.
- Python 3.10 or newer
- A C++ compiler and `make` for the engine
- A chessboard visible on screen, in a theme similar to `startboard.png`

## Install

```bash
sudo apt install build-essential python3-venv python3-gi   # Debian/Ubuntu
git clone git@github.com:vheinitz/chessbook.git
cd chessbook
./build.sh
```

`build.sh` compiles the bundled Stockfish (picking the instruction set for your CPU),
copies the binary to `stockfish/stockfish` where the app expects it, and installs the
Python packages from `requirements.txt`.

```bash
./build.sh --venv           # create ./.venv and install into it
./build.sh --engine-only    # only compile Stockfish
./build.sh --python-only    # only install the Python packages
./build.sh --clean          # recompile the engine from scratch
ARCH=x86-64-avx2 ./build.sh # override the detected architecture
```

`python3-gi` (PyGObject) is what makes the screenshots silent. Without it everything
still works, but every capture flashes the screen and plays the shutter sound.
`--venv` creates the virtualenv with `--system-site-packages` so the distribution's
`python3-gi` stays visible.

## Run

```bash
python3 gui.py        # or .venv/bin/python gui.py
```

Run it from the repository directory - `gui.ui`, `startboard.png`, `templ1.png` and
`stockfish/stockfish` are all loaded by relative path.

## Using it

| Control | What it does |
| --- | --- |
| **GetPos** | Screenshot -> find board -> recognise pieces -> fill the FEN field and show the position |
| **Turn board** | The board on screen is seen from Black's side (also flips the analysis board) |
| **Set** | Apply the FEN from the text field to the board (empty field resets to the initial position) |
| **Move** | Play the move from the input field, or, if it is empty, play Stockfish's best move |
| **&lt;** | Take back the last move |
| **&gt;** | Not implemented yet |
| **AddVar** | Print the current FEN to the console |
| Click board | Click a piece then a target square to play a move |

The board shows a light-blue arrow for the engine's best move, red/blue arrows for
unprotected pieces, and the *Detected check-mat* box lists the forced mate sequence
when there is one.

## Repository layout

```
gui.py                  the application: Qt window, board model, engine, calibration
screengrab.py           silent screen capture with backend fallbacks
getboard.py             finds the chessboard inside a screenshot
init_figures.py         piece templates and FEN recognition
gui.ui                  Qt Designer layout
startboard.png          reference board the piece templates are cut from
templ1.png              colour sample used to locate the board
build.sh                builds Stockfish and installs the Python packages
requirements.txt        Python dependencies
stockfish/              vendored Stockfish 16 source and NNUE network
```

Run the recognition on its own, without the GUI:

```bash
python3 init_figures.py     # prints template counts and the FEN of extracted_chessboard.png
python3 screengrab.py       # captures the screen to /tmp/screengrab_test.png
```

## Troubleshooting

**The wrong region is cropped instead of the board.** `templ1.png` is a small sample
of your board's colours. Crop a piece-free patch of your board (a few squares) and
save it as `templ1.png`.

**Pieces are recognised wrongly.** Your board theme probably differs too much from
`startboard.png`. Easiest fix: open the initial position in the browser, press
**GetPos** once so Chessbook calibrates on your theme, and continue from there. To
change the reference permanently, replace `startboard.png` and update `startfen` in
`gui.py` accordingly (see above). `extracted_chessboard.png` always holds the last
cropped board - a good place to look when something goes wrong.

**Every screenshot flashes and beeps.** PyGObject is missing:
`sudo apt install python3-gi`. `python3 screengrab.py` prints which backend is in use.

**`Stockfish` fails to start.** The binary must be at `stockfish/stockfish`; run
`./build.sh --engine-only`.

## Licensing

`stockfish/` contains a copy of [Stockfish](https://github.com/official-stockfish/Stockfish),
which is licensed under the **GNU GPL v3** - see `stockfish/Copying.txt`. Chessbook
itself has no license file yet; add one before distributing.
