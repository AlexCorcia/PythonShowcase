from pathlib import Path
import chess.pgn
import pandas as pd


def parse_result(result: str) -> str:
    if result == "1-0":
        return "white_win"
    if result == "0-1":
        return "black_win"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def parse_player_result(result: str, player_color: str | None) -> str:
    if player_color is None:
        return "unknown"

    if result == "1-0":
        return "win" if player_color == "white" else "loss"

    if result == "0-1":
        return "win" if player_color == "black" else "loss"

    if result == "1/2-1/2":
        return "draw"

    return "unknown"


def extract_year(date: str) -> int | None:
    """
    Convierte fechas tipo:
    - '1975.04.12'
    - '1961.??.??'
    - '????.??.??'

    En año entero.
    """
    if not date or date == "????.??.??":
        return None

    year = date.split(".")[0]

    try:
        return int(year)
    except ValueError:
        return None


def clean_elo(elo: str | None) -> int | None:
    """
    Convierte Elo a entero.
    Si está vacío o no existe, devuelve None.
    """
    if elo is None:
        return None

    elo = str(elo).strip()

    if elo == "" or elo == "?":
        return None

    try:
        return int(elo)
    except ValueError:
        return None


def detect_player_perspective(
    white: str,
    black: str,
    main_player: str | None,
) -> tuple[str | None, str | None]:
    """
    Determina si el jugador principal juega con blancas o negras.

    Devuelve:
    - player_color
    - opponent
    """
    if main_player is None:
        return None, None

    main_player_norm = main_player.lower().strip()
    white_norm = white.lower().strip()
    black_norm = black.lower().strip()

    if main_player_norm in white_norm:
        return "white", black

    if main_player_norm in black_norm:
        return "black", white

    return None, None


def count_basic_features(game: chess.pgn.Game) -> dict:
    """
    Extrae features básicas de una partida completa.

    Estas features todavía NO dependen del jugador principal,
    sino de la partida en general.
    """
    board = game.board()

    num_plies = 0
    captures = 0
    checks = 0
    castles = 0
    promotions = 0

    white_castled = False
    black_castled = False

    for move in game.mainline_moves():
        moving_color = board.turn

        if board.is_capture(move):
            captures += 1

        if board.is_castling(move):
            castles += 1

            if moving_color == chess.WHITE:
                white_castled = True
            else:
                black_castled = True

        if move.promotion is not None:
            promotions += 1

        board.push(move)

        if board.is_check():
            checks += 1

        num_plies += 1

    return {
        "num_plies": num_plies,
        "num_moves": num_plies / 2,
        "captures": captures,
        "checks": checks,
        "castles": castles,
        "promotions": promotions,
        "white_castled": white_castled,
        "black_castled": black_castled,
    }


def count_player_features(
    game: chess.pgn.Game,
    player_color: str | None,
) -> dict:
    """
    Extrae features desde la perspectiva del jugador principal.

    Esto es importante porque el jugador puede estar jugando
    con blancas o con negras.
    """
    if player_color is None:
        return {
            "player_captures": None,
            "player_checks": None,
            "player_castled": None,
            "player_promotions": None,
            "opponent_captures": None,
            "opponent_checks": None,
        }

    player_is_white = player_color == "white"

    board = game.board()

    player_captures = 0
    player_checks = 0
    player_castled = False
    player_promotions = 0

    opponent_captures = 0
    opponent_checks = 0

    for move in game.mainline_moves():
        moving_color = board.turn
        is_player_move = (moving_color == chess.WHITE and player_is_white) or (
            moving_color == chess.BLACK and not player_is_white
        )

        if board.is_capture(move):
            if is_player_move:
                player_captures += 1
            else:
                opponent_captures += 1

        if board.is_castling(move) and is_player_move:
            player_castled = True

        if move.promotion is not None and is_player_move:
            player_promotions += 1

        board.push(move)

        if board.is_check():
            if is_player_move:
                player_checks += 1
            else:
                opponent_checks += 1

    return {
        "player_captures": player_captures,
        "player_checks": player_checks,
        "player_castled": player_castled,
        "player_promotions": player_promotions,
        "opponent_captures": opponent_captures,
        "opponent_checks": opponent_checks,
    }


def parse_pgn_file(
    pgn_path: str | Path,
    main_player: str | None = None,
    min_plies: int = 20,
) -> pd.DataFrame:
    """
    Lee un archivo PGN y lo convierte en un DataFrame limpio.

    Parámetros:
    - pgn_path: ruta al archivo PGN.
    - main_player: jugador principal del archivo. Ej: 'Karpov'.
    - min_plies: mínimo de medias jugadas para aceptar una partida.
    """
    pgn_path = Path(pgn_path)

    rows = []
    total_games = 0
    skipped_games = 0
    skipped_not_player = 0
    skipped_short = 0
    skipped_parse_error = 0

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            total_games += 1

            headers = game.headers

            white = headers.get("White", "").strip()
            black = headers.get("Black", "").strip()
            result = headers.get("Result", "").strip()

            player_color, opponent = detect_player_perspective(
                white=white,
                black=black,
                main_player=main_player,
            )

            if main_player is not None and player_color is None:
                skipped_games += 1
                skipped_not_player += 1
                continue

            try:
                basic_features = count_basic_features(game)
                player_features = count_player_features(game, player_color)
            except Exception:
                skipped_games += 1
                skipped_parse_error += 1
                continue

            if basic_features["num_plies"] < min_plies:
                skipped_games += 1
                skipped_short += 1
                continue

            row = {
                "source_file": pgn_path.name,
                # Identificación
                "main_player": main_player,
                "player_color": player_color,
                "opponent": opponent,
                # Headers PGN
                "event": headers.get("Event", ""),
                "site": headers.get("Site", ""),
                "date": headers.get("Date", ""),
                "year": extract_year(headers.get("Date", "")),
                "round": headers.get("Round", ""),
                # Jugadores
                "white": white,
                "black": black,
                "white_elo": clean_elo(headers.get("WhiteElo")),
                "black_elo": clean_elo(headers.get("BlackElo")),
                # Resultado
                "result": result,
                "result_label": parse_result(result),
                "player_result": parse_player_result(result, player_color),
                # Apertura
                "eco": headers.get("ECO", None),
                "opening": headers.get("Opening", None),
            }

            row.update(basic_features)
            row.update(player_features)

            rows.append(row)

    df = pd.DataFrame(rows)

    print("=" * 60)
    print(f"Archivo: {pgn_path}")
    print(f"Partidas leídas: {total_games}")
    print(f"Partidas válidas: {len(df)}")
    print(f"Partidas omitidas: {skipped_games}")
    print(f"  - No aparece el jugador principal: {skipped_not_player}")
    print(f"  - Demasiado cortas: {skipped_short}")
    print(f"  - Error de parseo: {skipped_parse_error}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    input_path = "app/data/raw/Karpov.pgn"
    output_path = "app/data/processed/karpov_parsed.csv"

    df = parse_pgn_file(
        pgn_path=input_path,
        main_player="Karpov",
        min_plies=20,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.head())
    print(f"CSV guardado en: {output_path}")
