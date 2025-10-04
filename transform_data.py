import json


def transform_data(input_file: str) -> list[list[float]]:
    with open(input_file, "r") as f:
        data = json.load(f)
    # Expects {PlayerName: {market_value: {"it": [values]}, ...}}
    # Need to transform it into [[player1_values], [player2_values], ...]
    transformed_data = []
    for player, metrics in data.items():
        player_values = []
        market_values = metrics.get("market_value", {}).get("it", [])
        if market_values:
            player_values.extend(
                point.get("mv")
                for point in market_values
                if point.get("mv") != 0
            )
        if len(player_values) < 365:
            continue
        elif all(v == 500000 for v in player_values):
            continue
        # We filter players with less than 365 days of data, only 0 or 500000 values
        transformed_data.append(player_values)
    return transformed_data
